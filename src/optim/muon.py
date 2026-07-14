"""Muon optimizer (MomentUm Orthogonalized by Newton-Schulz).

Reference: Jordan et al., "Muon: An optimizer for hidden layers in neural
networks" (2024). Muon applies SGD-momentum and then orthogonalizes each 2D
weight-matrix update via a quintic Newton-Schulz iteration. It is meant ONLY
for the 2D hidden weights of the network body; embeddings, output/classifier
head, norms, biases and any 1D parameters should stay on AdamW.

Use `build_hybrid_optimizer(model, cfg)` to get the correct split for the
vendored GPT-BERT model automatically.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


@torch.no_grad()
def newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5,
                                eps: float = 1e-7) -> torch.Tensor:
    """Approximate UV^T for G = USV^T via quintic Newton-Schulz iteration.

    Coefficients from the reference implementation (Keller Jordan). Operates
    in bfloat16 for speed when on CUDA; handles rectangular matrices by
    transposing so rows <= cols.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.is_cuda else G.float()
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon for 2D parameters. Single-device implementation.

    lr semantics follow the reference: the orthogonalized update is scaled by
    sqrt(max(1, rows/cols)) to keep the update RMS roughly constant across
    matrix shapes.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5,
                 weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon requires 2D params; got shape {tuple(p.shape)}. "
                        "Route non-2D params to AdamW.")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)
                update = g.add(buf, alpha=mu) if group["nesterov"] else buf
                update = newton_schulz_orthogonalize(update,
                                                     steps=group["ns_steps"])
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr * scale)
        return loss


def split_params_for_muon(model) -> tuple[list, list]:
    """(muon_params, adamw_params) for the vendored GPT-BERT model.

    Muon: 2D weights inside the transformer body (attention + FFN linears).
    AdamW: everything else — embeddings (word + relative position), all
    LayerNorm/bias/1D params, and the classifier head (which is weight-tied
    to the word embedding in GPT-BERT, so it must NOT go to Muon anyway).
    """
    muon, adamw = [], []
    tied_ids = set()
    # weight tying: classifier's last Linear shares the embedding weight
    if hasattr(model, "embedding"):
        tied_ids.add(id(model.embedding.word_embedding.weight))
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        in_body = name.startswith("transformer.")
        if in_body and p.ndim == 2 and id(p) not in tied_ids:
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw


class HybridOptimizer:
    """Muon(body 2D) + AdamW(rest) behind a single Optimizer-like facade."""

    def __init__(self, model, muon_lr=0.02, muon_momentum=0.95,
                 adamw_lr=1e-2, adamw_betas=(0.9, 0.98), adamw_eps=1e-8,
                 weight_decay=0.1):
        muon_params, adamw_params = split_params_for_muon(model)
        self.muon = Muon(muon_params, lr=muon_lr, momentum=muon_momentum,
                         weight_decay=weight_decay)
        self.adamw = torch.optim.AdamW(adamw_params, lr=adamw_lr,
                                       betas=adamw_betas, eps=adamw_eps,
                                       weight_decay=weight_decay)
        self._opts = [self.muon, self.adamw]

    @property
    def param_groups(self):
        return [g for o in self._opts for g in o.param_groups]

    def step(self):
        for o in self._opts:
            o.step()

    def zero_grad(self, set_to_none: bool = True):
        for o in self._opts:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"muon": self.muon.state_dict(),
                "adamw": self.adamw.state_dict()}

    def load_state_dict(self, s):
        self.muon.load_state_dict(s["muon"])
        self.adamw.load_state_dict(s["adamw"])
