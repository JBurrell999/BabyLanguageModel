"""ObjectiveMixer — turns a schedule value p_mask(t) into a training batch.

Design
------
GPT-BERT's `Bert.forward(input_ids, attention_mask, labels, num_masked, ratio)`
already supports a heterogeneous batch: the first `num_masked` examples are
scored with the MNTP head/loss, the rest with the causal loss, and `ratio`
weights the two losses. The official training code never exploits this —
it freezes the split by dedicating whole GPUs to one objective
(train_100m.py, rank-based dataset assignment).

The mixer exploits it. Each optimizer step:

    n_masked = round(B * p_mask(t))            # batch composition
    batch    = [n_masked from MaskedDataset] ++ [B - n_masked from CausalDataset]
    ratio    = n_masked / B  (default)          # loss weight tracks composition

Two independent knobs fall out of this, and we expose both because their
dissociation is an ablation in its own right:

  * composition_p — what fraction of examples are MNTP-formatted;
  * loss weight   — how the two loss terms are combined. `ratio_mode`:
        "composition" (default): weight == realized composition, so the
            per-token gradient contribution of each objective is proportional
            to its share of the batch (matches the multi-GPU reference
            semantics, where each GPU averages equally);
        "fixed:<x>": constant weight x regardless of composition;
        "schedule": weight follows p_mask(t) exactly even when rounding
            makes the realized composition differ.

Data-exposure note: the two source datasets stream independently, so under a
dynamic schedule the masked and causal streams advance at different rates.
Both are full shuffled copies of the same corpus, so the *distribution* of
text is identical across arms; total exposure is governed by the
ExposureCounter in src/pretrain/exposure.py, which is what the BabyLM epoch
cap regulates.

Batching mechanics: we keep a prefetch buffer per stream fed by an ordinary
DataLoader (workers > 0 fine), and slice `n_masked` / `B - n_masked` items per
step, carrying remainders forward. Streams re-shuffle and continue when
exhausted (epoch boundaries are tracked, not enforced, here).
"""

from __future__ import annotations

from collections import deque
from typing import Iterator, Optional

import torch
from torch.utils.data import DataLoader


def _round_composition(batch_size: int, p_mask: float) -> int:
    """Deterministic round of B*p to an integer count, clamped to [0, B]."""
    n = int(round(batch_size * p_mask))
    return min(max(n, 0), batch_size)


class _Stream:
    """Endless per-example stream over a Dataset with a prefetch buffer."""

    def __init__(self, dataset, num_workers: int = 2, prefetch_batches: int = 64,
                 seed: int = 0):
        self.dataset = dataset
        self.num_workers = num_workers
        self.prefetch = prefetch_batches
        self.seed = seed
        self.epoch = 0
        self.buffer: deque = deque()
        self._iter: Optional[Iterator] = None

    def _make_loader(self) -> DataLoader:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        return DataLoader(
            self.dataset,
            batch_size=self.prefetch,
            shuffle=True,
            generator=g,
            num_workers=self.num_workers,
            collate_fn=lambda items: items,   # keep per-example tuples
            drop_last=False,
            persistent_workers=False,
        )

    def set_global_step(self, step: int) -> None:
        # GPT-BERT datasets use this for their internal mask_p ramp.
        if hasattr(self.dataset, "set_global_step"):
            self.dataset.set_global_step(step)

    def take(self, n: int) -> list:
        out = []
        while len(out) < n:
            if not self.buffer:
                if self._iter is None:
                    self._iter = iter(self._make_loader())
                try:
                    self.buffer.extend(next(self._iter))
                except StopIteration:
                    self.epoch += 1
                    self._iter = iter(self._make_loader())
            else:
                out.append(self.buffer.popleft())
        return out


class ObjectiveMixer:
    """Compose mixed MNTP/causal batches according to a schedule value.

    Yields dicts ready for GPT-BERT's forward. NOTE the model is
    sequence-first (train_100m.py transposes after loading; MaskClassifier
    tensor_splits on dim=1 = batch), so:
        input_ids      [L, B]     (long) — masked examples in the first
                                   `num_masked` batch columns
        target_ids     [L, B]     (long, -100 = unsupervised)
        attention_mask [B, L, L]  (bool; model unsqueezes to [B, 1, L, L])
        num_masked     int        (# MNTP examples)
        ratio          float      (loss weight for the masked term)
        mask_p         float      (mean realized mask prob over MNTP examples)
    """

    def __init__(self, masked_dataset, causal_dataset, batch_size: int,
                 ratio_mode: str = "composition",
                 num_workers: int = 2, seed: int = 0):
        self.batch_size = batch_size
        self.ratio_mode = ratio_mode
        self.masked = _Stream(masked_dataset, num_workers=num_workers, seed=seed)
        self.causal = _Stream(causal_dataset, num_workers=num_workers,
                              seed=seed + 1)

    # -- ratio semantics ----------------------------------------------------
    def _loss_ratio(self, p_schedule: float, n_masked: int) -> float:
        if self.ratio_mode == "composition":
            return n_masked / self.batch_size
        if self.ratio_mode == "schedule":
            return p_schedule
        if self.ratio_mode.startswith("fixed:"):
            return float(self.ratio_mode.split(":", 1)[1])
        raise ValueError(f"unknown ratio_mode {self.ratio_mode!r}")

    # -- main entry ----------------------------------------------------------
    def next_batch(self, p_mask: float, global_step: int, device=None) -> dict:
        self.masked.set_global_step(global_step)
        self.causal.set_global_step(global_step)

        n_masked = _round_composition(self.batch_size, p_mask)
        items = (self.masked.take(n_masked)
                 + self.causal.take(self.batch_size - n_masked))

        # Each item: (input_ids [L], target_ids [L], attention_mask [L, L], mask_p)
        # Stack batch-first, then transpose ids to the model's seq-first layout.
        input_ids = torch.stack([it[0] for it in items]).t().contiguous()
        target_ids = torch.stack([it[1] for it in items]).t().contiguous()
        attention_mask = torch.stack([it[2] for it in items])
        if n_masked > 0:
            mask_ps = [float(it[3]) for it in items[:n_masked]]
            mask_p = sum(mask_ps) / len(mask_ps)
        else:
            mask_p = 0.0

        batch = {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask,
            "num_masked": n_masked,
            "ratio": self._loss_ratio(p_mask, n_masked),
            "mask_p": mask_p,
            "masked_epoch": self.masked.epoch,
            "causal_epoch": self.causal.epoch,
        }
        if device is not None:
            for k in ("input_ids", "target_ids", "attention_mask"):
                batch[k] = batch[k].to(device, non_blocking=True)
        return batch
