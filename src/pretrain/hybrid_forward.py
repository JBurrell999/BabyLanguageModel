"""Per-token hybrid forward on top of the vendored GPT-BERT model.

Reimplements exactly the mixed branch of `Bert.forward` (model.py lines
39-84 of the reference) but with `reduction="none"` cross-entropies, so the
training loop can:

  * hand the adaptive schedules honest per-token mean losses per objective
    (LossFeedback), and
  * feed (target_id, loss) pairs of MNTP targets to the DifficultyTable
    for adaptive masking,

without ever modifying the vendored files. The scalar losses returned are
numerically identical to the reference (means of the unreduced vectors),
so loss-parity with upstream is testable (tests/test_hybrid_forward.py).

Layout reminders: input_ids/target_ids are seq-first [L, B];
`num_masked` counts batch columns; attention_mask is [B, L, L] bool where
True = MASKED OUT (the reference inverts with `~`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class HybridOutput:
    loss: torch.Tensor                 # ratio-weighted total (scalar)
    z_loss: torch.Tensor               # ratio-weighted z-loss (scalar)
    masked_loss: float                 # mean nats/token (0.0 if no MNTP)
    causal_loss: float                 # mean nats/token (0.0 if no CLM)
    masked_accuracy: float
    causal_accuracy: float
    num_masked_tokens: int
    num_causal_tokens: int
    masked_target_ids: torch.Tensor    # [Nm] long, for the difficulty table
    masked_token_losses: torch.Tensor  # [Nm] float (detached)


def hybrid_forward(model, input_ids, attention_mask, target_ids,
                   num_masked: int, ratio: float) -> HybridOutput:
    contextualized = model.get_contextualized(input_ids, attention_mask)
    masked_pred, causal_pred = model.classifier(contextualized, target_ids,
                                                num_masked)

    zero = input_ids.new_zeros([], dtype=torch.float32)

    if masked_pred is not None:
        m_gold = target_ids[:, :num_masked].flatten()
        m_gold = m_gold[m_gold != -100]
        m_tok_loss = F.cross_entropy(masked_pred, m_gold, reduction="none")
        masked_loss = m_tok_loss.mean()
        masked_z = torch.logsumexp(masked_pred, dim=-1).pow(2).mean()
        with torch.no_grad():
            masked_acc = (masked_pred.argmax(-1) == m_gold).float().mean()
    else:
        m_gold = input_ids.new_empty([0], dtype=torch.long)
        m_tok_loss = zero.new_empty([0])
        masked_loss, masked_z, masked_acc = zero, zero, zero

    if causal_pred is not None:
        c_gold = target_ids[:, num_masked:].flatten()
        c_gold = c_gold[c_gold != -100]
        c_tok_loss = F.cross_entropy(causal_pred, c_gold, reduction="none")
        causal_loss = c_tok_loss.mean()
        causal_z = torch.logsumexp(causal_pred, dim=-1).pow(2).mean()
        with torch.no_grad():
            causal_acc = (causal_pred.argmax(-1) == c_gold).float().mean()
    else:
        c_gold = input_ids.new_empty([0], dtype=torch.long)
        c_tok_loss = zero.new_empty([0])
        causal_loss, causal_z, causal_acc = zero, zero, zero

    loss = ratio * masked_loss + (1.0 - ratio) * causal_loss
    z_loss = ratio * masked_z + (1.0 - ratio) * causal_z

    return HybridOutput(
        loss=loss,
        z_loss=z_loss,
        masked_loss=float(masked_loss.detach()),
        causal_loss=float(causal_loss.detach()),
        masked_accuracy=float(masked_acc),
        causal_accuracy=float(causal_acc),
        num_masked_tokens=int(m_gold.numel()),
        num_causal_tokens=int(c_gold.numel()),
        masked_target_ids=m_gold.detach(),
        masked_token_losses=m_tok_loss.detach(),
    )
