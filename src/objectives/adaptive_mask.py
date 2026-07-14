"""Difficulty-adaptive masking for the MNTP side of the hybrid objective.

Port of the idea in Edman & Fraser (2025, BabyLM): adapt which tokens get
masked according to the model's current ability to predict them — mask what
the model finds hard, stop wasting mask budget on tokens it has mastered.
Their implementation was MLM-only; here it plugs into GPT-BERT's
SpanMaskingStrategy so it composes with the causal loss and with dynamic
objective scheduling.

Mechanism
---------
Maintain a vocab-sized table of per-token-TYPE difficulty: an EMA of the
per-token cross-entropy observed whenever that token id was a supervised
MNTP target. GPT-BERT's SpanMaskingStrategy draws a random `mask_ratio` per
span; smaller ratio => masked earlier (topk smallest). We bias those ratios:

    biased_ratio = ratio * (1 - strength * norm_difficulty(token))

so harder tokens (norm_difficulty near 1) get systematically smaller ratios
and are preferentially masked, while the stochastic span structure is
preserved. strength=0 recovers the reference behaviour exactly.

The table lives on CPU, is updated from the training loop with
(token_ids, per_token_losses) of the masked targets each step, and is part
of the checkpoint (state_dict/load_state_dict).

Unseen tokens default to the current mean difficulty so they are neither
favored nor ignored before evidence accumulates.
"""

from __future__ import annotations

import torch


class DifficultyTable:
    """EMA per-token-id difficulty, in nats/token."""

    def __init__(self, vocab_size: int, beta: float = 0.99):
        self.vocab_size = vocab_size
        self.beta = beta
        self.ema = torch.zeros(vocab_size)
        self.count = torch.zeros(vocab_size, dtype=torch.long)

    @torch.no_grad()
    def update(self, token_ids: torch.Tensor, losses: torch.Tensor) -> None:
        """token_ids [N] long, losses [N] float — the supervised MNTP targets
        of the current step and their unreduced cross-entropies."""
        token_ids = token_ids.cpu()
        losses = losses.detach().float().cpu()
        # scatter-style EMA: for repeated ids in one step, average first
        uniq, inverse = torch.unique(token_ids, return_inverse=True)
        sums = torch.zeros(uniq.size(0)).index_add_(0, inverse, losses)
        cnts = torch.zeros(uniq.size(0)).index_add_(
            0, inverse, torch.ones_like(losses))
        means = sums / cnts
        old = self.ema[uniq]
        seen = self.count[uniq] > 0
        blended = torch.where(seen, self.beta * old + (1 - self.beta) * means,
                              means)
        self.ema[uniq] = blended
        self.count[uniq] += cnts.long()

    @torch.no_grad()
    def normalized(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Difficulty per token mapped to [0, 1] by rank-free min-max over the
        currently-seen vocabulary; unseen ids get 0.5 (neutral)."""
        seen_mask = self.count > 0
        if seen_mask.sum() < 2:
            return torch.full(token_ids.shape, 0.5)
        seen_vals = self.ema[seen_mask]
        lo, hi = seen_vals.min(), seen_vals.max()
        span = (hi - lo).clamp_min(1e-8)
        vals = (self.ema[token_ids] - lo) / span
        neutral = torch.full_like(vals, 0.5)
        return torch.where(seen_mask[token_ids], vals.clamp(0, 1), neutral)

    def state_dict(self):
        return {"ema": self.ema, "count": self.count}

    def load_state_dict(self, s):
        self.ema = s["ema"]
        self.count = s["count"]


class AdaptiveSpanMasking:
    """Drop-in wrapper around GPT-BERT's SpanMaskingStrategy.

    Same __call__(tokens, counts) -> (mask_ratios, replacement_tokens)
    contract as the original, so MaskedDataset can use it unchanged:

        ds.masking_strategy = AdaptiveSpanMasking(ds.masking_strategy,
                                                  table, strength=0.5)
    """

    def __init__(self, base_strategy, table: DifficultyTable,
                 strength: float = 0.5):
        if not (0.0 <= strength < 1.0):
            raise ValueError("strength must be in [0, 1)")
        self.base = base_strategy
        self.table = table
        self.strength = strength
        # expose attrs some callers read off the strategy
        self.n_special_tokens = base_strategy.n_special_tokens

    def __call__(self, tokens, counts=None):
        mask_ratios, replacement_tokens = self.base(tokens, counts)
        if self.strength > 0.0:
            difficulty = self.table.normalized(tokens)
            # keep special-token sentinels (inf) untouched
            finite = torch.isfinite(mask_ratios)
            bias = 1.0 - self.strength * difficulty
            mask_ratios = torch.where(finite, mask_ratios * bias, mask_ratios)
        return mask_ratios, replacement_tokens
