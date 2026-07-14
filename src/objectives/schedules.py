"""Objective-mixture schedules for hybrid masked/causal pretraining.

Every schedule answers one question at every optimizer step t:

    p_mask(t) ∈ [p_min, p_max]  — the fraction of the batch trained with the
                                  MNTP (masked-next-token-prediction) objective.
                                  The remaining 1 - p_mask(t) is causal LM.

The official GPT-BERT baselines are the FixedSchedule special cases
(15/16, 1/2, 1/16). Everything else in this file is the contribution.

Static schedules depend only on progress = t / T.
Adaptive schedules additionally consume a per-step LossFeedback signal
(the separate masked and causal losses that GPT-BERT's forward already
returns) and reallocate probability mass online.

All schedules are:
  * deterministic given (config, feedback history) — no hidden RNG;
  * serializable via state_dict()/load_state_dict() so runs resume exactly;
  * observable via last_info() for per-step W&B/CSV logging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LossFeedback:
    """Per-step training signal handed to adaptive schedules.

    Losses must be per-token cross-entropies (nats/token) so the two
    objectives are on a comparable scale. `*_tokens` are the counts of
    supervised tokens contributing to each loss (0 when the batch had no
    examples of that objective this step).
    """

    masked_loss: float
    causal_loss: float
    masked_tokens: int = 0
    causal_tokens: int = 0


class Schedule:
    """Base class. Subclasses implement _p(progress) or override step()."""

    name = "base"

    def __init__(self, p_min: float = 0.05, p_max: float = 0.95):
        if not (0.0 <= p_min <= p_max <= 1.0):
            raise ValueError(f"need 0 <= p_min <= p_max <= 1, got {p_min}, {p_max}")
        self.p_min = p_min
        self.p_max = p_max
        self._last: dict = {}

    # -- interface ---------------------------------------------------------
    def step(self, step: int, max_steps: int,
             feedback: Optional[LossFeedback] = None) -> float:
        progress = min(max(step / max(max_steps, 1), 0.0), 1.0)
        p = self._clamp(self._p(progress))
        self._last = {"p_mask": p, "progress": progress}
        return p

    def _p(self, progress: float) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- utilities ---------------------------------------------------------
    def _clamp(self, p: float) -> float:
        return min(max(p, self.p_min), self.p_max)

    def last_info(self) -> dict:
        """Loggable dict describing the most recent decision."""
        return dict(self._last)

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


# ---------------------------------------------------------------------------
# Static schedules
# ---------------------------------------------------------------------------

class FixedSchedule(Schedule):
    """Constant ratio — reproduces the official baselines.

    p = 15/16 → GPT-BERT default;  p = 1/2 → mixed;  p = 1/16 → causal-focus.
    """

    name = "fixed"

    def __init__(self, p: float, **kw):
        super().__init__(**kw)
        self.p = p

    def _p(self, progress: float) -> float:
        return self.p


class LinearSchedule(Schedule):
    """Linear interpolation p_start → p_end over training."""

    name = "linear"

    def __init__(self, p_start: float, p_end: float, **kw):
        super().__init__(**kw)
        self.p_start, self.p_end = p_start, p_end

    def _p(self, progress: float) -> float:
        return self.p_start + (self.p_end - self.p_start) * progress


class CosineSchedule(Schedule):
    """Cosine interpolation p_start → p_end (slow-fast-slow)."""

    name = "cosine"

    def __init__(self, p_start: float, p_end: float, **kw):
        super().__init__(**kw)
        self.p_start, self.p_end = p_start, p_end

    def _p(self, progress: float) -> float:
        w = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.p_start + (self.p_end - self.p_start) * w


class StepSchedule(Schedule):
    """Piecewise-constant schedule.

    boundaries: fractions of training at which the value changes, e.g.
        boundaries=[0.5], values=[0.9375, 0.0625]
    switches from MLM-heavy to causal-heavy at the midpoint.
    """

    name = "step"

    def __init__(self, boundaries: list, values: list, **kw):
        super().__init__(**kw)
        if len(values) != len(boundaries) + 1:
            raise ValueError("need len(values) == len(boundaries) + 1")
        if list(boundaries) != sorted(boundaries):
            raise ValueError("boundaries must be sorted ascending")
        self.boundaries = list(boundaries)
        self.values = list(values)

    def _p(self, progress: float) -> float:
        for b, v in zip(self.boundaries, self.values):
            if progress < b:
                return v
        return self.values[-1]


# ---------------------------------------------------------------------------
# Adaptive schedules
# ---------------------------------------------------------------------------

class _Ema:
    """Exponential moving average with bias correction, resumable."""

    def __init__(self, beta: float):
        self.beta = beta
        self.value = 0.0
        self.count = 0

    def update(self, x: float) -> float:
        self.count += 1
        self.value = self.beta * self.value + (1.0 - self.beta) * x
        return self.corrected()

    def corrected(self) -> float:
        if self.count == 0:
            return 0.0
        return self.value / (1.0 - self.beta ** self.count)

    def state_dict(self):
        return {"value": self.value, "count": self.count}

    def load_state_dict(self, s):
        self.value, self.count = s["value"], s["count"]


class LossAdaptiveSchedule(Schedule):
    """Allocate batch share to whichever objective currently has the higher
    normalized loss ("need-based" allocation).

    Mechanism
    ---------
    Maintain EMAs of the per-token masked and causal losses. Because the two
    objectives have intrinsically different difficulty (MNTP sees bidirectional
    context → lower loss), raw losses are not comparable; we therefore
    normalize each EMA by a slowly-updated reference (its own long-horizon
    EMA), yielding a unitless "relative pressure" per objective:

        r_m = ema_fast(L_m) / ema_slow(L_m),   r_c likewise

    r > 1 means the objective is currently doing worse than its own recent
    history. p_mask is then a tempered softmax over the pressures:

        p_mask = sigmoid((r_m - r_c) / temperature)

    mapped into [p_min, p_max]. `update_every` decouples the control loop
    from the optimizer loop; between updates the previous p is reused
    (keeps batch composition stable across gradient-accumulation windows).

    A warmup period holds p at `p_warmup` until both EMAs are populated.
    """

    name = "loss_adaptive"

    def __init__(self,
                 temperature: float = 0.05,
                 beta_fast: float = 0.98,
                 beta_slow: float = 0.999,
                 update_every: int = 10,
                 warmup_steps: int = 200,
                 p_warmup: float = 0.5,
                 **kw):
        super().__init__(**kw)
        self.temperature = temperature
        self.update_every = update_every
        self.warmup_steps = warmup_steps
        self.p_warmup = self._clamp(p_warmup)
        self.fast_m, self.slow_m = _Ema(beta_fast), _Ema(beta_slow)
        self.fast_c, self.slow_c = _Ema(beta_fast), _Ema(beta_slow)
        self._p_current = self.p_warmup

    def step(self, step: int, max_steps: int,
             feedback: Optional[LossFeedback] = None) -> float:
        if feedback is not None:
            if feedback.masked_tokens > 0:
                self.fast_m.update(feedback.masked_loss)
                self.slow_m.update(feedback.masked_loss)
            if feedback.causal_tokens > 0:
                self.fast_c.update(feedback.causal_loss)
                self.slow_c.update(feedback.causal_loss)

        warm = (step < self.warmup_steps
                or self.fast_m.count == 0 or self.fast_c.count == 0)
        if warm:
            self._p_current = self.p_warmup
        elif step % self.update_every == 0:
            r_m = self.fast_m.corrected() / max(self.slow_m.corrected(), 1e-8)
            r_c = self.fast_c.corrected() / max(self.slow_c.corrected(), 1e-8)
            z = (r_m - r_c) / self.temperature
            p_raw = 1.0 / (1.0 + math.exp(-z))
            self._p_current = self.p_min + (self.p_max - self.p_min) * p_raw

        self._last = {
            "p_mask": self._p_current,
            "progress": step / max(max_steps, 1),
            "ema_fast_masked": self.fast_m.corrected(),
            "ema_fast_causal": self.fast_c.corrected(),
            "ema_slow_masked": self.slow_m.corrected(),
            "ema_slow_causal": self.slow_c.corrected(),
            "warmup": float(warm),
        }
        return self._p_current

    def state_dict(self):
        return {
            "fast_m": self.fast_m.state_dict(), "slow_m": self.slow_m.state_dict(),
            "fast_c": self.fast_c.state_dict(), "slow_c": self.slow_c.state_dict(),
            "p_current": self._p_current,
        }

    def load_state_dict(self, s):
        self.fast_m.load_state_dict(s["fast_m"])
        self.slow_m.load_state_dict(s["slow_m"])
        self.fast_c.load_state_dict(s["fast_c"])
        self.slow_c.load_state_dict(s["slow_c"])
        self._p_current = s["p_current"]


class SlopeAdaptiveSchedule(Schedule):
    """Allocate batch share to whichever objective is currently IMPROVING
    faster ("greedy learning-progress" allocation, à la automated curricula:
    Graves et al. 2017 learning-progress signals).

    Mechanism
    ---------
    Track fast and slow EMAs of each per-token loss; the improvement signal
    is the normalized gap slope

        s = (ema_slow - ema_fast) / ema_slow      (>0 while improving)

    p_mask = sigmoid((s_m - s_c) / temperature) mapped to [p_min, p_max]:
    mass flows to the objective with steeper current descent. This is the
    exploitative counterpart to LossAdaptiveSchedule's need-based rule; the
    contrast between the two is itself an ablation axis.
    """

    name = "slope_adaptive"

    def __init__(self,
                 temperature: float = 0.02,
                 beta_fast: float = 0.97,
                 beta_slow: float = 0.998,
                 update_every: int = 10,
                 warmup_steps: int = 200,
                 p_warmup: float = 0.5,
                 **kw):
        super().__init__(**kw)
        self.temperature = temperature
        self.update_every = update_every
        self.warmup_steps = warmup_steps
        self.p_warmup = self._clamp(p_warmup)
        self.fast_m, self.slow_m = _Ema(beta_fast), _Ema(beta_slow)
        self.fast_c, self.slow_c = _Ema(beta_fast), _Ema(beta_slow)
        self._p_current = self.p_warmup

    def step(self, step: int, max_steps: int,
             feedback: Optional[LossFeedback] = None) -> float:
        if feedback is not None:
            if feedback.masked_tokens > 0:
                self.fast_m.update(feedback.masked_loss)
                self.slow_m.update(feedback.masked_loss)
            if feedback.causal_tokens > 0:
                self.fast_c.update(feedback.causal_loss)
                self.slow_c.update(feedback.causal_loss)

        warm = (step < self.warmup_steps
                or self.fast_m.count == 0 or self.fast_c.count == 0)
        if warm:
            self._p_current = self.p_warmup
        elif step % self.update_every == 0:
            s_m = ((self.slow_m.corrected() - self.fast_m.corrected())
                   / max(self.slow_m.corrected(), 1e-8))
            s_c = ((self.slow_c.corrected() - self.fast_c.corrected())
                   / max(self.slow_c.corrected(), 1e-8))
            z = (s_m - s_c) / self.temperature
            p_raw = 1.0 / (1.0 + math.exp(-z))
            self._p_current = self.p_min + (self.p_max - self.p_min) * p_raw

        self._last = {
            "p_mask": self._p_current,
            "progress": step / max(max_steps, 1),
            "slope_masked": (self.slow_m.corrected() - self.fast_m.corrected()),
            "slope_causal": (self.slow_c.corrected() - self.fast_c.corrected()),
            "warmup": float(warm),
        }
        return self._p_current

    state_dict = LossAdaptiveSchedule.state_dict
    load_state_dict = LossAdaptiveSchedule.load_state_dict


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCHEDULES = {
    cls.name: cls
    for cls in (FixedSchedule, LinearSchedule, CosineSchedule, StepSchedule,
                LossAdaptiveSchedule, SlopeAdaptiveSchedule)
}


def build_schedule(cfg: dict) -> Schedule:
    """Build a schedule from a config dict, e.g. from YAML:

        schedule:
          name: linear
          p_start: 0.9375   # 15/16, MLM-heavy
          p_end:   0.0625   # 1/16,  causal-heavy
          p_min: 0.05
          p_max: 0.95
    """
    cfg = dict(cfg)
    name = cfg.pop("name")
    if name not in SCHEDULES:
        raise KeyError(f"unknown schedule '{name}'; options: {sorted(SCHEDULES)}")
    return SCHEDULES[name](**cfg)
