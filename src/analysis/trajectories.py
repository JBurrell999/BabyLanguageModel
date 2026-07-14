"""Acquisition-trajectory analyses over word-milestone checkpoints.

Consumes OUTPUTS of the official babylm-eval harness (never re-implements
scoring): point the harness at runs/<name>/checkpoints/words_*.pt exports
and collect per-checkpoint JSON here.

Analyses (the Track-B / award angle):
  * BLiMP phenomenon learning order: for each phenomenon, the first
    milestone at which accuracy exceeds a criterion (e.g. 0.75) — compare
    orderings across schedules (Kendall tau vs fixed-ratio baselines).
  * AoA/CDI curves per checkpoint vs child age-of-acquisition norms.
  * Schedule-event alignment: for adaptive runs, overlay p_mask(t) (from
    metrics.csv) on acquisition curves; test whether allocation shifts
    precede/follow phenomenon acquisition.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def load_schedule_trace(run_dir: str) -> list[dict]:
    rows = []
    with open(Path(run_dir) / "metrics.csv") as f:
        for row in csv.DictReader(f):
            rows.append({"step": int(row["step"]),
                         "words": float(row["words_seen"]),
                         "p_mask": float(row["p_mask"])})
    return rows


def acquisition_order(per_checkpoint_scores: dict[int, dict[str, float]],
                      criterion: float = 0.75) -> dict[str, int | None]:
    """per_checkpoint_scores: {milestone_words: {phenomenon: accuracy}} ->
    first milestone each phenomenon crosses the criterion (None = never)."""
    order: dict[str, int | None] = {}
    for words in sorted(per_checkpoint_scores):
        for phen, acc in per_checkpoint_scores[words].items():
            if acc >= criterion and phen not in order:
                order[phen] = words
    all_phens = {p for s in per_checkpoint_scores.values() for p in s}
    for p in all_phens:
        order.setdefault(p, None)
    return order


def kendall_tau(order_a: dict, order_b: dict) -> float:
    """Kendall tau-a between two acquisition orderings on shared, acquired
    phenomena. Pure-python; no scipy dependency."""
    shared = [p for p in order_a
              if order_a.get(p) is not None and order_b.get(p) is not None]
    n = len(shared)
    if n < 2:
        return float("nan")
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = order_a[shared[i]] - order_a[shared[j]]
            b = order_b[shared[i]] - order_b[shared[j]]
            s = (a > 0) - (a < 0), (b > 0) - (b < 0)
            if 0 in s:
                continue
            if s[0] == s[1]:
                concordant += 1
            else:
                discordant += 1
    total = n * (n - 1) / 2
    return (concordant - discordant) / total
