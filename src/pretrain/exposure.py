"""Exposure accounting: enforce the BabyLM epoch cap and drive word-based
checkpointing without trusting the dataloader.

BabyLM rules (verify the 2026 CfP wording and pin it in
docs/rules_snapshot.md):
  * Strict track corpus <= 100M words; training may not exceed the epoch cap
    (10 epochs in recent editions) over that corpus.
  * Learning-trajectory evals require intermediate checkpoints at word-count
    milestones (2025-style: every 1M words to 10M, then every 10M to 100M,
    then every 100M to 1B of *exposure*).

Tokens are what the trainer sees; words are what the rules count. We convert
with `words_per_token`, measured once on the tokenized corpus
(total_corpus_words / total_corpus_tokens) and stored in the run config —
NOT hardcoded, since it depends on the tokenizer.

The counter counts every non-pad, non-special input position the model
receives, masked or causal alike. Padding does not count; text seen twice
counts twice. `check()` raises once the cap is exceeded, so a config bug
cannot silently disqualify a run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_MILESTONES_WORDS = (
    [m * 1_000_000 for m in range(1, 11)]          # 1M..10M every 1M
    + [m * 10_000_000 for m in range(2, 11)]       # 20M..100M every 10M
    + [m * 100_000_000 for m in range(2, 11)]      # 200M..1B every 100M
)


class EpochCapExceeded(RuntimeError):
    pass


@dataclass
class ExposureCounter:
    corpus_words: int                    # words in the training corpus
    words_per_token: float               # measured on the tokenized corpus
    max_epochs: float = 10.0             # pin against the 2026 rules!
    tokens_seen: int = 0
    milestones_words: list = field(
        default_factory=lambda: list(DEFAULT_MILESTONES_WORDS))
    _next_milestone_idx: int = 0

    # -- accounting ----------------------------------------------------------
    def add_tokens(self, n: int) -> None:
        self.tokens_seen += int(n)

    @property
    def words_seen(self) -> float:
        return self.tokens_seen * self.words_per_token

    @property
    def epochs(self) -> float:
        return self.words_seen / max(self.corpus_words, 1)

    def check(self) -> None:
        if self.epochs > self.max_epochs:
            raise EpochCapExceeded(
                f"exposure {self.words_seen:,.0f} words = {self.epochs:.3f} "
                f"epochs exceeds cap of {self.max_epochs}")

    # -- checkpoint milestones -------------------------------------------------
    def due_milestones(self) -> list:
        """Word milestones crossed since last call (each returned once)."""
        due = []
        while (self._next_milestone_idx < len(self.milestones_words)
               and self.words_seen >=
               self.milestones_words[self._next_milestone_idx]):
            due.append(self.milestones_words[self._next_milestone_idx])
            self._next_milestone_idx += 1
        return due

    # -- persistence -----------------------------------------------------------
    def state_dict(self) -> dict:
        return {"tokens_seen": self.tokens_seen,
                "next_milestone_idx": self._next_milestone_idx}

    def load_state_dict(self, s: dict) -> None:
        self.tokens_seen = s["tokens_seen"]
        self._next_milestone_idx = s["next_milestone_idx"]


# ---------------------------------------------------------------------------
# Word-budget ledger: every data-touching script registers its reads here so
# tests/test_budget_ledger.py can fail on unregistered readers.
# ---------------------------------------------------------------------------

class BudgetLedger:
    def __init__(self, path: str = "data/word_budget_ledger.json"):
        self.path = Path(path)
        if self.path.exists():
            self.entries = json.loads(self.path.read_text())
        else:
            self.entries = []

    def register(self, reader: str, source: str, words: int, note: str = ""):
        self.entries.append({"reader": reader, "source": source,
                             "words": int(words), "note": note})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2))

    def total_unique_sources(self) -> dict:
        """words per unique source (a source read twice is one budget entry)."""
        out = {}
        for e in self.entries:
            out[e["source"]] = max(out.get(e["source"], 0), e["words"])
        return out

    def budget_used(self) -> int:
        return sum(self.total_unique_sources().values())
