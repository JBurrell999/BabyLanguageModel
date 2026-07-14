# Dynamic Objective Scheduling for Hybrid Masked–Causal Pretraining
**BabyLM 2026 — Strict track (≤100M words)**

GPT-BERT trains one transformer on a mix of masked-next-token-prediction
(MNTP) and causal LM, but the official baselines freeze the mask:causal
ratio (15:1, 1:1, 1:15) for the whole run — implemented by dedicating whole
GPUs to one objective. This repo makes the mixture **dynamic**: annealed or
loss-adaptive over training, decided per optimizer step, on a single GPU.

Contributions (code is organized around these):
1. **Dynamic objective scheduling** — `src/objectives/schedules.py` + `mixer.py`.
   Fixed (= official baselines), linear, cosine, step, loss-adaptive
   (need-based) and slope-adaptive (learning-progress) policies over
   p_mask(t), exploiting `Bert.forward`'s native mixed-batch support
   (`num_masked`, `ratio`).
2. **Difficulty-adaptive masking in the hybrid** — `src/objectives/adaptive_mask.py`.
   Edman & Fraser (2025) style per-token-difficulty masking, previously
   MLM-only, wrapped around GPT-BERT's SpanMaskingStrategy.
3. **Muon vs AdamW/LAMB on the hybrid objective** — `src/optim/muon.py`
   (Muon on 2D body weights, AdamW on embeddings/norms/tied head).
4. **Acquisition-trajectory analysis** — `src/analysis/` over the word-milestone
   checkpoints the exposure counter emits automatically.

## Quick start
```bash
pip install -r requirements.txt
python -m pytest tests/ -q                     # 32 tests incl. loss-parity vs upstream
python scripts/train.py --config configs/smoke.yaml            # CPU end-to-end
python scripts/train.py --config configs/schedules/adaptive_loss.yaml   # real run
bash scripts/screen_grid.sh                    # 10M-scale screening grid
```

## Before any real run (Phase 0 — do not skip)
1. Fetch and pin the rules verbatim into `docs/rules_snapshot.md`:
   guidelines page, 2026 CfP, and the eval repo (checkpoint schedule,
   task list, submission format). `exposure.max_epochs` and the milestone
   list in `src/pretrain/exposure.py` must match the pinned wording.
2. Download the official Strict corpus; tokenize with `src/tokenizer/`
   (BPE trained ONLY on the budget corpus); register reads in
   `data/word_budget_ledger.json`.
3. Measure `words_per_token` on the tokenized corpus and set it in
   `configs/base.yaml` — checkpoint milestones and the epoch cap depend on it.
4. Reproduce one fixed-ratio baseline at 10M scale and sanity-check loss
   curves against the GPT-BERT paper before running the grid.

## Design notes
- **Exposure-based accounting** (`src/pretrain/exposure.py`): the epoch cap is
  enforced by counting non-pad tokens actually fed to the model, converted to
  words — never by trusting the dataloader. Word-milestone checkpoints
  (1M..10M by 1M, then by 10M, then by 100M) fire from the same counter.
- **ratio_mode** separates *batch composition* from *loss weighting*
  ("composition" | "schedule" | "fixed:x") — their dissociation is an
  ablation axis.
- **Per-token feedback** (`src/pretrain/hybrid_forward.py`) reimplements the
  vendored mixed forward with `reduction="none"`; parity is unit-tested, and
  the per-token losses drive both the adaptive schedules and the difficulty
  table.
- Screen at 10M words (hours/run on one H100), confirm winners at 100M
  (~1–2 days/run). The ONLY difference between screen and confirm is config.

## Layout
```
configs/            baselines/ schedules/ masking/ optimizer/ + base.yaml
src/objectives/     schedules.py  mixer.py  adaptive_mask.py   ← core novelty
src/optim/muon.py   src/pretrain/{trainer,hybrid_forward,exposure}.py
src/model/          vendored from github.com/ltgoslo/gpt-bert (see LICENSE)
src/analysis/       checkpoint-trajectory analyses
scripts/ tests/ docs/
```

## Still TODO (tracked in docs/experiment_log.md)
- HF-format checkpoint export (use GPT-BERT's released remote-code modeling
  files so the official eval loads LL and PLL modes natively).
- Wire `src/eval/` to the official babylm-eval harness (`--fast` subset for
  screening). Never re-implement scoring.
- Validation loop on the held-out split (reference: validation_epoch).
- Multi-seed variance at 10M scale before claiming any ranking.
