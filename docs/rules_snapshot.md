# BabyLM 2026 rules snapshot — PIN BEFORE ANY REAL RUN
Status: NOT YET PINNED. Fetch and quote verbatim, with retrieval dates:
1. https://babylm.github.io/guidelines.html  — track definitions, data budget, epoch cap wording
2. 2026 CfP — evaluation tasks, deadlines, submission format
3. babylm eval repo — exact checkpoint schedule for trajectory tasks, --fast subset
Then update: configs/base.yaml (exposure.max_epochs), src/pretrain/exposure.py
(DEFAULT_MILESTONES_WORDS) if the 2026 schedule differs.
