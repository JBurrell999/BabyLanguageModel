# Design notes
- Mixed batches exploit Bert.forward's (num_masked, ratio) interface; the
  model is SEQUENCE-FIRST ([L, B]); num_masked splits dim=1. attention_mask
  [B, L, L] bool, True = blocked; causal columns are ~tril.
- Adaptive schedules normalize each objective's loss by its own history
  (fast/slow EMA) because MNTP and CLM losses are not on the same scale.
- The masked and causal source streams advance at different rates under a
  dynamic schedule; both are shuffled copies of the same corpus, and total
  exposure (the thing the rules regulate) is governed by ExposureCounter.
- Cognitive framing: MLM→CLM anneal ≈ comprehension-driven learning
  preceding production; slope-adaptive ≈ learning-progress curricula
  (Graves et al. 2017). The trajectory analyses test whether schedules shift
  WHEN phenomena are acquired, not just endpoint scores.
