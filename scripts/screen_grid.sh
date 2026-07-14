#!/usr/bin/env bash
# 10M-word screening grid: run every schedule variant, then masking/optimizer
# variants of the winner. Point configs at the 10M corpus first.
set -euo pipefail
for cfg in configs/baselines/*.yaml configs/schedules/*.yaml; do
  echo "=== $cfg ==="
  python scripts/train.py --config "$cfg" "$@"
done
