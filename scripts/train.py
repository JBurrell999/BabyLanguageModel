#!/usr/bin/env python
"""Train entrypoint.

    python scripts/train.py --config configs/schedules/linear_mlm_to_clm.yaml
    python scripts/train.py --config ... --smoke --device cpu

--smoke shrinks steps/batch for CI; it does NOT change the schedule logic,
so a smoke run exercises exactly the code paths of a real run.
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pretrain.trainer import Trainer  # noqa: E402


def deep_update(base: dict, patch: dict) -> dict:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None,
                    help="cuda | mps | cpu (overrides config)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run for CI / laptops")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    # optional single-level inheritance: `extends: ../base.yaml`
    if "extends" in cfg:
        base = yaml.safe_load((cfg_path.parent / cfg.pop("extends")).read_text())
        cfg = deep_update(base, cfg)

    if args.device:
        cfg["device"] = args.device
    if args.smoke:
        deep_update(cfg, {
            "max_steps": 30,
            "local_batch_size": 8,
            "accumulate_steps": 1,
            "seq_length": 64,
            "log_every": 5,
            "mixed_precision": False,
            "wandb": {"enabled": False},
            "output_dir": cfg["output_dir"].rstrip("/") + "_smoke",
        })

    Trainer(cfg).train()


if __name__ == "__main__":
    main()
