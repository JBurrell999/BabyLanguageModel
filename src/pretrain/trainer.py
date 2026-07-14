"""Single-GPU (H100 / MPS / CPU-smoke) trainer for dynamic-objective GPT-BERT.

Forked in spirit from gpt-bert/pretraining/train_100m.py, restructured around
the ObjectiveMixer: instead of dedicating devices to one objective, every
batch is composed per-step according to schedule.p_mask(t), and the separate
masked/causal losses are fed back into adaptive schedules and the
difficulty-adaptive masking table.

Run:  python scripts/train.py --config configs/schedules/linear_mlm_to_clm.yaml
      python scripts/train.py --config ... --smoke        # tiny CI run
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn

from src.model.model import Bert
from src.model.dataset import MaskedDataset, CausalDataset
from src.objectives.schedules import build_schedule, LossFeedback
from src.objectives.mixer import ObjectiveMixer
from src.objectives.adaptive_mask import DifficultyTable, AdaptiveSpanMasking
from src.optim.muon import HybridOptimizer
from src.pretrain.exposure import ExposureCounter
from src.pretrain.hybrid_forward import hybrid_forward


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, max_steps: int, warmup: int, peak: float,
              floor_frac: float = 0.1) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(max_steps - warmup, 1)
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * t)))


class CsvLogger:
    """Always-on CSV fallback; W&B is layered on top when available."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._keys = None
        self._fh = None

    def log(self, row: dict) -> None:
        if self._fh is None:
            self._keys = list(row.keys())
            self._fh = open(self.path, "w")
            self._fh.write(",".join(self._keys) + "\n")
        self._fh.write(",".join(str(row.get(k, "")) for k in self._keys) + "\n")
        self._fh.flush()


class Args:
    """Attribute bag for the vendored GPT-BERT dataset classes."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        seed_everything(cfg.get("seed", 42))
        self.device = torch.device(cfg.get("device", "cuda"
                                           if torch.cuda.is_available()
                                           else "cpu"))
        self.out_dir = Path(cfg["output_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "config_resolved.json").write_text(
            json.dumps(cfg, indent=2))

        # -- tokenizer + datasets ------------------------------------------
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(cfg["tokenizer_path"])
        ds_args = Args(
            n_special_tokens=cfg.get("n_special_tokens", 16),
            mask_random_p=cfg.get("mask_random_p", 0.1),
            mask_keep_p=cfg.get("mask_keep_p", 0.1),
            vocab_size=self.tokenizer.get_vocab_size(),
            mask_p_start=cfg.get("mask_p_start", 0.3),
            mask_p_end=cfg.get("mask_p_end", 0.15),
            max_steps=cfg["max_steps"],
            seq_length=cfg.get("seq_length", 128),
            seed=cfg.get("seed", 42),
        )
        seq_len = ds_args.seq_length
        self.masked_ds = MaskedDataset(cfg["train_path"], self.tokenizer,
                                       ds_args, seq_len, rank=None,
                                       world_size=1)
        self.causal_ds = CausalDataset(cfg["train_path"], self.tokenizer,
                                       ds_args, seq_len, rank=None,
                                       world_size=1)

        # -- adaptive masking (optional) ------------------------------------
        am_cfg = cfg.get("adaptive_masking", {}) or {}
        self.difficulty_table = None
        if am_cfg.get("enabled", False):
            self.difficulty_table = DifficultyTable(
                vocab_size=self.tokenizer.get_vocab_size(),
                beta=am_cfg.get("beta", 0.99))
            self.masked_ds.masking_strategy = AdaptiveSpanMasking(
                self.masked_ds.masking_strategy, self.difficulty_table,
                strength=am_cfg.get("strength", 0.5))

        # -- schedule + mixer -----------------------------------------------
        self.schedule = build_schedule(cfg["schedule"])
        self.mixer = ObjectiveMixer(
            self.masked_ds, self.causal_ds,
            batch_size=cfg["local_batch_size"],
            ratio_mode=cfg.get("ratio_mode", "composition"),
            num_workers=cfg.get("num_workers", 2),
            seed=cfg.get("seed", 42))

        # -- model ------------------------------------------------------------
        with open(cfg["model_config"]) as f:
            model_cfg_dict = json.load(f)
        model_cfg = Args(**model_cfg_dict)
        self.model = Bert(model_cfg, activation_checkpointing=cfg.get(
            "activation_checkpointing", False)).to(self.device)
        self.ema_model = Bert(model_cfg).to(self.device)
        self.ema_model.load_state_dict(self.model.state_dict())
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_decay = cfg.get("ema_decay", 0.999)

        # -- optimizer ----------------------------------------------------------
        opt_cfg = cfg.get("optimizer", {"name": "adamw"})
        name = opt_cfg.get("name", "adamw")
        if name == "muon":
            self.optimizer = HybridOptimizer(
                self.model,
                muon_lr=opt_cfg.get("muon_lr", 0.02),
                muon_momentum=opt_cfg.get("muon_momentum", 0.95),
                adamw_lr=opt_cfg.get("adamw_lr", 1e-2),
                weight_decay=opt_cfg.get("weight_decay", 0.1))
        elif name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_cfg.get("lr", 1e-2),
                betas=tuple(opt_cfg.get("betas", (0.9, 0.98))),
                eps=opt_cfg.get("eps", 1e-8),
                weight_decay=opt_cfg.get("weight_decay", 0.1))
        elif name == "lamb":
            from src.model.lamb import Lamb
            self.optimizer = Lamb(
                self.model.parameters(), lr=opt_cfg.get("lr", 1e-2),
                betas=tuple(opt_cfg.get("betas", (0.9, 0.98))),
                eps=opt_cfg.get("eps", 1e-8),
                weight_decay=opt_cfg.get("weight_decay", 0.1))
        else:
            raise ValueError(f"unknown optimizer {name!r}")
        self.peak_lrs = [g["lr"] for g in self.optimizer.param_groups]

        # -- exposure accounting ----------------------------------------------
        exp_cfg = cfg["exposure"]
        self.exposure = ExposureCounter(
            corpus_words=exp_cfg["corpus_words"],
            words_per_token=exp_cfg["words_per_token"],
            max_epochs=exp_cfg.get("max_epochs", 10.0))
        self.pad_id = self.tokenizer.token_to_id("<pad>")

        # -- logging --------------------------------------------------------------
        self.csv = CsvLogger(self.out_dir / "metrics.csv")
        self.wandb = None
        if cfg.get("wandb", {}).get("enabled", False):
            try:
                import wandb
                self.wandb = wandb
                wandb.init(project=cfg["wandb"].get("project",
                                                    "babylm2026-strict"),
                           name=cfg.get("run_name"), config=cfg,
                           mode=cfg["wandb"].get("mode", "online"))
            except Exception as e:  # offline machines: CSV still works
                print(f"[warn] wandb unavailable: {e}")

        self.max_steps = cfg["max_steps"]
        self.accumulate = cfg.get("accumulate_steps", 1)
        self.warmup = cfg.get("warmup_steps", int(0.016 * self.max_steps))
        self.z_weight = cfg.get("z_loss_weight", 1e-4)
        self.clip = cfg.get("max_gradient", 2.0)
        self.log_every = cfg.get("log_every", 20)
        self.save_every_steps = cfg.get("save_every_steps", 0)  # 0 = only word milestones
        self.amp = cfg.get("mixed_precision", True) and self.device.type == "cuda"

    # ------------------------------------------------------------------
    def _set_lr(self, step: int) -> float:
        scale = cosine_lr(step, self.max_steps, self.warmup, 1.0)
        for g, peak in zip(self.optimizer.param_groups, self.peak_lrs):
            g["lr"] = peak * scale
        return scale

    @torch.no_grad()
    def _ema_update(self):
        for pq, pk in zip(self.model.parameters(),
                          self.ema_model.parameters()):
            pk.data.mul_(self.ema_decay).add_(pq.detach().data,
                                              alpha=1.0 - self.ema_decay)

    def _save(self, tag: str, step: int) -> None:
        ckpt_dir = self.out_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        payload = {
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "schedule": self.schedule.state_dict(),
            "exposure": self.exposure.state_dict(),
            "difficulty_table": (self.difficulty_table.state_dict()
                                 if self.difficulty_table else None),
            "global_step": step,
            "config": self.cfg,
        }
        torch.save(payload, ckpt_dir / f"{tag}.pt")
        print(f"[ckpt] saved {tag} at step {step} "
              f"({self.exposure.words_seen:,.0f} words)")

    # ------------------------------------------------------------------
    def train(self) -> None:
        self.model.train()
        feedback = None
        step = 0
        while step < self.max_steps:
            lr_scale = self._set_lr(step)
            # one schedule decision per optimizer step, shared by micro-batches
            p_mask = self.schedule.step(step, self.max_steps, feedback)

            self.optimizer.zero_grad(set_to_none=True)
            agg = {"masked_loss": 0.0, "causal_loss": 0.0,
                   "masked_tokens": 0, "causal_tokens": 0,
                   "loss": 0.0, "mask_p": 0.0}
            for _ in range(self.accumulate):
                batch = self.mixer.next_batch(p_mask, step,
                                              device=self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=self.amp):
                    out = hybrid_forward(self.model, batch["input_ids"],
                                         batch["attention_mask"],
                                         batch["target_ids"],
                                         batch["num_masked"],
                                         batch["ratio"])
                ((out.loss + self.z_weight * out.z_loss)
                 / self.accumulate).backward()

                # exposure: every non-pad input position counts
                self.exposure.add_tokens(
                    int((batch["input_ids"] != self.pad_id).sum()))

                if self.difficulty_table is not None \
                        and out.num_masked_tokens > 0:
                    self.difficulty_table.update(out.masked_target_ids,
                                                 out.masked_token_losses)

                w = 1.0 / self.accumulate
                agg["loss"] += float(out.loss.detach()) * w
                agg["mask_p"] += float(batch["mask_p"]) * w
                agg["masked_loss"] += out.masked_loss * w
                agg["causal_loss"] += out.causal_loss * w
                agg["masked_tokens"] += out.num_masked_tokens
                agg["causal_tokens"] += out.num_causal_tokens

            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(),
                                                 self.clip)
            self.optimizer.step()
            self._ema_update()
            self.exposure.check()

            feedback = LossFeedback(masked_loss=agg["masked_loss"],
                                    causal_loss=agg["causal_loss"],
                                    masked_tokens=agg["masked_tokens"],
                                    causal_tokens=agg["causal_tokens"])

            # -- logging ------------------------------------------------
            if step % self.log_every == 0:
                row = {
                    "step": step,
                    "lr_scale": lr_scale,
                    "p_mask": p_mask,
                    "num_masked": batch["num_masked"],
                    "loss": agg["loss"],
                    "mlm_loss": agg["masked_loss"],
                    "clm_loss": agg["causal_loss"],
                    "mask_p": agg["mask_p"],
                    "grad_norm": float(grad_norm),
                    "words_seen": self.exposure.words_seen,
                    "epochs": self.exposure.epochs,
                }
                row.update({f"sched/{k}": v
                            for k, v in self.schedule.last_info().items()})
                self.csv.log(row)
                if self.wandb:
                    self.wandb.log(row, step=step)

            # -- checkpoints ---------------------------------------------
            for milestone in self.exposure.due_milestones():
                self._save(f"words_{milestone // 1_000_000}M", step)
            if self.save_every_steps and step and \
                    step % self.save_every_steps == 0:
                self._save(f"step_{step}", step)

            step += 1

        self._save("final", step)
        print("[done] training complete;",
              f"{self.exposure.words_seen:,.0f} words "
              f"({self.exposure.epochs:.2f} epochs) of exposure")
