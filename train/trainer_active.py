"""Trainer implementation for SS8 distillation baseline."""

from __future__ import annotations

import contextlib
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.cluster import AgglomerativeClustering, KMeans

from data.pyg_dataset import DistillationGraphDataset
from eval.metrics import compute_teacher_metrics
from models.factory import build_model
from train.losses import hard_cross_entropy, soft_target_cross_entropy, teacher_kl_divergence
# new packages
from torch.utils.data import Subset
import numpy as np

class Trainer:
    def __init__(
        self,
        cfg: dict,
        run_dir: Path,
        logger,
        device: torch.device,
        wandb_run=None,
    ) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.logger = logger
        self.device = device
        self.wandb_run = wandb_run

        self.ckpt_dir = Path(cfg["paths"]["checkpoints_root"]) / run_dir.name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.history_path = self.run_dir / "history.csv"
        self.metrics_path = self.run_dir / "metrics_final.json"

        self._init_s3_sync()

        self.multi_gpu = bool(cfg["train"].get("multi_gpu", False))
        self.use_amp = bool(cfg["train"].get("mixed_precision", True)) and device.type == "cuda"
        self.teacher_loss_type = cfg["train"].get("teacher_loss_type", "soft_ce")
        self.temperature = float(cfg["train"].get("distill_temperature", 1.0))
        self.lambda_teacher = float(cfg["train"].get("lambda_teacher", 1.0))
        self.lambda_dssp = float(cfg["train"].get("lambda_dssp", 0.0))
        self.require_dssp_labels = bool(cfg["train"].get("require_dssp_labels", False))
        self.use_validation = bool(cfg["train"].get("use_validation", True))
        self.log_every_steps = int(cfg["train"].get("log_every_steps", 100))
        self.global_step = 0

        self._validate_teacher_cache()
        train_cfg = cfg["train"]
        self.batch_size = int(train_cfg.get("batch_size", 50))
        self.train_loader, self.val_loader, self.test_loader = self._build_loaders()

        self.model = build_model(cfg)
        self._setup_parallel_model()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(train_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-6)),
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=int(train_cfg.get("lr_step_size", 3)),
            gamma=float(train_cfg.get("lr_gamma", 0.9)),
        )
        self.scaler = self._build_grad_scaler()

        self.max_epochs = int(train_cfg.get("epochs", 120))
        self.patience = int(train_cfg.get("early_stopping_patience", 20))

        self.start_epoch = 1
        self.best_val = math.inf
        self.best_epoch = 0
        self.no_improve_epochs = 0
        self._last_completed_epoch = 0

        resume_path = train_cfg.get("resume_checkpoint", "")
        if resume_path:
            self._load_checkpoint(Path(resume_path))

    def _init_s3_sync(self) -> None:
        s3_cfg = self.cfg.get("s3_sync", {})
        self.s3_enabled = bool(s3_cfg.get("enabled", False))
        self.s3_fail_on_error = bool(s3_cfg.get("fail_on_error", False))
        self.s3_upload_last_each_epoch = bool(s3_cfg.get("upload_last_each_epoch", True))
        self.s3_upload_best = bool(s3_cfg.get("upload_best", True))
        self.s3_upload_epoch_checkpoints = bool(s3_cfg.get("upload_epoch_checkpoints", False))
        self.s3_upload_run_artifacts = bool(s3_cfg.get("upload_run_artifacts", True))
        self.s3_sync_every_epochs = int(s3_cfg.get("sync_every_epochs", 1))

        bucket_prefix = str(s3_cfg.get("bucket_prefix", "")).strip().rstrip("/")
        if self.s3_enabled and not bucket_prefix:
            raise ValueError("s3_sync.enabled=true requires s3_sync.bucket_prefix")

        self.s3_run_prefix = f"{bucket_prefix}/{self.run_dir.name}" if bucket_prefix else ""
        if self.s3_enabled:
            self.logger.info("S3 autosync enabled: %s", self.s3_run_prefix)

    def _s3_cp(self, local_path: Path, s3_uri: str) -> bool:
        cmd = ["aws", "s3", "cp", str(local_path), s3_uri, "--only-show-errors"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            msg = (
                f"S3 upload failed for {local_path} -> {s3_uri}: "
                f"{proc.stderr.strip() or proc.stdout.strip() or 'unknown aws error'}"
            )
            if self.s3_fail_on_error:
                raise RuntimeError(msg)
            self.logger.warning(msg)
            return False
        return True

    def _maybe_upload_run_artifacts(self) -> None:
        if not self.s3_enabled or not self.s3_upload_run_artifacts:
            return
        for p in [self.run_dir / "train.log", self.history_path]:
            if p.exists():
                self._s3_cp(p, f"{self.s3_run_prefix}/outputs/{self.run_dir.name}/{p.name}")

    def _maybe_upload_checkpoint(self, ckpt_path: Path, epoch: int, is_best: bool) -> None:
        if not self.s3_enabled:
            return
        if self.s3_sync_every_epochs > 1 and (epoch % self.s3_sync_every_epochs) != 0 and not is_best:
            return

        name = ckpt_path.name
        if name == "last.pt" and not self.s3_upload_last_each_epoch:
            return
        if name == "best.pt" and not self.s3_upload_best:
            return
        if name.startswith("epoch_") and not self.s3_upload_epoch_checkpoints:
            return

        self._s3_cp(ckpt_path, f"{self.s3_run_prefix}/checkpoints/{self.run_dir.name}/{name}")
        self._maybe_upload_run_artifacts()

    def _build_grad_scaler(self):
        if not self.use_amp:
            # Keep API-compatible no-op scaler object.
            try:
                return torch.amp.GradScaler(device="cuda", enabled=False)
            except Exception:
                return torch.cuda.amp.GradScaler(enabled=False)

        # torch>=2.3 preferred API
        try:
            return torch.amp.GradScaler(device="cuda", enabled=True)
        except Exception:
            # Backward compatibility
            return torch.cuda.amp.GradScaler(enabled=True)

    def _autocast_context(self):
        if not self.use_amp:
            return contextlib.nullcontext()
        try:
            return torch.amp.autocast(device_type="cuda", enabled=True)
        except Exception:
            return torch.cuda.amp.autocast(enabled=True)

    def _validate_teacher_cache(self) -> None:
        dataset_name = self.cfg["data"]["dataset_name"]
        processed_root = Path(self.cfg["paths"]["processed_root"])
        teacher_root = Path(self.cfg["paths"]["teacher_cache_root"]) / dataset_name
        split_path = processed_root / dataset_name / "splits.json"

        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        if not teacher_root.exists():
            raise RuntimeError(
                f"Missing teacher cache directory: {teacher_root}\n"
                "Generate teacher labels first with:\n"
                "python -m teacher.generate_teacher_labels "
                f"--processed-root {processed_root} --dataset-name {dataset_name} "
                f"--teacher-cache-root {Path(self.cfg['paths']['teacher_cache_root'])} "
                "--provider esm3 --esm-backend local --split all --device cuda"
            )

        splits = json.loads(split_path.read_text())
        missing_report = {}
        required_splits = ["train", "test"]
        if self.use_validation:
            required_splits.insert(1, "val")

        for split in required_splits:
            ids = splits.get(split, [])
            missing = [sid for sid in ids if not (teacher_root / f"{sid}.npz").exists()]
            if missing:
                missing_report[split] = {
                    "missing": len(missing),
                    "total": len(ids),
                    "examples": missing[:5],
                }

        if missing_report:
            raise RuntimeError(
                "Teacher cache is incomplete. Missing teacher labels detected:\n"
                f"{json.dumps(missing_report, indent=2)}\n"
                "Generate/continue teacher caching before training:\n"
                "python -m teacher.generate_teacher_labels "
                f"--processed-root {processed_root} --dataset-name {dataset_name} "
                f"--teacher-cache-root {Path(self.cfg['paths']['teacher_cache_root'])} "
                "--provider esm3 --esm-backend local --split all --device cuda"
            )

    def _setup_parallel_model(self) -> None:
        if self.multi_gpu and torch.cuda.device_count() > 1 and self.device.type == "cuda":
            from torch_geometric.nn import DataParallel as PyGDataParallel

            self.logger.info("Using PyG DataParallel across %d GPUs", torch.cuda.device_count())
            self.model = PyGDataParallel(self.model)
            self.parallel_mode = "pyg_dataparallel"
            self.primary_device = torch.device("cuda:0")
            self.model = self.model.to(self.primary_device)
        else:
            if self.multi_gpu and torch.cuda.device_count() <= 1:
                self.logger.warning("multi_gpu=true but only one CUDA device detected; using single GPU")
            self.parallel_mode = "single"
            self.primary_device = self.device
            self.model = self.model.to(self.primary_device)

    def _build_dataset(self, split: str) -> DistillationGraphDataset:
        cfg = self.cfg
        return DistillationGraphDataset(
            processed_root=Path(cfg["paths"]["processed_root"]),
            dataset_name=cfg["data"]["dataset_name"],
            split_name=split,
            teacher_root=Path(cfg["paths"]["teacher_cache_root"]) / cfg["data"]["dataset_name"],
            cutoff=float(cfg["graph"].get("cutoff", 8.0)),
            max_neighbors=int(cfg["graph"].get("max_neighbors", 64)),
            cache_graphs=bool(cfg["data"].get("cache_graphs", False)),
        )

    def _build_loaders(self):
        train_ds = self._build_dataset("train")
        test_ds = self._build_dataset("test")
        val_ds = self._build_dataset("val") if self.use_validation else None

        batch_size = int(self.cfg["train"].get("batch_size", 50))
        num_workers = int(self.cfg["train"].get("num_workers", 4))
        pin_memory = bool(self.device.type == "cuda")

        if self.multi_gpu and torch.cuda.device_count() > 1 and self.device.type == "cuda":
            train_loader = DataListLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = (
                DataListLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                if val_ds is not None
                else None
            )
            test_loader = DataListLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            val_loader = (
                DataLoader(
                    val_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
                if val_ds is not None
                else None
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        return train_loader, val_loader, test_loader

    def _extract_targets(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.parallel_mode == "pyg_dataparallel":
            logits = self.model(batch)
            teacher_probs = torch.cat([d.teacher_probs for d in batch], dim=0).to(self.primary_device)
            dssp_idx = torch.cat([d.dssp_idx for d in batch], dim=0).to(self.primary_device)
            return logits, teacher_probs, dssp_idx

        batch = batch.to(self.primary_device)
        logits = self.model(batch)
        if not hasattr(batch, "teacher_probs"):
            raise RuntimeError("Missing teacher_probs in graph batch; generate teacher cache first.")
        teacher_probs = batch.teacher_probs
        dssp_idx = batch.dssp_idx
        return logits, teacher_probs, dssp_idx

    def _compute_losses(
        self,
        logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        dssp_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        teacher_ce = soft_target_cross_entropy(logits, teacher_probs, reduction="mean")
        teacher_kl = teacher_kl_divergence(
            logits,
            teacher_probs,
            temperature=self.temperature,
            reduction="batchmean",
        )

        if self.teacher_loss_type == "soft_ce":
            teacher_loss = teacher_ce
        elif self.teacher_loss_type == "kl":
            teacher_loss = teacher_kl
        else:
            raise ValueError(f"Unsupported teacher_loss_type: {self.teacher_loss_type}")

        valid_dssp = dssp_idx >= 0
        if self.lambda_dssp > 0:
            if valid_dssp.any():
                dssp_loss = hard_cross_entropy(logits, dssp_idx, ignore_index=-100)
            else:
                if self.require_dssp_labels:
                    raise RuntimeError(
                        "lambda_dssp > 0 but this batch has no valid DSSP labels (all are ignore_index). "
                        "Regenerate processed data with usable DSSP labels to enable the auxiliary loss."
                    )
                dssp_loss = torch.zeros((), device=logits.device)
        else:
            dssp_loss = torch.zeros((), device=logits.device)

        total = self.lambda_teacher * teacher_loss + self.lambda_dssp * dssp_loss

        return {
            "total": total,
            "teacher_loss": teacher_loss,
            "teacher_ce": teacher_ce,
            "teacher_kl": teacher_kl,
            "dssp_loss": dssp_loss,
        }

    def _epoch_loop(self, loader, train: bool, epoch: int) -> Dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        accum = {
            "total_loss": 0.0,
            "teacher_loss": 0.0,
            "teacher_ce": 0.0,
            "teacher_kl": 0.0,
            "dssp_loss": 0.0,
            "teacher_top1_acc": 0.0,
            "dssp_acc": 0.0,
            "n_nodes": 0.0,
            "n_dssp": 0.0,
        }

        for step_idx, batch in enumerate(loader, start=1):
            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                with self._autocast_context():
                    logits, teacher_probs, dssp_idx = self._extract_targets(batch)
                    losses = self._compute_losses(logits, teacher_probs, dssp_idx)
                    total = losses["total"]

                if train:
                    self.scaler.scale(total).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.global_step += 1

            metrics = compute_teacher_metrics(logits.detach(), teacher_probs.detach(), dssp_idx.detach())
            n_nodes = float(logits.shape[0])
            n_dssp = float(metrics.get("dssp_count", 0.0))

            accum["total_loss"] += float(losses["total"].detach().item()) * n_nodes
            accum["teacher_loss"] += float(losses["teacher_loss"].detach().item()) * n_nodes
            accum["teacher_ce"] += float(losses["teacher_ce"].detach().item()) * n_nodes
            accum["teacher_kl"] += float(losses["teacher_kl"].detach().item()) * n_nodes
            accum["dssp_loss"] += float(losses["dssp_loss"].detach().item()) * n_nodes
            accum["teacher_top1_acc"] += float(metrics["teacher_top1_acc"]) * n_nodes
            if n_dssp > 0:
                accum["dssp_acc"] += float(metrics["dssp_acc"]) * n_dssp
            accum["n_nodes"] += n_nodes
            accum["n_dssp"] += n_dssp

            if train and self.wandb_run is not None and self.log_every_steps > 0:
                if (self.global_step % self.log_every_steps) == 0:
                    self.wandb_run.log(
                        {
                            "epoch": epoch,
                            "step_in_epoch": step_idx,
                            "lr": float(self.optimizer.param_groups[0]["lr"]),
                            "batch_total_loss": float(losses["total"].detach().item()),
                            "batch_teacher_loss": float(losses["teacher_loss"].detach().item()),
                            "batch_teacher_ce": float(losses["teacher_ce"].detach().item()),
                            "batch_teacher_kl": float(losses["teacher_kl"].detach().item()),
                            "batch_dssp_loss": float(losses["dssp_loss"].detach().item()),
                            "batch_teacher_top1_acc": float(metrics["teacher_top1_acc"]),
                        },
                        step=self.global_step,
                    )

        denom = max(accum["n_nodes"], 1.0)
        out = {
            "total_loss": accum["total_loss"] / denom,
            "teacher_loss": accum["teacher_loss"] / denom,
            "teacher_ce": accum["teacher_ce"] / denom,
            "teacher_kl": accum["teacher_kl"] / denom,
            "dssp_loss": accum["dssp_loss"] / denom,
            "teacher_top1_acc": accum["teacher_top1_acc"] / denom,
            "dssp_acc": accum["dssp_acc"] / max(accum["n_dssp"], 1.0) if accum["n_dssp"] > 0 else float("nan"),
            "n_nodes": accum["n_nodes"],
            "n_dssp": accum["n_dssp"],
        }
        return out

    def _save_checkpoint(self, path: Path, epoch: int, is_best: bool = False) -> None:
        if hasattr(self.model, "module"):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        torch.save(
            {
                "epoch": epoch,
                "model_state": model_state,
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state": self.scaler.state_dict(),
                "best_val": self.best_val,
                "best_epoch": self.best_epoch,
                "cfg": self.cfg,
            },
            path,
        )
        self._maybe_upload_checkpoint(path, epoch=epoch, is_best=is_best)
        if is_best:
            self.logger.info("Saved new best checkpoint: %s", path)

    def _load_checkpoint(self, path: Path) -> None:
        self.logger.info("Resuming from checkpoint: %s", path)
        ckpt = torch.load(path, map_location=self.primary_device)
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(ckpt["model_state"])
        else:
            self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.scaler.load_state_dict(ckpt.get("scaler_state", {}))

        self.start_epoch = int(ckpt["epoch"]) + 1
        self.best_val = float(ckpt.get("best_val", math.inf))
        self.best_epoch = int(ckpt.get("best_epoch", 0))

    def _write_history(self, history: List[Dict[str, float]]) -> None:
        if not history:
            return
        keys = sorted(history[0].keys())
        with self.history_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in history:
                writer.writerow(row)

    def _plot_history(self, history: List[Dict[str, float]]) -> None:
        if not history:
            return

        epochs = [row["epoch"] for row in history]

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(epochs, [row["train_teacher_ce"] for row in history], label="train")
        ax[0].plot(epochs, [row["val_teacher_ce"] for row in history], label="val")
        ax[0].set_title("Teacher Cross-Entropy")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()

        ax[1].plot(epochs, [row["train_teacher_top1_acc"] for row in history], label="train")
        ax[1].plot(epochs, [row["val_teacher_top1_acc"] for row in history], label="val")
        ax[1].set_title("Teacher Top-1 Accuracy")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()

        fig.tight_layout()
        fig.savefig(self.run_dir / "learning_curves.png", dpi=150)
        plt.close(fig)

    def fit(self) -> Dict[str, float]:
        history: List[Dict[str, float]] = []

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            train_metrics = self._epoch_loop(self.train_loader, train=True, epoch=epoch)
            if self.use_validation:
                if self.val_loader is None:
                    raise RuntimeError("use_validation=true but validation loader is missing.")
                val_metrics = self._epoch_loop(self.val_loader, train=False, epoch=epoch)
            else:
                val_metrics = {
                    "total_loss": float("nan"),
                    "teacher_loss": float("nan"),
                    "teacher_ce": float("nan"),
                    "teacher_kl": float("nan"),
                    "dssp_loss": float("nan"),
                    "teacher_top1_acc": float("nan"),
                    "dssp_acc": float("nan"),
                    "n_nodes": 0.0,
                    "n_dssp": 0.0,
                }
            self.scheduler.step()

            row = {
                "epoch": epoch,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "train_total_loss": train_metrics["total_loss"],
                "train_teacher_loss": train_metrics["teacher_loss"],
                "train_teacher_ce": train_metrics["teacher_ce"],
                "train_teacher_kl": train_metrics["teacher_kl"],
                "train_dssp_loss": train_metrics["dssp_loss"],
                "train_teacher_top1_acc": train_metrics["teacher_top1_acc"],
                "train_dssp_acc": train_metrics["dssp_acc"],
                "val_total_loss": val_metrics["total_loss"],
                "val_teacher_loss": val_metrics["teacher_loss"],
                "val_teacher_ce": val_metrics["teacher_ce"],
                "val_teacher_kl": val_metrics["teacher_kl"],
                "val_dssp_loss": val_metrics["dssp_loss"],
                "val_teacher_top1_acc": val_metrics["teacher_top1_acc"],
                "val_dssp_acc": val_metrics["dssp_acc"],
            }
            history.append(row)
            if self.wandb_run is not None:
                self.wandb_run.log(row, step=self.global_step)

            self.logger.info(
                "Epoch %d/%d | train_ce=%.5f val_ce=%.5f train_acc=%.4f val_acc=%.4f",
                epoch,
                self.max_epochs,
                train_metrics["teacher_ce"],
                val_metrics["teacher_ce"],
                train_metrics["teacher_top1_acc"],
                val_metrics["teacher_top1_acc"],
            )

            if self.use_validation:
                val_key = val_metrics["total_loss"]
                if val_key < self.best_val:
                    self.best_val = val_key
                    self.best_epoch = epoch
                    self.no_improve_epochs = 0
                    self._save_checkpoint(self.ckpt_dir / "best.pt", epoch=epoch, is_best=True)
                else:
                    self.no_improve_epochs += 1
            else:
                self.best_val = float("nan")
                self.best_epoch = epoch
                self.no_improve_epochs = 0

            self._save_checkpoint(self.ckpt_dir / "last.pt", epoch=epoch)
            if epoch % int(self.cfg["train"].get("checkpoint_every", 10)) == 0:
                self._save_checkpoint(self.ckpt_dir / f"epoch_{epoch:04d}.pt", epoch=epoch)

            self._write_history(history)

            if self.use_validation and self.no_improve_epochs >= self.patience:
                self.logger.info(
                    "Early stopping at epoch %d (best epoch=%d, best val total loss=%.6f)",
                    epoch,
                    self.best_epoch,
                    self.best_val,
                )
                break

        self._plot_history(history)

        ckpt_to_eval = self.ckpt_dir / ("best.pt" if self.use_validation else "last.pt")
        best_ckpt = torch.load(ckpt_to_eval, map_location=self.primary_device)
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(best_ckpt["model_state"])
        else:
            self.model.load_state_dict(best_ckpt["model_state"])

        test_metrics = self._epoch_loop(self.test_loader, train=False, epoch=self.best_epoch)
        final = {
            "best_epoch": self.best_epoch,
            "best_val_total_loss": self.best_val,
            "test_total_loss": test_metrics["total_loss"],
            "test_teacher_ce": test_metrics["teacher_ce"],
            "test_teacher_kl": test_metrics["teacher_kl"],
            "test_teacher_top1_acc": test_metrics["teacher_top1_acc"],
            "test_dssp_acc": test_metrics["dssp_acc"],
        }
        self.metrics_path.write_text(json.dumps(final, indent=2))
        if self.s3_enabled:
            self._s3_cp(self.metrics_path, f"{self.s3_run_prefix}/outputs/{self.run_dir.name}/{self.metrics_path.name}")
            curve_path = self.run_dir / "learning_curves.png"
            if curve_path.exists():
                self._s3_cp(curve_path, f"{self.s3_run_prefix}/outputs/{self.run_dir.name}/{curve_path.name}")
        if self.wandb_run is not None:
            self.wandb_run.summary.update(final)

        self.logger.info("Final test teacher_ce=%.5f top1=%.4f", final["test_teacher_ce"], final["test_teacher_top1_acc"])
        return final

    def _model_for_state(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _snapshot_model_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self._model_for_state().state_dict().items()}

    def _load_model_state_from_snapshot(self, snapshot: Dict[str, torch.Tensor]) -> None:
        self._model_for_state().load_state_dict(
            {k: v.to(self.primary_device) for k, v in snapshot.items()}
        )

    def active_fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader] = None,
        *,
        reset_training_state: bool = True,
        save_training_checkpoints: bool = True,
        run_final_test: bool = True,
        reload_best_weights_after_fit: bool = True,
    ) -> Dict[str, float]:
        """Train on custom loaders (e.g. an active-learning labeled subset).

        ``best_snapshot`` (when ``save_training_checkpoints`` is False): a CPU copy of weights
        at the best validation epoch, used only if ``reload_best_weights_after_fit`` is True.

        If ``reload_best_weights_after_fit`` is False, the model stays on the **last training
        epoch** instead of restoring validation-best weights.

        ``test_loader`` is only required when ``run_final_test`` is True.
        """
        if run_final_test and test_loader is None:
            raise RuntimeError("run_final_test=True requires test_loader.")
        if reset_training_state:
            self.start_epoch = 1
            self.best_val = math.inf
            self.best_epoch = 0
            self.no_improve_epochs = 0

        history: List[Dict[str, float]] = []
        best_snapshot: Optional[Dict[str, torch.Tensor]] = None
        last_completed_epoch = self.start_epoch - 1

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            last_completed_epoch = epoch
            train_metrics = self._epoch_loop(train_loader, train=True, epoch=epoch)
            if self.use_validation:
                if val_loader is None:
                    raise RuntimeError("use_validation=true but validation loader is missing.")
                val_metrics = self._epoch_loop(val_loader, train=False, epoch=epoch)
            else:
                val_metrics = {
                    "total_loss": float("nan"),
                    "teacher_loss": float("nan"),
                    "teacher_ce": float("nan"),
                    "teacher_kl": float("nan"),
                    "dssp_loss": float("nan"),
                    "teacher_top1_acc": float("nan"),
                    "dssp_acc": float("nan"),
                    "n_nodes": 0.0,
                    "n_dssp": 0.0,
                }
            self.scheduler.step()

            row = {
                "epoch": epoch,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "train_total_loss": train_metrics["total_loss"],
                "train_teacher_loss": train_metrics["teacher_loss"],
                "train_teacher_ce": train_metrics["teacher_ce"],
                "train_teacher_kl": train_metrics["teacher_kl"],
                "train_dssp_loss": train_metrics["dssp_loss"],
                "train_teacher_top1_acc": train_metrics["teacher_top1_acc"],
                "train_dssp_acc": train_metrics["dssp_acc"],
                "val_total_loss": val_metrics["total_loss"],
                "val_teacher_loss": val_metrics["teacher_loss"],
                "val_teacher_ce": val_metrics["teacher_ce"],
                "val_teacher_kl": val_metrics["teacher_kl"],
                "val_dssp_loss": val_metrics["dssp_loss"],
                "val_teacher_top1_acc": val_metrics["teacher_top1_acc"],
                "val_dssp_acc": val_metrics["dssp_acc"],
            }
            history.append(row)
            if self.wandb_run is not None:
                self.wandb_run.log(row, step=self.global_step)

            self.logger.info(
                "Epoch %d/%d | train_ce=%.5f val_ce=%.5f train_acc=%.4f val_acc=%.4f",
                epoch,
                self.max_epochs,
                train_metrics["teacher_ce"],
                val_metrics["teacher_ce"],
                train_metrics["teacher_top1_acc"],
                val_metrics["teacher_top1_acc"],
            )

            if self.use_validation:
                val_key = val_metrics["total_loss"]
                if val_key < self.best_val:
                    self.best_val = val_key
                    self.best_epoch = epoch
                    self.no_improve_epochs = 0
                    if save_training_checkpoints:
                        self._save_checkpoint(self.ckpt_dir / "best.pt", epoch=epoch, is_best=True)
                    else:
                        best_snapshot = self._snapshot_model_state()
                else:
                    self.no_improve_epochs += 1
            else:
                self.best_val = float("nan")
                self.best_epoch = epoch
                self.no_improve_epochs = 0
                if not save_training_checkpoints:
                    best_snapshot = self._snapshot_model_state()

            if save_training_checkpoints:
                self._save_checkpoint(self.ckpt_dir / "last.pt", epoch=epoch)
                if epoch % int(self.cfg["train"].get("checkpoint_every", 10)) == 0:
                    self._save_checkpoint(self.ckpt_dir / f"epoch_{epoch:04d}.pt", epoch=epoch)

            self._write_history(history)

            if self.use_validation and self.no_improve_epochs >= self.patience:
                self.logger.info(
                    "Early stopping at epoch %d (best epoch=%d, best val total loss=%.6f)",
                    epoch,
                    self.best_epoch,
                    self.best_val,
                )
                break

        self._last_completed_epoch = last_completed_epoch
        self._plot_history(history)

        if reload_best_weights_after_fit:
            if save_training_checkpoints:
                ckpt_to_eval = self.ckpt_dir / ("best.pt" if self.use_validation else "last.pt")
                best_ckpt = torch.load(ckpt_to_eval, map_location=self.primary_device)
                if hasattr(self.model, "module"):
                    self.model.module.load_state_dict(best_ckpt["model_state"])
                else:
                    self.model.load_state_dict(best_ckpt["model_state"])
            elif best_snapshot is not None:
                self._load_model_state_from_snapshot(best_snapshot)

        if not run_final_test:
            return {}

        eval_epoch = self.best_epoch if reload_best_weights_after_fit else self._last_completed_epoch
        test_metrics = self._epoch_loop(test_loader, train=False, epoch=eval_epoch)
        final = {
            "best_epoch": self.best_epoch,
            "best_val_total_loss": self.best_val,
            "test_total_loss": test_metrics["total_loss"],
            "test_teacher_ce": test_metrics["teacher_ce"],
            "test_teacher_kl": test_metrics["teacher_kl"],
            "test_teacher_top1_acc": test_metrics["teacher_top1_acc"],
            "test_dssp_acc": test_metrics["dssp_acc"],
        }
        self.metrics_path.write_text(json.dumps(final, indent=2))
        if self.s3_enabled:
            self._s3_cp(self.metrics_path, f"{self.s3_run_prefix}/outputs/{self.run_dir.name}/{self.metrics_path.name}")
            curve_path = self.run_dir / "learning_curves.png"
            if curve_path.exists():
                self._s3_cp(curve_path, f"{self.s3_run_prefix}/outputs/{self.run_dir.name}/{curve_path.name}")
        if self.wandb_run is not None:
            self.wandb_run.summary.update(final)

        self.logger.info("Final test teacher_ce=%.5f top1=%.4f", final["test_teacher_ce"], final["test_teacher_top1_acc"])
        return final

    def _active_learning_loop(self, selection_method: str) -> Dict[str, float]:
        """Grow the labeled pool to 10k (or size of train split). No test or extra metrics per round.

        Each round's ``active_fit`` restores **best validation** weights. After the last round,
        evaluate train loss on the **full** AL train subset, val loss, and test metrics using that
        **best-epoch** model. Saves ``active_learning_final.pt`` and ``metrics_active_learning_final.json``.
        """
        train_ds = self._build_dataset("train")
        val_loader = self.val_loader
        test_loader = self.test_loader
        if test_loader is None:
            raise RuntimeError("Active learning final evaluation requires test_loader (build loaders in __init__).")

        target_labeled = 10_000
        total_train_size = min(target_labeled, len(train_ds))
        initial = min(2500, len(train_ds))
        rng_idx = np.random.choice(len(train_ds), size=initial, replace=False)
        active_train_ids: List[int] = [int(x) for x in np.atleast_1d(rng_idx)]
        observed = set(active_train_ids)
        unobserved_train_ids = [i for i in range(len(train_ds)) if i not in observed]

        round_idx = 0
        while len(active_train_ids) < total_train_size:
            active_train_ds = Subset(train_ds, indices=active_train_ids)
            active_train_loader = DataLoader(
                active_train_ds, batch_size=self.batch_size, shuffle=True
            )
            self.active_fit(
                active_train_loader,
                val_loader,
                None,
                reset_training_state=True,
                save_training_checkpoints=False,
                run_final_test=False,
                reload_best_weights_after_fit=True,
            )
            round_idx += 1

            remaining_slots = total_train_size - len(active_train_ids)
            if remaining_slots <= 0 or not unobserved_train_ids:
                break

            if selection_method == "expected_model_change":
                new_active_train_ids = expected_model_change(
                    active_train_ds, unobserved_train_ids, train_ds, Trainer
                )
            elif selection_method == "diversity_sampling_with_k_means":
                new_active_train_ids = diversity_sampling_with_k_means(
                    self, unobserved_train_ids, train_ds, k=10
                )
            elif selection_method == "diversity_sampling_hierarchical_clustering":
                new_active_train_ids = diversity_sampling_hierarchical_clustering(
                    self, unobserved_train_ids, train_ds, k=10
                )
            elif selection_method == "passive_learning":
                new_active_train_ids = passive_learning(unobserved_train_ids)
            else:
                raise ValueError(f"Invalid selection method: {selection_method}")

            if not isinstance(new_active_train_ids, list):
                new_active_train_ids = list(np.atleast_1d(new_active_train_ids))
            new_active_train_ids = new_active_train_ids[:remaining_slots]

            picked = set(new_active_train_ids)
            active_train_ids = active_train_ids + list(new_active_train_ids)
            unobserved_train_ids = [i for i in unobserved_train_ids if i not in picked]

        if round_idx == 0:
            raise RuntimeError("No active-learning rounds ran.")

        best_ep = int(self.best_epoch)
        final_train_ds = Subset(train_ds, indices=active_train_ids)
        final_train_loader = DataLoader(
            final_train_ds, batch_size=self.batch_size, shuffle=False
        )
        train_metrics = self._epoch_loop(final_train_loader, train=False, epoch=best_ep)
        val_metrics = (
            self._epoch_loop(val_loader, train=False, epoch=best_ep)
            if val_loader is not None
            else None
        )
        test_metrics = self._epoch_loop(test_loader, train=False, epoch=best_ep)

        self._save_checkpoint(self.ckpt_dir / "active_learning_final.pt", epoch=best_ep)

        final_al = {
            "best_epoch": best_ep,
            "labeled_train_size": len(active_train_ids),
            "al_rounds": round_idx,
            "train_total_loss": train_metrics["total_loss"],
            "val_total_loss": val_metrics["total_loss"] if val_metrics is not None else float("nan"),
            "test_total_loss": test_metrics["total_loss"],
            "test_teacher_ce": test_metrics["teacher_ce"],
            "test_teacher_kl": test_metrics["teacher_kl"],
            "test_teacher_top1_acc": test_metrics["teacher_top1_acc"],
            "test_dssp_acc": test_metrics["dssp_acc"],
        }
        out_path = self.run_dir / "metrics_active_learning_final.json"
        out_path.write_text(json.dumps(final_al, indent=2))

        self.logger.info(
            "Active learning done: %d rounds, %d labeled train, best_epoch=%d. Metrics: %s",
            round_idx,
            len(active_train_ids),
            best_ep,
            out_path,
        )
        return {k: float(v) for k, v in final_al.items()}

    @torch.no_grad()
    def _graph_embeddings_for_indices(
        self,
        train_ds: DistillationGraphDataset,
        indices: List[int],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Build one fixed-size vector per graph for clustering-based selection.

        Primary path: mean-pool node embeddings from the current model's `encoder` (if present).
        Fallback path: simple handcrafted stats from graph inputs (aa/atom histograms + geometry).
        """
        if not indices:
            return np.zeros((0, 1), dtype=np.float32)

        bs = int(batch_size or self.batch_size or 50)
        subset = Subset(train_ds, indices=indices)
        loader = DataLoader(subset, batch_size=min(bs, len(subset)), shuffle=False)

        model = getattr(self.model, "module", self.model)
        use_encoder = hasattr(model, "encoder")

        out_chunks: List[np.ndarray] = []
        for batch in loader:
            batch = batch.to(self.primary_device)

            if not hasattr(batch, "batch"):
                batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long, device=batch.aa_idx.device)

            if use_encoder:
                h = model.encoder(batch.aa_idx, batch.atom_idx)
                g = global_mean_pool(h, batch.batch)  # [num_graphs, hidden_dim]
                out_chunks.append(g.detach().float().cpu().numpy())
                continue

            # Fallback: graph-level handcrafted features (works for any model).
            num_graphs = int(batch.batch.max().item()) + 1 if batch.num_nodes > 0 else batch.num_graphs
            feats = []
            for gi in range(num_graphs):
                mask = batch.batch == gi
                aa = batch.aa_idx[mask]
                atom = batch.atom_idx[mask]
                pos = batch.pos[mask] if hasattr(batch, "pos") else None

                aa_hist = torch.bincount(aa.clamp_min(0), minlength=21).float()
                atom_hist = torch.bincount(atom.clamp_min(0), minlength=3).float()
                aa_hist = aa_hist / (aa_hist.sum() + 1e-8)
                atom_hist = atom_hist / (atom_hist.sum() + 1e-8)

                if pos is not None and pos.numel() > 0:
                    pos_mean = pos.mean(dim=0)
                    pos_std = pos.std(dim=0, unbiased=False)
                else:
                    pos_mean = torch.zeros(3, device=aa.device)
                    pos_std = torch.zeros(3, device=aa.device)

                n_nodes = torch.tensor([float(aa.numel())], device=aa.device)
                feats.append(torch.cat([aa_hist, atom_hist, pos_mean, pos_std, n_nodes], dim=0))

            out_chunks.append(torch.stack(feats, dim=0).detach().float().cpu().numpy())

        return np.concatenate(out_chunks, axis=0)


###################### sample selection for active learning ######################
# Expected Model Change (EMC), diversity (k-means), passive learning


def expected_model_change(self, train_ds: Subset, unobserved_train_ids: List[int], test_ds: Subset) -> float:
    # for each sample in the unobserved_train_ids, I need to compute the expected model change
    unobserved_train_ds = Subset(train_ds, indices=unobserved_train_ids)
    logits = Trainer.active_fit(unobserved_train_ds)
    teacher_probs = unobserved_train_ds.teacher_probs
    losses = soft_target_cross_entropy(logits, teacher_probs)
    grads = torch.autograd.grad(losses, Trainer.model.output_layer.weight)[0]
    score = grads.norm()
    return score


def diversity_sampling_with_k_means(
    trainer: Trainer, unobserved_train_ids: List[int], train_ds: DistillationGraphDataset, k: int
) -> List[int]:
    if not unobserved_train_ids:
        return []
    k_eff = int(min(k, len(unobserved_train_ids)))
    if k_eff <= 0:
        return []
    if k_eff >= len(unobserved_train_ids):
        return list(unobserved_train_ids)

    X = trainer._graph_embeddings_for_indices(train_ds, unobserved_train_ids)  # [N, D]
    kmeans = KMeans(n_clusters=k_eff, random_state=0, n_init="auto").fit(X)
    centers = kmeans.cluster_centers_  # [k, D]

    picked: List[int] = []
    for ci in range(k_eff):
        members = np.where(kmeans.labels_ == ci)[0]
        if members.size == 0:
            continue
        d = np.linalg.norm(X[members] - centers[ci], axis=1)
        chosen_local = int(members[int(np.argmin(d))])
        picked.append(int(unobserved_train_ids[chosen_local]))

    # In rare cases (empty clusters), top up with closest-to-any-center points.
    if len(picked) < k_eff:
        picked_set = set(picked)
        d_all = np.min(np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2), axis=1)
        order = np.argsort(d_all)
        for idx in order:
            gid = int(unobserved_train_ids[int(idx)])
            if gid not in picked_set:
                picked.append(gid)
                picked_set.add(gid)
            if len(picked) >= k_eff:
                break

    return picked[:k_eff]

def diversity_sampling_hierarchical_clustering(
    trainer: Trainer, unobserved_train_ids: List[int], train_ds: DistillationGraphDataset, k: int
) -> List[int]:
    if not unobserved_train_ids:
        return []
    k_eff = int(min(k, len(unobserved_train_ids)))
    if k_eff <= 0:
        return []
    if k_eff >= len(unobserved_train_ids):
        return list(unobserved_train_ids)

    X = trainer._graph_embeddings_for_indices(train_ds, unobserved_train_ids)  # [N, D]
    labels = AgglomerativeClustering(n_clusters=k_eff).fit_predict(X)

    picked: List[int] = []
    for ci in range(k_eff):
        members = np.where(labels == ci)[0]
        if members.size == 0:
            continue
        centroid = X[members].mean(axis=0, keepdims=False)
        d = np.linalg.norm(X[members] - centroid[None, :], axis=1)
        chosen_local = int(members[int(np.argmin(d))])
        picked.append(int(unobserved_train_ids[chosen_local]))

    # Top up deterministically if any cluster ended empty.
    if len(picked) < k_eff:
        picked_set = set(picked)
        remaining = [i for i in unobserved_train_ids if i not in picked_set]
        picked.extend(remaining[: (k_eff - len(picked))])

    return picked[:k_eff]
def passive_learning(unobserved_train_ids: List[int]) -> List[int]:
    return np.random.choice(unobserved_train_ids, size=500, replace=False).tolist()
