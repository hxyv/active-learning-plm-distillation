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

from data.pyg_dataset import DistillationGraphDataset
from eval.metrics import compute_teacher_metrics
from models.factory import build_model
from train.losses import hard_cross_entropy, soft_target_cross_entropy, teacher_kl_divergence


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
        self.log_every_steps = int(cfg["train"].get("log_every_steps", 100))
        self.global_step = 0

        self._validate_teacher_cache()
        self.train_loader, self.val_loader, self.test_loader = self._build_loaders()

        self.model = build_model(cfg)
        self._setup_parallel_model()

        train_cfg = cfg["train"]
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
        for split in ["train", "val", "test"]:
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
        val_ds = self._build_dataset("val")
        test_ds = self._build_dataset("test")

        batch_size = int(self.cfg["train"].get("batch_size", 50))
        num_workers = int(self.cfg["train"].get("num_workers", 4))
        pin_memory = bool(self.device.type == "cuda")

        if self.multi_gpu and torch.cuda.device_count() > 1 and self.device.type == "cuda":
            train_loader = DataListLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = DataListLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            test_loader = DataListLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
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
        if self.lambda_dssp > 0 and valid_dssp.any():
            dssp_loss = hard_cross_entropy(logits, dssp_idx, ignore_index=-100)
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
            val_metrics = self._epoch_loop(self.val_loader, train=False, epoch=epoch)
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

            val_key = val_metrics["total_loss"]
            if val_key < self.best_val:
                self.best_val = val_key
                self.best_epoch = epoch
                self.no_improve_epochs = 0
                self._save_checkpoint(self.ckpt_dir / "best.pt", epoch=epoch, is_best=True)
            else:
                self.no_improve_epochs += 1

            self._save_checkpoint(self.ckpt_dir / "last.pt", epoch=epoch)
            if epoch % int(self.cfg["train"].get("checkpoint_every", 10)) == 0:
                self._save_checkpoint(self.ckpt_dir / f"epoch_{epoch:04d}.pt", epoch=epoch)

            self._write_history(history)

            if self.no_improve_epochs >= self.patience:
                self.logger.info(
                    "Early stopping at epoch %d (best epoch=%d, best val total loss=%.6f)",
                    epoch,
                    self.best_epoch,
                    self.best_val,
                )
                break

        self._plot_history(history)

        best_ckpt = torch.load(self.ckpt_dir / "best.pt", map_location=self.primary_device)
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
