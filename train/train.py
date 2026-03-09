"""CLI entrypoint for training the distillation baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from train.config_utils import load_config, save_config
from train.trainer import Trainer
from train.utils import infer_device, make_run_dir, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train backbone-only GNN on ESM3 SS8 distillation")
    parser.add_argument("--config", type=Path, required=True, help="YAML config")
    parser.add_argument("--run-name", type=str, default="baseline_dispef_m")
    return parser.parse_args()


def maybe_init_wandb(cfg: dict, run_name: str, run_dir: Path, logger):
    wb_cfg = cfg.get("wandb", {})
    if not wb_cfg.get("enabled", False):
        return None
    try:
        import wandb
    except Exception as exc:
        logger.warning("wandb enabled in config but package import failed: %s", exc)
        return None

    try:
        run = wandb.init(
            project=wb_cfg.get("project", "pLM_KD"),
            entity=wb_cfg.get("entity", None),
            name=wb_cfg.get("name", run_name),
            config=cfg,
            dir=str(run_dir),
            notes=wb_cfg.get("notes", ""),
            tags=wb_cfg.get("tags", []),
        )
        logger.info(
            "Initialized W&B run: entity=%s project=%s name=%s",
            wb_cfg.get("entity", None),
            wb_cfg.get("project", "pLM_KD"),
            wb_cfg.get("name", run_name),
        )
        return run
    except Exception as exc:
        logger.warning("W&B init failed; continuing without W&B logging: %s", exc)
        return None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(int(cfg["train"].get("seed", 42)))

    run_dir = make_run_dir(Path(cfg["paths"]["outputs_root"]), args.run_name)
    logger = setup_logging(run_dir / "train.log")
    save_config(cfg, run_dir / "config_resolved.yaml")

    device = infer_device(cfg["train"].get("device", "auto"))
    logger.info("Run directory: %s", run_dir)
    logger.info("Device: %s", device)

    wandb_run = maybe_init_wandb(cfg, args.run_name, run_dir, logger)
    try:
        trainer = Trainer(cfg=cfg, run_dir=run_dir, logger=logger, device=device, wandb_run=wandb_run)
        trainer.fit()
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
