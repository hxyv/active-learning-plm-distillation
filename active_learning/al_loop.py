"""Active learning loop CLI.

Usage
-----
Passive (random) baseline::

    python -m active_learning.al_loop \\
        --config configs/al_psc_dispef_m.yaml \\
        --strategy random \\
        --output-dir /path/to/outputs/al/passive_run1 \\
        --run-name psc_al_passive

Resume an interrupted run::

    python -m active_learning.al_loop \\
        --config configs/al_psc_dispef_m.yaml \\
        --strategy random \\
        --output-dir /path/to/outputs/al/passive_run1 \\
        --run-name psc_al_passive \\
        --resume
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from active_learning.acquisition import mc_dropout_acquisition, random_acquisition
from active_learning.pool_manager import ALPoolManager
from train.config_utils import load_config, save_config
from train.trainer import Trainer
from train.utils import infer_device, make_run_dir, set_seed, setup_logging

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Strategy registry
# ------------------------------------------------------------------

_STRATEGIES: Dict[str, Callable] = {
    "random": random_acquisition,
    "mc_dropout": mc_dropout_acquisition,
}


def get_acquisition_fn(strategy: str) -> Callable:
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Available: {sorted(_STRATEGIES.keys())}"
        )
    return _STRATEGIES[strategy]


# ------------------------------------------------------------------
# W&B helpers (mirrors train/train.py)
# ------------------------------------------------------------------

def maybe_init_wandb(cfg: dict, run_name: str, run_dir: Path, logger_):
    wb_cfg = cfg.get("wandb", {})
    if not wb_cfg.get("enabled", False):
        return None
    try:
        import wandb
    except Exception as exc:
        logger_.warning("wandb enabled but import failed: %s", exc)
        return None
    try:
        run = wandb.init(
            project=wb_cfg.get("project", "pLM_KD"),
            entity=wb_cfg.get("entity", None) or None,
            name=run_name,
            config=cfg,
            dir=str(run_dir),
            notes=wb_cfg.get("notes", ""),
            tags=wb_cfg.get("tags", []),
        )
        return run
    except Exception as exc:
        logger_.warning("W&B init failed; continuing without W&B: %s", exc)
        return None


# ------------------------------------------------------------------
# Results aggregation
# ------------------------------------------------------------------

def save_round_summary(
    output_dir: Path,
    round_idx: int,
    metrics: Dict,
    pool_manager: ALPoolManager,
) -> None:
    summary = {
        "round": round_idx,
        "num_train": len(pool_manager.get_train_ids()),
        "num_pool": len(pool_manager.get_pool_ids()),
        "metrics": metrics,
    }
    path = output_dir / f"round_{round_idx:02d}" / "round_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))


def aggregate_results(output_dir: Path) -> None:
    results = []
    for summary_path in sorted((output_dir).glob("round_*/round_summary.json")):
        results.append(json.loads(summary_path.read_text()))
    out_path = output_dir / "al_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Aggregated AL results written to %s", out_path)


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def run_al_loop(
    cfg: dict,
    strategy: str,
    output_dir: Path,
    run_name: str,
    resume: bool,
) -> None:
    al_cfg = cfg.get("active_learning", {})
    pool_size = int(al_cfg.get("pool_size", 12000))
    initial_train_size = int(al_cfg.get("initial_train_size", 2500))
    budget_per_round = int(al_cfg.get("budget_per_round", 500))
    num_rounds = int(al_cfg.get("num_rounds", 15))
    seed = int(al_cfg.get("seed", cfg["train"].get("seed", 42)))

    dataset_name = cfg["data"]["dataset_name"]
    processed_root = Path(cfg["paths"]["processed_root"])

    pool_manager = ALPoolManager(
        processed_root=processed_root,
        dataset_name=dataset_name,
        al_output_dir=output_dir,
        pool_size=pool_size,
        initial_train_size=initial_train_size,
        budget_per_round=budget_per_round,
        seed=seed,
    )

    if not pool_manager.initialized:
        pool_manager.initialize()
    elif not resume:
        raise RuntimeError(
            f"Output directory {output_dir} already contains an AL state. "
            "Pass --resume to continue, or choose a different --output-dir."
        )

    acquisition_fn = get_acquisition_fn(strategy)
    # Use a separate RNG seeded per-round for reproducible acquisition
    base_rng = np.random.default_rng(seed)

    device = infer_device(cfg["train"].get("device", "auto"))

    # Determine which rounds are already completed
    start_round = pool_manager.current_round
    logger.info(
        "Starting AL loop: strategy=%s, rounds=%d, resume_from_round=%d",
        strategy, num_rounds, start_round,
    )

    for round_idx in range(start_round, num_rounds + 1):
        logger.info("--- Round %d / %d ---", round_idx, num_rounds)
        logger.info(
            "  train=%d  pool=%d",
            len(pool_manager.get_train_ids()),
            len(pool_manager.get_pool_ids()),
        )

        # Build round-specific config: deep copy + override splits_file
        round_cfg = copy.deepcopy(cfg)
        round_cfg["data"]["splits_file"] = str(pool_manager.get_current_splits_file(round_idx))
        round_cfg["train"]["resume_checkpoint"] = ""  # always train from scratch

        # Create run directory for this round
        round_run_name = f"{run_name}_round{round_idx:02d}"
        round_run_dir = output_dir / f"round_{round_idx:02d}"
        round_run_dir.mkdir(parents=True, exist_ok=True)
        round_logger = setup_logging(round_run_dir / "train.log")
        save_config(round_cfg, round_run_dir / "config_resolved.yaml")

        set_seed(seed + round_idx)

        # Override wandb run name per round
        if round_cfg.get("wandb", {}).get("enabled", False):
            round_cfg["wandb"]["name"] = round_run_name

        wandb_run = maybe_init_wandb(round_cfg, round_run_name, round_run_dir, round_logger)
        try:
            trainer = Trainer(
                cfg=round_cfg,
                run_dir=round_run_dir,
                logger=round_logger,
                device=device,
                wandb_run=wandb_run,
            )
            metrics = trainer.fit()
        finally:
            if wandb_run is not None:
                wandb_run.finish()

        save_round_summary(output_dir, round_idx, metrics, pool_manager)

        # Acquire next batch (not needed after the last round)
        if round_idx < num_rounds:
            pool_ids = pool_manager.get_pool_ids()
            if not pool_ids:
                logger.warning("Pool is empty after round %d; stopping early.", round_idx)
                break
            # Advance the rng deterministically per round
            round_rng = np.random.default_rng(seed + round_idx)
            selected = acquisition_fn(
                pool_ids,
                budget_per_round,
                round_rng,
                cfg=round_cfg,
                checkpoint_path=Path(metrics["checkpoint_path"]),
                device=device,
            )
            pool_manager.advance_round(selected)

    aggregate_results(output_dir)
    logger.info("AL loop complete.")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Active learning loop for PLM distillation")
    parser.add_argument("--config", type=Path, required=True, help="YAML config (al_psc_dispef_m.yaml)")
    parser.add_argument(
        "--strategy",
        type=str,
        default="random",
        choices=list(_STRATEGIES.keys()),
        help="Acquisition strategy",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Root directory for this AL run")
    parser.add_argument("--run-name", type=str, default="al_run", help="W&B / logging run name prefix")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing AL run from the last completed round",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    root_logger = setup_logging(args.output_dir / "al_loop.log")

    root_logger.info("AL loop config: %s", args.config)
    root_logger.info("Strategy: %s", args.strategy)
    root_logger.info("Output dir: %s", args.output_dir)

    run_al_loop(
        cfg=cfg,
        strategy=args.strategy,
        output_dir=args.output_dir,
        run_name=args.run_name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
