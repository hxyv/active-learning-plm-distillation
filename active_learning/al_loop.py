"""Active learning loop CLI.

Usage
-----
Expected model change (EMC):

    python -m active_learning.al_loop \\
        --config configs/al_psc_dispef_m.yaml \\
        --strategy emc \\
        --output-dir /path/to/outputs/al/emc_run1 \\
        --run-name psc_al_emc

Diversity (agglomerative clustering on graph-level embeddings from the AL candidate pool)::

    python -m active_learning.al_loop \\
        --config configs/al_psc_dispef_m.yaml \\
        --strategy diversity \\
        --output-dir /path/to/outputs/al/div_run1 \\
        --run-name psc_al_div

Diversity with propagated node features before pooling::

    python -m active_learning.al_loop \\
        --config configs/al_psc_dispef_m.yaml \\
        --strategy diversity \\
        --output-dir /path/to/outputs/al/div_prop_run1 \\
        --run-name psc_al_div_prop \\
        --Use_propagation

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
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from active_learning.pool_manager import ALPoolManager
from active_learning.protein_meta import load_protein_meta, ss8_composition_of_selected
from train.config_utils import load_config, save_config
from train.trainer import Trainer
from train.utils import infer_device, make_run_dir, set_seed, setup_logging

from active_learning.acquisition import (
    diversity_acquisition,
    emc_acquisition,
    get_original_node_features,
    mc_dropout_acquisition,
    propagate_graph_embeddings_all,
    random_acquisition,
)
from data.pyg_dataset import DistillationGraphDataset

logger = logging.getLogger(__name__)


# Strategy registry
_STRATEGIES: Dict[str, Callable] = {
    "random": random_acquisition,
    "mc_dropout": mc_dropout_acquisition,
    "emc": emc_acquisition,
    "diversity": diversity_acquisition,
}

# error reminder
def get_acquisition_fn(strategy: str) -> Callable:
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Available: {sorted(_STRATEGIES.keys())}"
        )
    return _STRATEGIES[strategy]


# W&B helpers
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

def _sum_residues(protein_ids: List[str], processed_root: Path, dataset_name: str) -> int:
    """Total residue count across a list of proteins (0 if metadata missing)."""
    if not protein_ids:
        return 0
    meta = load_protein_meta(processed_root, dataset_name, protein_ids)
    return int(sum(meta[pid]["num_residues"] for pid in protein_ids if pid in meta))


def _jaccard(a: List[str], b: List[str]) -> float:
    """Jaccard similarity between two ID lists."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return float("nan")
    return len(sa & sb) / max(1, len(sa | sb))


def save_round_summary(
    output_dir: Path,
    round_idx: int,
    metrics: Dict,
    pool_manager: ALPoolManager,
    selected_ids: Optional[List[str]] = None,
    processed_root: Optional[Path] = None,
    dataset_name: Optional[str] = None,
    train_wall_time_s: Optional[float] = None,
    acquisition_wall_time_s: Optional[float] = None,
    acquisition_diagnostics: Optional[Dict] = None,
    random_overlap_jaccard: Optional[float] = None,
) -> None:
    """save results of each round"""
    train_ids = pool_manager.get_train_ids()
    summary: Dict = {
        "round": round_idx,
        "num_train": len(train_ids),
        "num_pool": len(pool_manager.get_pool_ids()),
        "metrics": metrics,
    }
    if processed_root and dataset_name:
        summary["num_train_residues"] = _sum_residues(train_ids, processed_root, dataset_name)
    if selected_ids and processed_root and dataset_name:
        summary["selected_ss8_composition"] = ss8_composition_of_selected(
            selected_ids, processed_root, dataset_name
        )
        summary["num_selected_residues"] = _sum_residues(selected_ids, processed_root, dataset_name)
    if train_wall_time_s is not None:
        summary["train_wall_time_s"] = float(train_wall_time_s)
    if acquisition_wall_time_s is not None:
        summary["acquisition_wall_time_s"] = float(acquisition_wall_time_s)
    if acquisition_diagnostics is not None:
        summary["acquisition"] = acquisition_diagnostics
    if random_overlap_jaccard is not None:
        summary["random_overlap_jaccard"] = float(random_overlap_jaccard)
    path = output_dir / f"round_{round_idx:02d}" / "round_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))

def aggregate_results(output_dir: Path) -> None:
    """aggregate results of all rounds"""
    results = []
    for summary_path in sorted((output_dir).glob("round_*/round_summary.json")):
        results.append(json.loads(summary_path.read_text()))
    out_path = output_dir / "al_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Aggregated AL results written to %s", out_path)


# Main loop
def run_al_loop(
    cfg: dict,
    strategy: str,
    output_dir: Path,
    run_name: str,
    resume: bool,
) -> None:
    """run the active learning loop"""
    # get the configuration
    al_cfg = cfg.get("active_learning", {})
    pool_size = int(al_cfg.get("pool_size", 12000))
    initial_train_size = int(al_cfg.get("initial_train_size", 2500))
    budget_per_round = int(al_cfg.get("budget_per_round", 500))
    num_rounds = int(al_cfg.get("num_rounds", 15))
    seed = int(al_cfg.get("seed", cfg["train"].get("seed", 42)))

    dataset_name = cfg["data"]["dataset_name"]
    processed_root = Path(cfg["paths"]["processed_root"])

    # initialize the pool manager
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

    # get the acquisition function
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

        # Build round-specific config
        round_cfg = copy.deepcopy(cfg)
        round_cfg["data"]["splits_file"] = str(pool_manager.get_current_splits_file(round_idx))
        round_cfg["train"]["resume_checkpoint"] = ""  # always train from scratch
        # Namespace checkpoints under the AL run directory so rounds from
        # different runs don't collide (trainer uses checkpoints_root/run_dir.name).
        round_cfg["paths"]["checkpoints_root"] = str(
            Path(cfg["paths"]["checkpoints_root"]) / run_name
        )

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
        train_start = time.perf_counter()
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
        train_wall_time_s = time.perf_counter() - train_start

        # Acquire next batch (not needed after the last round)
        selected: List[str] = []
        acquisition_diagnostics: Optional[Dict] = None
        acquisition_wall_time_s: Optional[float] = None
        random_overlap_jaccard: Optional[float] = None
        if round_idx < num_rounds:
            pool_ids = pool_manager.get_pool_ids()
            if not pool_ids:
                logger.warning("Pool is empty after round %d; stopping early.", round_idx)
                save_round_summary(
                    output_dir, round_idx, metrics, pool_manager,
                    train_wall_time_s=train_wall_time_s,
                    processed_root=processed_root, dataset_name=dataset_name,
                )
                break
            # Advance the rng deterministically per round
            round_rng = np.random.default_rng(seed + round_idx)
            acq_start = time.perf_counter()
            selected, acquisition_diagnostics = acquisition_fn(
                pool_ids,
                budget_per_round,
                round_rng,
                cfg=round_cfg,
                checkpoint_path=Path(metrics["checkpoint_path"]),
                device=device,
            )
            acquisition_wall_time_s = time.perf_counter() - acq_start

            # Overlap of this strategy's pick with an independent random draw
            # from the same pool. For strategy='random' this just measures
            # sampling noise across two independent random seeds; still a
            # useful calibration anchor for other strategies.
            overlap_rng = np.random.default_rng(seed + round_idx + 10_000_000)
            random_ref, _ = random_acquisition(pool_ids, budget_per_round, overlap_rng)
            random_overlap_jaccard = _jaccard(selected, random_ref)

            pool_manager.advance_round(selected)

        save_round_summary(
            output_dir, round_idx, metrics, pool_manager,
            selected_ids=selected if selected else None,
            processed_root=processed_root,
            dataset_name=dataset_name,
            train_wall_time_s=train_wall_time_s,
            acquisition_wall_time_s=acquisition_wall_time_s,
            acquisition_diagnostics=acquisition_diagnostics,
            random_overlap_jaccard=random_overlap_jaccard,
        )

        # Print compact round summary
        acc = metrics.get("test_teacher_top1_acc", float("nan"))
        ce = metrics.get("test_teacher_ce", float("nan"))
        n_res = _sum_residues(pool_manager.get_train_ids(), processed_root, dataset_name)
        logger.info(
            "Round %02d | labeled=%d proteins (%d residues) | test_acc=%.4f | "
            "test_ce=%.4f | train_wall=%.1fs | acq_wall=%s",
            round_idx, len(pool_manager.get_train_ids()), n_res, acc, ce,
            train_wall_time_s,
            f"{acquisition_wall_time_s:.1f}s" if acquisition_wall_time_s is not None else "—",
        )
        per_class = metrics.get("test_per_class_acc", {})
        if per_class:
            pc_str = "  ".join(f"{cls}:{per_class.get(cls, float('nan')):.3f}" for cls in per_class)
            logger.info("  per-class test acc: %s", pc_str)
        if selected:
            from active_learning.protein_meta import ss8_composition_of_selected
            comp = ss8_composition_of_selected(selected, processed_root, dataset_name)
            comp_str = "  ".join(f"{cls}:{v:.2f}" for cls, v in comp.items())
            logger.info("  acquired SS8 composition: %s", comp_str)
            if random_overlap_jaccard is not None:
                logger.info("  jaccard(selected, random_ref) = %.3f", random_overlap_jaccard)
            if acquisition_diagnostics and "pool_stats" in acquisition_diagnostics:
                ps = acquisition_diagnostics["pool_stats"]
                logger.info(
                    "  pool variance: mean=%.4e std=%.4e p10=%.4e p50=%.4e p90=%.4e",
                    ps["variance"]["mean"], ps["variance"]["std"],
                    ps["variance"]["p10"], ps["variance"]["p50"], ps["variance"]["p90"],
                )
                logger.info(
                    "  pool BALD:     mean=%.4e std=%.4e p10=%.4e p50=%.4e p90=%.4e",
                    ps["bald"]["mean"], ps["bald"]["std"],
                    ps["bald"]["p10"], ps["bald"]["p50"], ps["bald"]["p90"],
                )

    aggregate_results(output_dir)
    logger.info("AL loop complete.")

def run_al_loop_diversity(
    cfg: dict,
    strategy: str,
    output_dir: Path,
    run_name: str,
    resume: bool,
    Use_propagation: bool = False,
) -> None:
    """run the active learning loop for diversity strategy only"""
    # get the configuration
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

    graph_embedding_by_id: Optional[Dict[str, np.ndarray]] = None
    if strategy == "diversity":
        # get the pool candidates
        original_splits_path = processed_root / dataset_name / "splits.json"
        splits_json = json.loads(original_splits_path.read_text())
        all_train_ids = sorted(splits_json.get("train", []))
        pool_candidates = all_train_ids[:pool_size]

        cand_splits_path = output_dir / "_diversity_pool_candidates_splits.json"
        cand_splits_path.parent.mkdir(parents=True, exist_ok=True)
        cand_splits_path.write_text(json.dumps({"pool_candidates": pool_candidates}))

        div_ds = DistillationGraphDataset(
            processed_root=processed_root,
            dataset_name=dataset_name,
            split_name="pool_candidates",
            teacher_root=None,
            cutoff=float(cfg["graph"].get("cutoff", 8.0)),
            max_neighbors=int(cfg["graph"].get("max_neighbors", 64)),
            cache_graphs=False,
            splits_file=cand_splits_path,
        )
        # get the graph embeddings, may propagate the node features
        if Use_propagation:
            per_graph = propagate_graph_embeddings_all(div_ds)
        else:
            per_graph = get_original_node_features(div_ds)

        # average pooling
        graph_embedding_by_id = {
            sid: emb.mean(axis=0).astype(np.float32)
            for sid, emb in zip(div_ds.sample_ids, per_graph)
        }
    # get the acquisition function
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
        # Namespace checkpoints under the AL run directory so rounds from
        # different runs don't collide (trainer uses checkpoints_root/run_dir.name).
        round_cfg["paths"]["checkpoints_root"] = str(
            Path(cfg["paths"]["checkpoints_root"]) / run_name
        )

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

        # Acquire next batch (not needed after the last round)
        selected: List[str] = []
        if round_idx < num_rounds:
            pool_ids = pool_manager.get_pool_ids()
            if not pool_ids:
                logger.warning("Pool is empty after round %d; stopping early.", round_idx)
                save_round_summary(output_dir, round_idx, metrics, pool_manager)
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
                graph_embedding_by_id=graph_embedding_by_id,
            )
            pool_manager.advance_round(selected)

        save_round_summary(
            output_dir, round_idx, metrics, pool_manager,
            selected_ids=selected if selected else None,
            processed_root=processed_root,
            dataset_name=dataset_name,
        )

        # Print compact round summary
        acc = metrics.get("test_teacher_top1_acc", float("nan"))
        ce = metrics.get("test_teacher_ce", float("nan"))
        logger.info(
            "Round %02d | labeled=%d | test_acc=%.4f | test_ce=%.4f",
            round_idx, len(pool_manager.get_train_ids()), acc, ce,
        )
        if selected:
            from active_learning.protein_meta import ss8_composition_of_selected
            comp = ss8_composition_of_selected(selected, processed_root, dataset_name)
            comp_str = "  ".join(f"{cls}:{v:.2f}" for cls, v in comp.items())
            logger.info("  acquired SS8 composition: %s", comp_str)

    aggregate_results(output_dir)
    logger.info("AL loop complete.")


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
    parser.add_argument(
        "--Use_propagation",
        action="store_true",
        default=False,
        help="Use propagation to get node embeddings",
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

    if args.strategy == "diversity":
        run_al_loop_diversity(
            cfg=cfg,
            strategy=args.strategy,
            output_dir=args.output_dir,
            run_name=args.run_name,
            resume=args.resume,
            Use_propagation=args.Use_propagation, # default is False
        )
    else:
        run_al_loop(
          cfg=cfg,
          strategy=args.strategy,
          output_dir=args.output_dir,
          run_name=args.run_name,
          resume=args.resume,
        )
    


if __name__ == "__main__":
    main()
