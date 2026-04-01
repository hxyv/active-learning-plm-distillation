"""Acquisition functions for pool-based active learning.

Current implementations
-----------------------
random_acquisition      -- passive baseline (uniform random sampling)
mc_dropout_acquisition  -- MC Dropout uncertainty sampling on output MLP head

Planned (not yet implemented)
------------------------------
emc_acquisition         -- Expected Model Change via output-layer gradient norms
diversity_acquisition   -- Agglomerative hierarchical clustering on node embeddings
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def random_acquisition(pool_ids: List[str], budget: int, rng: np.random.Generator, **kwargs) -> List[str]:
    """Select ``budget`` proteins uniformly at random from the pool (passive baseline).

    Args:
        pool_ids: IDs of all currently unlabeled proteins.
        budget:   Number of proteins to select.
        rng:      Seeded numpy Generator for reproducibility.
        **kwargs: Ignored (present for a uniform call site with model-based strategies).

    Returns:
        Sorted list of selected protein IDs (length = min(budget, len(pool_ids))).
    """
    n = min(budget, len(pool_ids))
    indices = rng.choice(len(pool_ids), size=n, replace=False)
    return sorted([pool_ids[i] for i in indices])


def _enable_mc_dropout(model) -> None:
    """Put model in eval mode but keep all Dropout layers active (MC Dropout inference)."""
    import torch.nn as nn
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_acquisition(
    pool_ids: List[str],
    budget: int,
    rng: np.random.Generator,
    *,
    cfg: dict,
    checkpoint_path: Path,
    device,
    n_passes: int = 20,
    **kwargs,
) -> List[str]:
    """Select proteins with the highest mean predictive entropy under MC Dropout.

    For each pool protein, runs ``n_passes`` stochastic forward passes with
    dropout active and computes the mean per-node predictive entropy across
    passes.  The top-``budget`` proteins by descending uncertainty are returned.

    Args:
        pool_ids:        IDs of all currently unlabeled pool proteins.
        budget:          Number of proteins to select.
        rng:             Seeded numpy Generator (unused; kept for uniform call site).
        cfg:             Round config dict (must have ``data.splits_file`` set).
        checkpoint_path: Path to the best model checkpoint (.pt) from this round.
        device:          Torch device for inference.
        n_passes:        Number of MC Dropout forward passes (default 20).
        **kwargs:        Ignored.

    Returns:
        Sorted list of selected protein IDs (length = min(budget, len(pool_ids))).
    """
    import torch
    from torch_geometric.loader import DataLoader

    from data.pyg_dataset import DistillationGraphDataset
    from models.factory import build_model

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"MC Dropout acquisition: checkpoint not found: {checkpoint_path}")

    splits_file_cfg = cfg["data"].get("splits_file")
    if not splits_file_cfg:
        raise ValueError("mc_dropout_acquisition requires cfg['data']['splits_file'] to be set.")

    # Build pool dataset.  teacher_root=None: no teacher labels needed for inference.
    pool_dataset = DistillationGraphDataset(
        processed_root=Path(cfg["paths"]["processed_root"]),
        dataset_name=cfg["data"]["dataset_name"],
        split_name="pool_unassigned",
        teacher_root=None,
        cutoff=float(cfg["graph"].get("cutoff", 8.0)),
        max_neighbors=int(cfg["graph"].get("max_neighbors", 64)),
        cache_graphs=False,
        splits_file=Path(splits_file_cfg),
    )

    batch_size = int(cfg.get("eval", {}).get("batch_size", 48))
    pool_loader = DataLoader(pool_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Load model from checkpoint.
    model = build_model(cfg)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    _enable_mc_dropout(model)
    logger.info(
        "MC Dropout acquisition: %d pool proteins, %d passes, batch_size=%d",
        len(pool_dataset), n_passes, batch_size,
    )

    # Accumulate per-protein mean-node entropy summed across passes.
    entropy_sum: dict = defaultdict(float)

    for pass_idx in range(n_passes):
        graph_idx = 0
        for batch in pool_loader:
            batch = batch.to(device)
            with torch.no_grad():
                logits = model(batch)
            probs = torch.softmax(logits.float(), dim=-1)  # [N_nodes, 8]
            node_entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # [N_nodes]

            for i in range(batch.num_graphs):
                node_mask = batch.batch == i
                prot_id = pool_dataset.sample_ids[graph_idx + i]
                entropy_sum[prot_id] += node_entropy[node_mask].mean().item()

            graph_idx += batch.num_graphs

        if (pass_idx + 1) % 5 == 0:
            logger.debug("MC Dropout pass %d/%d done", pass_idx + 1, n_passes)

    # Average entropy over passes and rank.
    uncertainty = {pid: entropy_sum[pid] / n_passes for pid in pool_dataset.sample_ids}
    ranked = sorted(pool_dataset.sample_ids, key=lambda x: -uncertainty[x])
    selected = sorted(ranked[:min(budget, len(ranked))])

    logger.info(
        "MC Dropout acquisition: selected %d proteins (top uncertainty=%.4f, bottom=%.4f)",
        len(selected),
        uncertainty[ranked[0]] if ranked else float("nan"),
        uncertainty[ranked[min(budget, len(ranked)) - 1]] if ranked else float("nan"),
    )
    return selected
