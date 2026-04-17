"""Acquisition functions for pool-based active learning.

Return contract
---------------
Every acquisition function returns a tuple ``(selected_ids, diagnostics)`` where
``diagnostics`` is a JSON-serializable dict.  ``selected_ids`` is a sorted list
of protein IDs chosen from the pool.

Current implementations
-----------------------
random_acquisition      -- passive baseline (uniform random sampling)
mc_dropout_acquisition  -- MC Dropout epistemic uncertainty sampling.  Selection
                           is driven by mean-per-class softmax variance across T
                           stochastic passes (Gal & Ghahramani 2016 form).  BALD,
                           predictive entropy, and expected entropy are logged
                           alongside for downstream ablation.

Planned (not yet implemented)
------------------------------
emc_acquisition         -- Expected Model Change via output-layer gradient norms
diversity_acquisition   -- Agglomerative hierarchical clustering on node embeddings
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _pool_distribution_stats(values: Dict[str, float]) -> Dict[str, float]:
    """Return mean / std / p10 / p50 / p90 of a {id: score} map."""
    if not values:
        return {"mean": float("nan"), "std": float("nan"),
                "p10": float("nan"), "p50": float("nan"), "p90": float("nan"),
                "min": float("nan"), "max": float("nan")}
    arr = np.array(list(values.values()), dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std()),
        "p10":  float(np.percentile(arr, 10)),
        "p50":  float(np.percentile(arr, 50)),
        "p90":  float(np.percentile(arr, 90)),
        "min":  float(arr.min()),
        "max":  float(arr.max()),
    }


def random_acquisition(
    pool_ids: List[str],
    budget: int,
    rng: np.random.Generator,
    **kwargs,
) -> Tuple[List[str], Dict]:
    """Select ``budget`` proteins uniformly at random from the pool (passive baseline).

    Returns:
        (sorted list of selected IDs, diagnostics dict)
    """
    n = min(budget, len(pool_ids))
    indices = rng.choice(len(pool_ids), size=n, replace=False)
    selected = sorted([pool_ids[i] for i in indices])
    return selected, {"strategy": "random", "pool_size": len(pool_ids), "selected": n}


def _enable_mc_dropout(model) -> None:
    """Put model in eval mode but keep every Dropout layer active (MC Dropout inference)."""
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
) -> Tuple[List[str], Dict]:
    """Select proteins with the highest BALD (mutual information) under MC Dropout.

    For each pool protein, runs ``n_passes`` stochastic forward passes with
    dropout active.  Let :math:`p_{i,t}` be the softmax probability vector at
    node ``i`` under pass ``t``, and :math:`\\bar p_i = \\tfrac{1}{T}\\sum_t p_{i,t}`.
    The per-node BALD score is the mutual information between the predictive
    distribution and the weights:

    .. math::
        \\mathrm{BALD}_i = H\\!\\left[\\bar p_i\\right]
                           - \\tfrac{1}{T}\\sum_t H\\!\\left[p_{i,t}\\right]

    The per-protein score averages BALD across nodes (Houlsby 2011, Gal-Islam-
    Ghahramani 2017).  Top-``budget`` proteins by descending BALD are selected.

    Mean-per-class softmax variance (Gal & Ghahramani 2016 form), predictive
    entropy, and expected entropy are also computed and returned in
    ``diagnostics`` for downstream comparison and ablation.

    Returns:
        (sorted list of selected IDs, diagnostics dict)
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

    model = build_model(cfg)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    _enable_mc_dropout(model)
    logger.info(
        "MC Dropout acquisition: %d pool proteins, %d passes, batch_size=%d",
        len(pool_dataset), n_passes, batch_size,
    )

    # Per-protein running buffers: we accumulate sums across passes to compute
    # mean/variance/entropy at the end without holding a full [T, L, C] tensor
    # per protein.  probs is [L_p, C]; for T<=20 this is cheap.
    prob_sum: Dict[str, np.ndarray] = {}     # Σ_m p_m      → p̄ = sum/T
    prob_sq_sum: Dict[str, np.ndarray] = {}  # Σ_m p_m²     → Var = sq/T − (sum/T)²
    ent_sum: Dict[str, float] = {}           # Σ_m H(p_m).mean_over_nodes  → E[H]

    for pass_idx in range(n_passes):
        graph_idx = 0
        for batch in pool_loader:
            batch = batch.to(device)
            with torch.no_grad():
                logits = model(batch)
            probs = torch.softmax(logits.float(), dim=-1)               # [N_nodes, C]
            node_entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # [N_nodes]

            for i in range(batch.num_graphs):
                node_mask = batch.batch == i
                prot_id = pool_dataset.sample_ids[graph_idx + i]
                p_i = probs[node_mask].detach().cpu().numpy().astype(np.float64)  # [L_p, C]
                e_i = float(node_entropy[node_mask].mean().item())

                if prot_id not in prob_sum:
                    prob_sum[prot_id] = np.zeros_like(p_i)
                    prob_sq_sum[prot_id] = np.zeros_like(p_i)
                    ent_sum[prot_id] = 0.0
                prob_sum[prot_id] += p_i
                prob_sq_sum[prot_id] += p_i * p_i
                ent_sum[prot_id] += e_i

            graph_idx += batch.num_graphs

        if (pass_idx + 1) % 5 == 0:
            logger.debug("MC Dropout pass %d/%d done", pass_idx + 1, n_passes)

    # Aggregate per-protein scores.
    variance_score: Dict[str, float] = {}     # selection score (report formula)
    predictive_entropy: Dict[str, float] = {} # H(p̄)
    expected_entropy: Dict[str, float] = {}   # E[H(p_t)]  (aleatoric)
    bald_score: Dict[str, float] = {}         # predictive − expected  (epistemic)

    for pid in pool_dataset.sample_ids:
        T = float(n_passes)
        p_bar = prob_sum[pid] / T                               # [L_p, C]
        # Numerically-stable variance across passes per (node, class).
        var_per_cell = np.maximum(prob_sq_sum[pid] / T - p_bar * p_bar, 0.0)
        variance_score[pid] = float(var_per_cell.mean())

        h_pbar = -(p_bar * np.log(p_bar + 1e-10)).sum(axis=-1)  # [L_p]
        predictive_entropy[pid] = float(h_pbar.mean())
        expected_entropy[pid] = float(ent_sum[pid] / T)
        bald_score[pid] = predictive_entropy[pid] - expected_entropy[pid]

    # Rank by BALD (mutual information); the selection score.
    ranked = sorted(pool_dataset.sample_ids, key=lambda x: -bald_score[x])
    n_select = min(budget, len(ranked))
    selected = sorted(ranked[:n_select])

    diagnostics = {
        "strategy": "mc_dropout",
        "selection_score": "bald",
        "n_passes": n_passes,
        "pool_size": len(pool_dataset),
        "selected": n_select,
        "pool_stats": {
            "variance":            _pool_distribution_stats(variance_score),
            "predictive_entropy":  _pool_distribution_stats(predictive_entropy),
            "expected_entropy":    _pool_distribution_stats(expected_entropy),
            "bald":                _pool_distribution_stats(bald_score),
        },
        "selected_stats": {
            "variance":            _pool_distribution_stats({p: variance_score[p] for p in selected}),
            "predictive_entropy":  _pool_distribution_stats({p: predictive_entropy[p] for p in selected}),
            "expected_entropy":    _pool_distribution_stats({p: expected_entropy[p] for p in selected}),
            "bald":                _pool_distribution_stats({p: bald_score[p] for p in selected}),
        },
    }

    logger.info(
        "MC Dropout acquisition: selected %d proteins | BALD top=%.4e bottom=%.4e | "
        "pool BALD mean=%.4e std=%.4e | pool variance mean=%.4e",
        n_select,
        bald_score[ranked[0]] if ranked else float("nan"),
        bald_score[ranked[n_select - 1]] if ranked else float("nan"),
        diagnostics["pool_stats"]["bald"]["mean"],
        diagnostics["pool_stats"]["bald"]["std"],
        diagnostics["pool_stats"]["variance"]["mean"],
    )
    return selected, diagnostics
