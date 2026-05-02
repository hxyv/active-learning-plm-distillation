"""Acquisition functions for pool-based active learning.

Return contract
---------------
Every acquisition function returns a tuple ``(selected_ids, diagnostics)`` where
``diagnostics`` is a JSON-serializable dict, *except* ``diversity_acquisition``
which is dispatched through ``run_al_loop_diversity`` and returns a plain list
(see ``active_learning/al_loop.py``).  ``selected_ids`` is a sorted list of
protein IDs chosen from the pool.

Current implementations
-----------------------
random_acquisition      -- passive baseline (uniform random sampling)
mc_dropout_acquisition  -- MC Dropout epistemic uncertainty sampling.  Selection
                           is driven by per-residue BALD (mutual information)
                           across T stochastic passes (Houlsby 2011, Gal &
                           Ghahramani 2016).  Mean-per-class softmax variance,
                           predictive entropy, and expected entropy are logged
                           alongside for downstream ablation.
emc_acquisition         -- Expected Model Change (analytic ||∂CE/∂W_last||,
                           no GNN backward).
diversity_acquisition   -- Agglomerative clustering on graph-level embeddings.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data

from data.pyg_dataset import DistillationGraphDataset

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Random (passive baseline)
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# MC Dropout (BALD)
# ----------------------------------------------------------------------

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
    variance_score: Dict[str, float] = {}     # mean-per-class softmax variance
    predictive_entropy: Dict[str, float] = {} # H(p̄)
    expected_entropy: Dict[str, float] = {}   # E[H(p_t)]  (aleatoric)
    bald_score: Dict[str, float] = {}         # predictive − expected  (epistemic, selection score)

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

    diagnostics: Dict = {
        "strategy": "mc_dropout",
        "n_passes": n_passes,
        "pool_size": len(pool_dataset),
        "selected": n_select,
        "pool_stats": {
            "variance": _pool_distribution_stats(variance_score),
            "predictive_entropy": _pool_distribution_stats(predictive_entropy),
            "expected_entropy": _pool_distribution_stats(expected_entropy),
            "bald": _pool_distribution_stats(bald_score),
        },
        "selected_stats": {
            "variance": _pool_distribution_stats({k: variance_score[k] for k in selected}),
            "predictive_entropy": _pool_distribution_stats({k: predictive_entropy[k] for k in selected}),
            "expected_entropy": _pool_distribution_stats({k: expected_entropy[k] for k in selected}),
            "bald": _pool_distribution_stats({k: bald_score[k] for k in selected}),
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


# ----------------------------------------------------------------------
# Expected Model Change (Wen Li)
# ----------------------------------------------------------------------

def _emc_node_score_last_linear(h: "torch.Tensor", probs: "torch.Tensor") -> float:
    """EMC score for one node using only the last nn.Linear.

    Computes ``sum_j p_j * ||∂ CE(z, class=j) / ∂W||_F`` with cross-entropy loss,
    where ``z = h W^T + b`` and ``p = softmax(z)``.  Uses
    ``||∂L/∂W||_F = ||∂L/∂z||_2 * ||h||_2`` (Frobenius norm).

    Inputs:
    h:     Last-layer input for that node.
    probs: Class probabilities (softmax of logits).

    Returns:
        Expected gradient-norm score for that node.
    """
    import torch

    p = probs.float()
    hv = h.float()
    # gradient of the loss
    h_norm = torch.linalg.vector_norm(hv)
    # sum of the squared probabilities
    sq_p = (p * p).sum()
    # Frobenius norm of the difference
    diff_norms = torch.sqrt(sq_p + 1.0 - 2.0 * p)
    return float((h_norm * p * diff_norms).sum().item())


def emc_acquisition(
    pool_ids: List[str],
    budget: int,
    rng: np.random.Generator,
    cfg: dict,
    checkpoint_path: Path,
    device,
    **kwargs,
) -> Tuple[List[str], Dict]:
    """Select proteins with the highest expected model change.

    For each candidate graph, computes per-node EMC scores on a random subset
    of nodes and averages.  ``cfg['active_learning']['emc_max_pool_graphs']`` is
    the size of the random pool subset scored each round (faster than full-pool
    EMC).  ``cfg['active_learning']['emc_max_nodes_per_graph']`` is the random
    subset of nodes used per graph.

    Args:
        pool_ids:        IDs of all currently unlabeled pool proteins.
        budget:          Number of proteins to select.
        rng:             Seeded numpy Generator.
        cfg:             Round config dict.
        checkpoint_path: Path to the best model checkpoint.
        device:          Torch device for inference.
        **kwargs:        Ignored.

    Returns:
        (sorted list of selected protein IDs, diagnostics dict)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader

    from data.pyg_dataset import DistillationGraphDataset
    from models.factory import build_model

    # find checkpoint file
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"EMC acquisition: checkpoint not found: {checkpoint_path}")

    # get the splits file
    splits_file_cfg = cfg["data"].get("splits_file")
    if not splits_file_cfg:
        raise ValueError("emc_acquisition requires cfg['data']['splits_file'] to be set.")

    # load the pool dataset
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

    al_cfg = cfg.get("active_learning", {})
    emc_nodes_cap_raw = al_cfg.get("emc_max_nodes_per_graph")
    emc_cap_raw = al_cfg.get("emc_max_pool_graphs")
    n_pool = len(pool_dataset)
    parent_indices: List[int]

    # get the parent indices
    cap = int(emc_cap_raw)
    parent_indices = sorted(rng.choice(n_pool, size=cap, replace=False).tolist())
    logger.info(
        "EMC: scoring %d of %d pool graphs (active_learning.emc_max_pool_graphs=%d)",
        len(parent_indices),
        n_pool,
        cap,
    )
    loader_dataset: Union[DistillationGraphDataset, Subset] = Subset(pool_dataset, parent_indices)

    batch_size = int(cfg.get("eval", {}).get("batch_size", 48))
    pool_loader = DataLoader(loader_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # load the model
    model = build_model(cfg)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    # get the final layer
    final_layer = model.model.out_network[-1]

    # initialize the EMC dictionary
    emc_by_id: dict = {}
    graph_idx = 0

    # iterate over the pool loader
    for batch in pool_loader:
        # move the batch to the device
        batch = batch.to(device)
        h_last: Optional[torch.Tensor] = None

        # capture the last layer input, a specific method in torch
        def _capture_last_in(_mod, inp, _out):
            nonlocal h_last
            h_last = inp[0].detach()
        # register the forward hook to capture the last layer input
        hook = final_layer.register_forward_hook(_capture_last_in)
        try:
            with torch.no_grad():
                logits = model(batch)
        finally:
            hook.remove()

        # get the probabilities
        probs = torch.softmax(logits.float(), dim=-1)
        # iterate over the graphs in the batch
        for gi in range(batch.num_graphs):
            # find the graph
            mask = batch.batch == gi
            # get the nodes in the current graph
            node_h = h_last[mask]
            node_probs = probs[mask]
            n_nodes = int(node_h.size(0))
            if n_nodes == 0:
                continue

            pid = pool_dataset.sample_ids[parent_indices[graph_idx + gi]]
            graph_emc = 0.0

            # select some nodes randomly
            k_nodes = int(emc_nodes_cap_raw)
            node_iter = rng.choice(n_nodes, size=k_nodes, replace=False).tolist()

            # compute the EMC score for the selected nodes
            for ni in node_iter:
                graph_emc += _emc_node_score_last_linear(node_h[ni], node_probs[ni])

            # normalize the EMC score by the number of nodes
            denom = max(1, len(node_iter))
            emc_by_id[pid] = graph_emc / denom

        graph_idx += batch.num_graphs
    # sort the proteins by the EMC score and select the top k
    ranked = sorted(emc_by_id.keys(), key=lambda sid: -emc_by_id[sid])
    k = min(budget, len(ranked))
    top = ranked[:k]
    logger.info(
        "EMC acquisition: selected %d proteins (top EMC=%.6f, kth=%.6f)",
        len(top),
        emc_by_id[ranked[0]] if ranked else float("nan"),
        emc_by_id[ranked[k - 1]] if k else float("nan"),
    )
    diagnostics: Dict = {
        "strategy": "emc",
        "scored_pool_size": len(parent_indices),
        "pool_size": n_pool,
        "selected": len(top),
        "pool_stats": {"emc": _pool_distribution_stats(emc_by_id)},
    }
    return sorted(top), diagnostics


# ----------------------------------------------------------------------
# Diversity sampling (Wen Li)
# ----------------------------------------------------------------------

def get_diversity_cluster_labels(X: np.ndarray, budget: int) -> np.ndarray:
    """Cluster rows of X with n_clusters=budget; returns one label per row."""
    from sklearn.cluster import AgglomerativeClustering

    clustering = AgglomerativeClustering(n_clusters=budget)
    clustering.fit(X)
    return clustering.labels_


def diversity_acquisition(
    pool_ids: List[str],
    budget: int,
    rng: np.random.Generator,
    cfg: dict,
    checkpoint_path: Path,
    device,
    *,
    graph_embedding_by_id: Dict[str, np.ndarray],
    **kwargs,
) -> List[str]:
    """One random protein per agglomerative cluster, with graph-level embeddings.

    Args:
        pool_ids: list of pool protein IDs.
        budget:   number of proteins to select.
        rng:      random number generator.
        graph_embedding_by_id: {protein_id: graph-level embedding}.
        Other args ignored (kept for compatibility).

    Returns:
        Sorted list of selected protein IDs.
    """
    # stack the graph embeddings for the pool proteins
    X = np.stack([graph_embedding_by_id[sid] for sid in pool_ids], axis=0)
    labels = get_diversity_cluster_labels(X, budget)

    # cluster the proteins
    clusters: Dict[int, List[str]] = defaultdict(list)
    for sid, lab in zip(pool_ids, labels):
        clusters[int(lab)].append(sid)

    # select one random protein per cluster
    selected = [rng.choice(clusters[k]) for k in sorted(clusters)]
    return sorted(selected)


# Feature-propagation helpers used by diversity_acquisition's caller.

def propagate_graph_embeddings_for_one_graph(dataset: DistillationGraphDataset, index: int):
    """Propagate the graph embeddings to the node embeddings."""
    data = dataset[index]
    # raw adjacency matrix
    A = data_to_adjacency_sparse_featprop(data)

    # identity matrix
    I = sp.eye(A.shape[0], dtype=A.dtype)

    # diagonal degree matrix of A
    D = sp.diags(np.asarray(A.sum(axis=1)).ravel(), offsets=0)

    # normalized adjacency matrix S = (I + D)^(-1/2) (A + I) (I + D)^(-1/2)
    d_inv_sqrt = np.power(np.asarray((D + I).diagonal()).flatten(), -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    # normalized adjacency
    S = D_inv_sqrt @ (A + I) @ D_inv_sqrt

    # get the original node embeddings, then propagate
    original_node_embeddings = data_to_original_node_features(data)
    node_embedding = S @ original_node_embeddings

    return node_embedding


def propagate_graph_embeddings_all(dataset: DistillationGraphDataset) -> list[np.ndarray]:
    """Get propagated node embeddings for all graphs in the dataset."""
    return [propagate_graph_embeddings_for_one_graph(dataset, i) for i in range(len(dataset))]


def data_to_adjacency_sparse_featprop(data: Data) -> sp.csr_matrix:
    """Get adjacency matrix for the graph, used for feature propagation."""
    n = int(data.pos.shape[0])
    edge_index = data.edge_index
    if edge_index.numel() == 0:  # no edges, return empty matrix
        return sp.csr_matrix((n, n), dtype=np.float32)

    ei = edge_index.cpu().numpy()
    src, dst = ei[0], ei[1]
    m = src.shape[0]
    A = sp.coo_matrix(
        (np.ones(m, dtype=np.float32), (src, dst)),
        shape=(n, n),
        dtype=np.float32,
    )
    # symmetrize and binarize
    A = (A + A.T).tocsr()
    A.data = np.ones_like(A.data)
    return A


def data_to_original_node_features(
    data: Data,
    aa_vocab_size: int = 21,
    atom_vocab_size: int = 3,
) -> np.ndarray:
    """Per-node features [N, aa_vocab_size + atom_vocab_size + 3]: AA one-hot, atom one-hot, normalized pos."""
    aa_idx = data.aa_idx.cpu().numpy()
    atom_idx = data.atom_idx.cpu().numpy()
    pos = data.pos.detach().cpu().numpy().astype(np.float32)

    aa = np.eye(aa_vocab_size, dtype=np.float32)[aa_idx]
    atom = np.eye(atom_vocab_size, dtype=np.float32)[atom_idx]
    pos = (pos - pos.mean(axis=0, keepdims=True)) / (pos.std(axis=0, keepdims=True) + 1e-8)
    return np.concatenate([aa, atom, pos], axis=1)


def get_original_node_features(
    dataset: DistillationGraphDataset,
    aa_vocab_size: int = 21,
    atom_vocab_size: int = 3,
) -> Union[np.ndarray, List[np.ndarray]]:
    """One feature matrix per graph (same layout as ``data_to_original_node_features``)."""
    return [
        data_to_original_node_features(dataset[i], aa_vocab_size, atom_vocab_size)
        for i in range(len(dataset))
    ]
