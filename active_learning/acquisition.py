"""Acquisition functions for pool-based active learning.

Current implementations
-----------------------
random_acquisition      -- passive baseline (uniform random sampling)
mc_dropout_acquisition  -- MC Dropout uncertainty sampling on output MLP head

emc_acquisition         -- Expected Model Change via output-layer gradient norms
diversity_acquisition   -- Agglomerative clustering on graph-level embeddings (pool candidates)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data

from data.pyg_dataset import DistillationGraphDataset

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

def emc_acquisition(pool_ids: List[str], budget: int, rng: np.random.Generator, cfg: dict, checkpoint_path: Path, device, **kwargs) -> List[str]:
    """Select proteins with the highest expected model change.

    Per graph (protein), pools node-level SS8 logits, sums class-weighted
    ``||∂CE/∂W||`` on ``model.model.out_network[-1]`` (mean over nodes), then ranks graphs.

    If ``cfg['active_learning']['emc_max_pool_graphs']`` is a positive integer smaller than the
    pool, a **random subset** of that size is scored each round (much faster, not equivalent to
    full-pool EMC).

    If ``cfg['active_learning']['emc_max_nodes_per_graph']`` is a positive integer smaller than a
    graph's node count, EMC uses a **random subset of nodes** inside that graph (sample mean of
    the same per-node statistic; faster, approximate).

    Args:
        pool_ids:        IDs of all currently unlabeled pool proteins.
        budget:          Number of proteins to select.
        rng:             Seeded numpy Generator (used for optional EMC pool subsampling).
        cfg:             Round config dict (must have ``data.splits_file`` set).
        checkpoint_path: Path to the best model checkpoint (.pt) from this round.
        device:          Torch device for inference.
        **kwargs:        Ignored.

    Returns:
        Sorted list of selected protein IDs (length = min(budget, len(pool_ids))).
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader

    from data.pyg_dataset import DistillationGraphDataset
    from models.factory import build_model

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"EMC acquisition: checkpoint not found: {checkpoint_path}")

    splits_file_cfg = cfg["data"].get("splits_file")
    if not splits_file_cfg:
        raise ValueError("emc_acquisition requires cfg['data']['splits_file'] to be set.")

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
    if emc_cap_raw is not None and int(emc_cap_raw) > 0 and int(emc_cap_raw) < n_pool:
        cap = int(emc_cap_raw)
        parent_indices = sorted(rng.choice(n_pool, size=cap, replace=False).tolist())
        logger.info(
            "EMC: scoring %d of %d pool graphs (active_learning.emc_max_pool_graphs=%d)",
            len(parent_indices),
            n_pool,
            cap,
        )
        loader_dataset: Union[DistillationGraphDataset, Subset] = Subset(pool_dataset, parent_indices)
    else:
        parent_indices = list(range(n_pool))
        loader_dataset = pool_dataset

    if budget > len(parent_indices):
        logger.warning(
            "EMC: budget=%d exceeds scored pool size=%d; selecting %d this round",
            budget,
            len(parent_indices),
            len(parent_indices),
        )

    batch_size = int(cfg.get("eval", {}).get("batch_size", 48))
    pool_loader = DataLoader(loader_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(cfg)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    final_layer = model.model.out_network[-1]

    emc_by_id: dict = {}
    graph_idx = 0

    for batch in pool_loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.softmax(logits.float(), dim=-1)
        num_classes = int(logits.size(-1))

        for gi in range(batch.num_graphs):
            mask = batch.batch == gi
            node_logits = logits[mask]
            node_probs = probs[mask]
            n_nodes = int(node_logits.size(0))
            if n_nodes == 0:
                continue

            pid = pool_dataset.sample_ids[parent_indices[graph_idx + gi]]
            graph_emc = 0.0

            if (
                emc_nodes_cap_raw is not None
                and int(emc_nodes_cap_raw) > 0
                and int(emc_nodes_cap_raw) < n_nodes
            ):
                k_nodes = int(emc_nodes_cap_raw)
                node_iter = rng.choice(n_nodes, size=k_nodes, replace=False).tolist()
            else:
                node_iter = list(range(n_nodes))

            for ni in node_iter:
                logits_row = node_logits[ni : ni + 1]
                p_row = node_probs[ni]
                for j in range(num_classes):
                    model.zero_grad()
                    target = torch.tensor([j], device=device, dtype=torch.long)
                    loss = F.cross_entropy(logits_row, target)
                    # Same forward for all (node, class) steps; keep graph until next batch overwrites `logits`.
                    loss.backward(retain_graph=True)
                    grad = final_layer.weight.grad
                    gn = float(grad.norm().item()) if grad is not None else 0.0
                    graph_emc += p_row[j].item() * gn

            denom = max(1, len(node_iter))
            emc_by_id[pid] = graph_emc / denom

        graph_idx += batch.num_graphs

    ranked = sorted(emc_by_id.keys(), key=lambda sid: -emc_by_id[sid])
    k = min(budget, len(ranked))
    top = ranked[:k]
    logger.info(
        "EMC acquisition: selected %d proteins (top EMC=%.6f, kth=%.6f)",
        len(top),
        emc_by_id[ranked[0]] if ranked else float("nan"),
        emc_by_id[ranked[k - 1]] if k else float("nan"),
    )
    return sorted(top)

def get_diversity_cluster_labels(X: np.ndarray, budget: int) -> np.ndarray:
    """Cluster rows of ``X`` with ``n_clusters=budget``; returns one label per row."""
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
    """One random protein per agglomerative cluster (graph-level embeddings)."""

    X = np.stack([graph_embedding_by_id[sid] for sid in pool_ids], axis=0)
    labels = get_diversity_cluster_labels(X, budget)

    clusters: Dict[int, List[str]] = defaultdict(list)
    for sid, lab in zip(pool_ids, labels):
        clusters[int(lab)].append(sid)

    selected = [rng.choice(clusters[k]) for k in sorted(clusters)]
    return sorted(selected)


def propagate_graph_embeddings_for_one_graph(dataset: DistillationGraphDataset, index: int):
    """Propagate the graph embeddings to the node embeddings."""
    data = dataset[index]
    # raw adjacency matrix 
    A = data_to_adjacency_sparse_featprop(data)

    # identity matrix
    I = sp.eye(A.shape[0], dtype=A.dtype)

    # diagonal degree matrix of A (SciPy >=1.11: use offsets=0, not diags(vec, [0]))
    D = sp.diags(np.asarray(A.sum(axis=1)).ravel(), offsets=0)

    # normalized adjacency matrix S = (I + D)^(-1/2) (A + I) (I + D)^(-1/2)
    d_inv_sqrt = np.power(np.asarray((D + I).diagonal()).flatten(), -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    # normalized adjacency
    S = D_inv_sqrt @ (A + I) @ D_inv_sqrt

    # get the node embeddings
    graph_embedding = data_to_original_node_features(data)
    node_embedding = S @ graph_embedding

    return node_embedding

def propagate_graph_embeddings_all(dataset: DistillationGraphDataset) -> list[np.ndarray]:
    """Run the above for every graph (list in, list out)."""
    return [propagate_graph_embeddings_for_one_graph(dataset, i) for i in range(len(dataset))]

def data_to_adjacency_sparse_featprop(data: Data) -> sp.csr_matrix:
    """Binary undirected adjacency as sparse CSR (for e.g. feature propagation).

    Starts from directed ``edge_index``, symmetrizes with ``A + A.T``, then forces entries to ``1``
    so merged duplicate indices stay binary after ``tocsr``.
    """
    n = int(data.pos.shape[0])
    edge_index = data.edge_index
    if edge_index.numel() == 0:
        return sp.csr_matrix((n, n), dtype=np.float32)

    ei = edge_index.cpu().numpy()
    src, dst = ei[0], ei[1]
    m = src.shape[0]
    A = sp.coo_matrix(
        (np.ones(m, dtype=np.float32), (src, dst)),
        shape=(n, n),
        dtype=np.float32,
    )
    A = (A + A.T).tocsr()
    A.data = np.ones_like(A.data)
    return A

def data_to_original_node_features(
    data: Data,
    aa_vocab_size: int = 21,
    atom_vocab_size: int = 3,
) -> np.ndarray:
    """Hand-built per-node matrix ``[N, aa_vocab_size + atom_vocab_size + 3]``: AA one-hot, atom one-hot, normalized ``pos``.

    Matches DISPEF node semantics (``aa_idx`` / ``atom_idx`` / ``pos`` on each ``Data``); not the learned encoder output.
    """
    aa_idx = data.aa_idx.cpu().numpy()
    atom_idx = data.atom_idx.cpu().numpy()
    pos = data.pos.detach().cpu().numpy().astype(np.float32)

    aa = np.eye(aa_vocab_size, dtype=np.float32)[aa_idx]
    atom = np.eye(atom_vocab_size, dtype=np.float32)[atom_idx]

    pos = (pos - pos.mean(axis=0, keepdims=True)) / (pos.std(axis=0, keepdims=True) + 1e-8)
    return np.concatenate([aa, atom, pos], axis=1)


def get_original_node_features(
    dataset: DistillationGraphDataset,
    index: Optional[int] = None,
    aa_vocab_size: int = 21,
    atom_vocab_size: int = 3,
) -> Union[np.ndarray, List[np.ndarray]]:
    """One feature matrix per graph (same layout as ``data_to_original_node_features``)."""
    if index is not None:
        return data_to_original_node_features(dataset[index], aa_vocab_size, atom_vocab_size)
    return [
        data_to_original_node_features(dataset[i], aa_vocab_size, atom_vocab_size)
        for i in range(len(dataset))
    ]