"""Acquisition functions for emc and diversity sampling in hierarchical learning.

Current implementations
-----------------------
emc_acquisition         -- Expected Model Change (analytic ||∂CE/∂W_last||, no GNN backward)
diversity_acquisition   -- Agglomerative clustering on graph-level embeddings 
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


def _emc_node_score_last_linear(h: "torch.Tensor", probs: "torch.Tensor") -> float:
    """EMC score for one node using only the last nn.Linear.

    Computes sum_j p_j * ||∂ CE(z, class=j) / ∂W||_F with cross-entropy loss, where z = h W^T + b 
    and p = softmax(z).  Uses ||∂L/∂W||_F = ||∂L/∂z||_2 * ||h||_2 (Frobenius norm).

    Inputs:
    h:    Last-layer input for that node.
    probs: Class probabilities (softmax of logits)

    Returns:
        expected gradient-norm score for that node.
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


def emc_acquisition(pool_ids: List[str], budget: int, rng: np.random.Generator, cfg: dict, checkpoint_path: Path, device, **kwargs) -> List[str]:
    """Select proteins with the highest expected model change.
        for comparing graphs, calculate the EMC score for selected nodes in the graph and average over the nodes
    cfg['active_learning']['emc_max_pool_graphs'] is the size of the pool that is scored each round (much faster, not equivalent to
    full-pool EMC).

    cfg['active_learning']['emc_max_nodes_per_graph'] is the random subset of nodes inside that graph for the average EMC score.
    Args:
        pool_ids:        IDs of all currently unlabeled pool proteins.
        budget:          Number of proteins to select.
        rng:             Seeded numpy Generator.
        cfg:             Round config dict.
        checkpoint_path: Path to the best model checkpoint.
        device:          Torch device for inference.
        **kwargs:        Ignored.

    Returns:
        Sorted list of selected protein IDs.
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
    return sorted(top)

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
    """One random protein per agglomerative cluster with graph-level embeddings.
       Args:
        pool_ids: list of pool protein IDs
        budget: number of proteins to select
        rng: random number generator
        ohter is ignored, just for compatibility
        graph_embedding_by_id: dictionary of graph embeddings by protein ID
    Returns:
        list of selected protein IDs
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

# actually node feature propagation
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

    # get the originalnode embeddings
    original_node_embeddings = data_to_original_node_features(data)
    # propagate
    node_embedding = S @ original_node_embeddings

    return node_embedding

def propagate_graph_embeddings_all(dataset: DistillationGraphDataset) -> list[np.ndarray]:
    """get propagated node embeddings for all graphs in the dataset"""
    return [propagate_graph_embeddings_for_one_graph(dataset, i) for i in range(len(dataset))]

def data_to_adjacency_sparse_featprop(data: Data) -> sp.csr_matrix:
    """get adjacency matrix for the graph, used for feature propagation
    """
    n = int(data.pos.shape[0])
    edge_index = data.edge_index
    if edge_index.numel() == 0: # no edges, return empty matrix
        return sp.csr_matrix((n, n), dtype=np.float32)

    ei = edge_index.cpu().numpy()
    # get the source and destination indices
    src, dst = ei[0], ei[1]
    m = src.shape[0]
    # create a sparse matrix
    A = sp.coo_matrix(
        (np.ones(m, dtype=np.float32), (src, dst)),
        shape=(n, n),
        dtype=np.float32,
    )
    # symmetrize the matrix
    A = (A + A.T).tocsr()
    # set the data to 1
    A.data = np.ones_like(A.data)
    return A

def data_to_original_node_features(
    data: Data,
    aa_vocab_size: int = 21,
    atom_vocab_size: int = 3,
) -> np.ndarray:
    """get original node features [N, aa_vocab_size + atom_vocab_size + 3]: AA one-hot, atom one-hot, normalized pos.
    """
    aa_idx = data.aa_idx.cpu().numpy()
    atom_idx = data.atom_idx.cpu().numpy()
    pos = data.pos.detach().cpu().numpy().astype(np.float32)

    aa = np.eye(aa_vocab_size, dtype=np.float32)[aa_idx]
    atom = np.eye(atom_vocab_size, dtype=np.float32)[atom_idx]
    # normalize the position
    pos = (pos - pos.mean(axis=0, keepdims=True)) / (pos.std(axis=0, keepdims=True) + 1e-8)
    return np.concatenate([aa, atom, pos], axis=1)


def get_original_node_features(
    dataset: DistillationGraphDataset,
    aa_vocab_size: int = 21,
    atom_vocab_size: int = 3,
) -> Union[np.ndarray, List[np.ndarray]]:
    """get original node features for the dataset"""
    return [
        data_to_original_node_features(dataset[i], aa_vocab_size, atom_vocab_size)
        for i in range(len(dataset))
    ]