"""Build PyG graph objects from processed backbone-only DISPEF NPZ files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from data.constants import ATOM_TYPE_TO_INDEX, ATOM_TYPES


def _expand_residue_values_to_nodes(arr: np.ndarray) -> np.ndarray:
    """Duplicate per-residue values to per-node values for N/CA/C."""
    return np.repeat(arr, repeats=len(ATOM_TYPES), axis=0)


def _make_node_arrays(sample: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords_res = sample["coords"].astype(np.float32)  # [L, 3, 3]
    aa_res = sample["aa_idx"].astype(np.int64)  # [L]

    num_res = coords_res.shape[0]

    # Node order follows residue-major then atom order [N, CA, C].
    coords_node = coords_res.reshape(num_res * len(ATOM_TYPES), 3)
    aa_node = _expand_residue_values_to_nodes(aa_res)

    atom_pattern = np.asarray([ATOM_TYPE_TO_INDEX[a] for a in ATOM_TYPES], dtype=np.int64)
    atom_node = np.tile(atom_pattern, reps=num_res)

    node_to_residue = np.repeat(np.arange(num_res, dtype=np.int64), repeats=len(ATOM_TYPES))
    return coords_node, aa_node, atom_node, node_to_residue


def _build_edges(
    coords: np.ndarray,
    cutoff: float,
    max_neighbors: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Radius graph with optional per-node neighbor cap."""
    n = coords.shape[0]
    if n == 0 or max_neighbors == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 4), dtype=torch.float32)

    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    src_list = []
    dst_list = []
    edge_attr = []

    for i in range(n):
        nbrs = np.where((dist[i] <= cutoff) & (dist[i] > 0.0))[0]
        if max_neighbors > 0 and len(nbrs) > max_neighbors:
            order = np.argsort(dist[i, nbrs])[:max_neighbors]
            nbrs = nbrs[order]

        for j in nbrs:
            src_list.append(i)
            dst_list.append(int(j))
            rel = coords[j] - coords[i]
            edge_attr.append([dist[i, j], rel[0], rel[1], rel[2]])

    if not src_list:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 4), dtype=torch.float32)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr_t = torch.tensor(np.asarray(edge_attr, dtype=np.float32), dtype=torch.float32)
    return edge_index, edge_attr_t


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def build_graph_data(
    sample_path: Path,
    teacher_path: Optional[Path],
    cutoff: float,
    max_neighbors: int,
    center_coords: bool = True,
) -> Data:
    sample = load_npz(sample_path)
    coords_node, aa_node, atom_node, node_to_residue = _make_node_arrays(sample)

    if center_coords:
        coords_node = coords_node - coords_node.mean(axis=0, keepdims=True)

    edge_index, edge_attr = _build_edges(coords=coords_node, cutoff=cutoff, max_neighbors=max_neighbors)

    teacher_probs_node = None
    if teacher_path is not None and teacher_path.exists():
        t = load_npz(teacher_path)
        if "teacher_probs_node" in t:
            teacher_probs_node = t["teacher_probs_node"].astype(np.float32)
        elif "teacher_probs_residue" in t:
            teacher_probs_node = _expand_residue_values_to_nodes(
                t["teacher_probs_residue"].astype(np.float32)
            )

    dssp_res = sample.get("dssp_idx", None)
    if dssp_res is not None:
        dssp_node = _expand_residue_values_to_nodes(dssp_res.astype(np.int64))
    else:
        dssp_node = np.full((coords_node.shape[0],), fill_value=-100, dtype=np.int64)

    seq_chars = sample.get("sequence", None)
    seq_str = "" if seq_chars is None else "".join([str(c) for c in seq_chars.tolist()])

    data = Data(
        aa_idx=torch.tensor(aa_node, dtype=torch.long),
        atom_idx=torch.tensor(atom_node, dtype=torch.long),
        pos=torch.tensor(coords_node, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_to_residue=torch.tensor(node_to_residue, dtype=torch.long),
        dssp_idx=torch.tensor(dssp_node, dtype=torch.long),
    )

    if teacher_probs_node is not None:
        data.teacher_probs = torch.tensor(teacher_probs_node, dtype=torch.float32)

    sample_id = str(sample.get("sample_id", sample_path.stem))
    data.sample_id = sample_id
    data.sequence = seq_str
    data.num_residues = int(sample["aa_idx"].shape[0])
    return data
