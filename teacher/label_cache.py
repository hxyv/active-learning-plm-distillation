"""Teacher label expansion and caching utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def expand_residue_probs_to_nodes(residue_probs: np.ndarray, atoms_per_residue: int = 3) -> np.ndarray:
    if residue_probs.ndim != 2 or residue_probs.shape[1] != 8:
        raise ValueError(f"Expected residue_probs shape [L, 8], got {residue_probs.shape}")
    return np.repeat(residue_probs, repeats=atoms_per_residue, axis=0)


def save_teacher_probs(
    out_path: Path,
    sample_id: str,
    residue_probs: np.ndarray,
    node_probs: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        sample_id=sample_id,
        teacher_probs_residue=residue_probs.astype(np.float32),
        teacher_probs_node=node_probs.astype(np.float32),
    )
