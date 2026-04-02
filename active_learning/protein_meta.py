"""Per-protein metadata utilities for active learning analysis.

Computes SS8 composition (fraction of residues in each of the 8 DSSP classes)
from processed NPZ files.  Results are cached to ``protein_meta.json`` in the
dataset folder so the computation runs only once.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from data.constants import SS8_CLASSES

logger = logging.getLogger(__name__)

# SS8 index → label, matching constants.SS8_CLASSES order: G H I T E B S C
_N_SS8 = len(SS8_CLASSES)  # 8


def compute_ss8_fractions(dssp_idx: np.ndarray) -> List[float]:
    """Return fraction of residues in each SS8 class for one protein.

    Args:
        dssp_idx: Integer array of shape [L] with values in 0..7 (-100 = missing).

    Returns:
        List of 8 floats (fractions summing to ≤1; missing residues excluded).
    """
    valid = dssp_idx[dssp_idx >= 0]
    if len(valid) == 0:
        return [0.0] * _N_SS8
    counts = np.bincount(valid, minlength=_N_SS8).astype(float)
    return (counts / len(valid)).tolist()


def load_protein_meta(
    processed_root: Path,
    dataset_name: str,
    protein_ids: List[str],
) -> Dict[str, Dict]:
    """Load (or compute and cache) per-protein metadata.

    Metadata includes:
        - ``ss8_fractions``: list of 8 floats, one per SS8 class
        - ``num_residues``: int

    Uses a JSON cache at ``<dataset_root>/protein_meta.json`` so NPZ files are
    only read once across all AL rounds.

    Args:
        processed_root: Root of processed data (parent of ``dataset_name/``).
        dataset_name:   Dataset subdirectory name (e.g. ``dispef_m``).
        protein_ids:    IDs to look up.

    Returns:
        Dict mapping protein_id → metadata dict.
    """
    dataset_root = Path(processed_root) / dataset_name
    proteins_dir = dataset_root / "proteins"
    cache_path = dataset_root / "protein_meta.json"

    # Load existing cache
    cache: Dict[str, Dict] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}

    missing = [pid for pid in protein_ids if pid not in cache]
    if missing:
        logger.debug("Computing protein_meta for %d proteins", len(missing))
        for pid in missing:
            npz_path = proteins_dir / f"{pid}.npz"
            if not npz_path.exists():
                logger.warning("protein_meta: NPZ not found for %s, skipping", pid)
                continue
            try:
                data = np.load(npz_path, allow_pickle=True)
                dssp_idx = data["dssp_idx"].astype(np.int64)
                # dssp_idx is node-level (3 atoms per residue); take every 3rd
                dssp_res = dssp_idx[::3]
                num_res = int(data["aa_idx"].shape[0])
                cache[pid] = {
                    "ss8_fractions": compute_ss8_fractions(dssp_res),
                    "num_residues": num_res,
                }
            except Exception as exc:
                logger.warning("protein_meta: failed to load %s: %s", pid, exc)

        # Persist updated cache
        try:
            cache_path.write_text(json.dumps(cache, indent=2))
        except Exception as exc:
            logger.warning("protein_meta: could not write cache: %s", exc)

    return {pid: cache[pid] for pid in protein_ids if pid in cache}


def ss8_composition_of_selected(
    selected_ids: List[str],
    processed_root: Path,
    dataset_name: str,
) -> Dict[str, float]:
    """Compute mean SS8 class fractions across a set of selected proteins.

    Returns a dict mapping SS8 label (e.g. 'H', 'E') to mean fraction.
    """
    meta = load_protein_meta(processed_root, dataset_name, selected_ids)
    if not meta:
        return {cls: 0.0 for cls in SS8_CLASSES}

    fracs = np.array([meta[pid]["ss8_fractions"] for pid in selected_ids if pid in meta])
    mean_fracs = fracs.mean(axis=0).tolist()
    return {cls: float(f) for cls, f in zip(SS8_CLASSES, mean_fracs)}
