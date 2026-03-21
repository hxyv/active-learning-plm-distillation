"""Generate mock teacher labels from DSSP hard labels for Colab / offline testing.

When ESM3 is not available (no GPU large enough / no Forge API key), this script
creates teacher_probs_node files in the same format expected by the trainer, using
the DSSP-derived one-hot class distribution as a stand-in for real ESM3 outputs.

This lets you run the full distillation pipeline on Colab to inspect the data
structures and training loop — the "teacher" is just DSSP itself, so the soft-CE
loss collapses to the standard CE loss.  Real distillation requires actual ESM3
labels generated via teacher/generate_teacher_labels.py.

Usage:
    python scripts/generate_mock_teacher.py \
        --processed-root /content/dispef_ws/data/processed \
        --dataset-name dispef_m \
        --teacher-cache-root /content/dispef_ws/cache/teacher \
        [--label-smoothing 0.05]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

SS8_CLASSES = 8
ATOMS_PER_RESIDUE = 3  # N, CA, C

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate mock teacher labels from DSSP")
    p.add_argument("--processed-root", type=Path, required=True)
    p.add_argument("--dataset-name", type=str, default="dispef_m")
    p.add_argument("--teacher-cache-root", type=Path, required=True)
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Label smoothing epsilon (0 = one-hot, >0 = soft labels)",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")
    return p.parse_args()


def make_mock_probs(dssp_node: np.ndarray, label_smoothing: float) -> np.ndarray:
    """Convert per-node DSSP indices to soft probability vectors [N, 8]."""
    n = len(dssp_node)
    eps = float(label_smoothing)
    probs = np.full((n, SS8_CLASSES), eps / SS8_CLASSES, dtype=np.float32)

    valid = dssp_node >= 0
    if valid.any():
        probs[valid, dssp_node[valid]] += 1.0 - eps

    # Nodes with invalid DSSP (index -100) get uniform distribution.
    invalid = dssp_node < 0
    if invalid.any():
        probs[invalid] = 1.0 / SS8_CLASSES

    return probs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    dataset_root = args.processed_root / args.dataset_name
    proteins_dir = dataset_root / "proteins"
    split_path = dataset_root / "splits.json"
    out_dir = args.teacher_cache_root / args.dataset_name

    if not split_path.exists():
        raise FileNotFoundError(f"Missing splits.json at {split_path}. Run preprocessing first.")

    out_dir.mkdir(parents=True, exist_ok=True)

    splits = json.loads(split_path.read_text())
    all_ids: list[str] = []
    for key in ["train", "val", "test", "pool_unassigned"]:
        all_ids.extend(splits.get(key, []))

    written = 0
    skipped = 0
    missing = 0

    for sample_id in all_ids:
        out_path = out_dir / f"{sample_id}.npz"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        npz_path = proteins_dir / f"{sample_id}.npz"
        if not npz_path.exists():
            logger.warning("Protein NPZ not found: %s", npz_path)
            missing += 1
            continue

        sample = dict(np.load(npz_path, allow_pickle=True))
        dssp_res = sample.get("dssp_idx", None)
        if dssp_res is None:
            logger.warning("No dssp_idx in %s; using uniform probs", sample_id)
            num_res = int(sample["aa_idx"].shape[0])
            dssp_node = np.full(num_res * ATOMS_PER_RESIDUE, -100, dtype=np.int64)
        else:
            dssp_node = np.repeat(dssp_res.astype(np.int64), ATOMS_PER_RESIDUE)

        probs = make_mock_probs(dssp_node, args.label_smoothing)
        np.savez_compressed(out_path, teacher_probs_node=probs)
        written += 1

    logger.info(
        "Mock teacher labels: %d written, %d skipped (already exist), %d missing source NPZ",
        written,
        skipped,
        missing,
    )
    logger.info("Cache written to: %s", out_dir)


if __name__ == "__main__":
    main()
