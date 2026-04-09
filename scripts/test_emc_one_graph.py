#!/usr/bin/env python3
"""Run EMC acquisition logic on exactly one graph (sanity check / timing).

EMC uses an analytic last-layer score (no per-class full-graph backward).
Use this before waiting on the whole pool (~9500 graphs after round 0).

Example::

    conda activate /opt/dlami/nvme/envs/esm3_gnn_distill
    cd /opt/dlami/nvme/esm3_gnn_distill_baseline
    python scripts/test_emc_one_graph.py \\
        --config configs/al_aws_dispef_m.yaml \\
        --checkpoint outputs/al/your_run/round_00/best.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Allow `python scripts/test_emc_one_graph.py` from repo root without PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from active_learning.acquisition import emc_acquisition
from train.config_utils import load_config
from train.utils import infer_device

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("test_emc_one_graph")


def main() -> None:
    p = argparse.ArgumentParser(description="EMC on a single protein graph")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="best.pt or last.pt from a training round")
    p.add_argument(
        "--sample-id",
        type=str,
        default="",
        help="UniProt id (must have proteins/<id>.npz). Default: first id in original splits.json train split.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(cfg["paths"]["processed_root"])
    dataset_name = cfg["data"]["dataset_name"]
    dataset_root = processed_root / dataset_name
    default_splits = dataset_root / "splits.json"

    sample_id = args.sample_id.strip()
    if not sample_id:
        splits_all = json.loads(default_splits.read_text())
        train_ids = sorted(splits_all.get("train", []))
        if not train_ids:
            raise SystemExit(f"No train ids in {default_splits}")
        sample_id = train_ids[0]
        logger.info("Using first train id: %s", sample_id)

    npz_path = dataset_root / "proteins" / f"{sample_id}.npz"
    if not npz_path.exists():
        raise SystemExit(f"Missing processed graph: {npz_path}")

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    one_split = {
        "train": [],
        "val": [],
        "test": [],
        "pool_unassigned": [sample_id],
    }

    test_cfg = copy.deepcopy(cfg)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        tf.write(json.dumps(one_split, indent=2))
        tmp_splits = Path(tf.name)
    test_cfg["data"]["splits_file"] = str(tmp_splits)

    device = infer_device(test_cfg["train"].get("device", "auto"))
    logger.info("Device: %s", device)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Sample: %s (%s)", sample_id, npz_path)

    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    try:
        selected = emc_acquisition(
            pool_ids=[sample_id],
            budget=1,
            rng=rng,
            cfg=test_cfg,
            checkpoint_path=args.checkpoint,
            device=device,
        )
    finally:
        tmp_splits.unlink(missing_ok=True)

    elapsed = time.perf_counter() - t0
    logger.info("Done in %.2f s", elapsed)
    logger.info("Selected (should be [%s]): %s", sample_id, selected)


if __name__ == "__main__":
    main()
