#!/usr/bin/env python3
"""Compare EMC timing: emc_max_nodes_per_graph=32 vs all nodes (one protein graph).

Loads the model once, then reports:
  (1) Time for the analytic node-score loop only (32 vs all nodes).
  (2) End-to-end ``emc_acquisition`` wall time for each setting (includes model load each call).

Example::

    conda activate /opt/dlami/nvme/envs/esm3_gnn_distill
    cd /opt/dlami/nvme/esm3_gnn_distill_baseline
    python scripts/benchmark_emc_node_cap.py \\
        --config configs/al_aws_dispef_m.yaml \\
        --checkpoint checkpoints/aws_al_emc_20260409_173011/round_00/best.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from active_learning.acquisition import _emc_node_score_last_linear, emc_acquisition
from data.pyg_dataset import DistillationGraphDataset
from models.factory import build_model
from train.config_utils import load_config
from train.utils import infer_device


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=_REPO_ROOT / "configs/al_aws_dispef_m.yaml")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sample-id", type=str, default="", help="Default: first train id in splits.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(cfg["paths"]["processed_root"])
    dataset_name = cfg["data"]["dataset_name"]
    default_splits = processed_root / dataset_name / "splits.json"

    sample_id = args.sample_id.strip()
    if not sample_id:
        splits_all = json.loads(default_splits.read_text())
        train_ids = sorted(splits_all.get("train", []))
        if not train_ids:
            raise SystemExit(f"No train ids in {default_splits}")
        sample_id = train_ids[0]

    npz_path = processed_root / dataset_name / "proteins" / f"{sample_id}.npz"
    if not npz_path.exists():
        raise SystemExit(f"Missing processed graph: {npz_path}")
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    one_split = {"train": [], "val": [], "test": [], "pool_unassigned": [sample_id]}
    test_cfg = copy.deepcopy(cfg)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        tf.write(json.dumps(one_split, indent=2))
        tmp_splits = Path(tf.name)
    test_cfg["data"]["splits_file"] = str(tmp_splits)

    device = infer_device(test_cfg["train"].get("device", "auto"))
    rng = np.random.default_rng(0)

    try:
        pool_dataset = DistillationGraphDataset(
            processed_root=processed_root,
            dataset_name=dataset_name,
            split_name="pool_unassigned",
            teacher_root=None,
            cutoff=float(test_cfg["graph"].get("cutoff", 8.0)),
            max_neighbors=int(test_cfg["graph"].get("max_neighbors", 64)),
            cache_graphs=False,
            splits_file=tmp_splits,
        )
        loader = DataLoader(pool_dataset, batch_size=1, shuffle=False, num_workers=0)
        batch = next(iter(loader)).to(device)

        model = build_model(test_cfg)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        final_layer = model.model.out_network[-1]
        if not isinstance(final_layer, nn.Linear):
            raise SystemExit(f"Expected nn.Linear head, got {type(final_layer)}")

        h_last = None

        def _cap(_m, inp, _out):
            nonlocal h_last
            h_last = inp[0].detach()

        hook = final_layer.register_forward_hook(_cap)
        t_fwd0 = time.perf_counter()
        with torch.no_grad():
            logits = model(batch)
        t_fwd = time.perf_counter() - t_fwd0
        hook.remove()

        probs = torch.softmax(logits.float(), dim=-1)
        node_h = h_last
        node_probs = probs
        n_nodes = int(node_h.size(0))

        # --- Micro: node-score loop only (same 32-node subsample EMC uses when n_nodes > 32) ---
        rng32 = np.random.default_rng(0)
        if n_nodes > 32:
            node_iter_32 = sorted(rng32.choice(n_nodes, size=32, replace=False).tolist())
        else:
            node_iter_32 = list(range(n_nodes))

        n_rep = 100 if n_nodes > 32 else 1
        t0 = time.perf_counter()
        for _ in range(n_rep):
            s32 = 0.0
            for ni in node_iter_32:
                s32 += _emc_node_score_last_linear(node_h[ni], node_probs[ni])
        t_loop_32 = (time.perf_counter() - t0) / n_rep

        n_rep_all = max(1, min(20, 2000 // max(1, n_nodes)))
        t0 = time.perf_counter()
        for _ in range(n_rep_all):
            s_all = 0.0
            for ni in range(n_nodes):
                s_all += _emc_node_score_last_linear(node_h[ni], node_probs[ni])
        t_loop_all = (time.perf_counter() - t0) / n_rep_all

        est_all_if_one_pass = t_loop_all
        est_32_if_one_pass = t_loop_32
        ratio_loop = est_all_if_one_pass / est_32_if_one_pass if est_32_if_one_pass > 0 else float("nan")

        print("=== One graph EMC node-cap benchmark ===")
        print(f"sample_id={sample_id}  n_nodes={n_nodes}  device={device}")
        print(f"single GNN forward (batch_size=1): {t_fwd*1000:.2f} ms")
        print()
        print("Analytic node-score loop only (excludes forward/load):")
        print(f"  32 nodes (mean over {len(node_iter_32)} sampled): {est_32_if_one_pass*1e6:.2f} µs per graph")
        print(f"  all {n_nodes} nodes:                         {est_all_if_one_pass*1e6:.2f} µs per graph")
        print(f"  ratio (all / 32): {ratio_loop:.2f}x  (theory ~{n_nodes / min(32, n_nodes):.2f}x if n>>32)")
        print()

        # --- End-to-end emc_acquisition (model load each call) ---
        def e2e_cfg(nodes_cap: int | None) -> dict:
            c = copy.deepcopy(test_cfg)
            al = c.setdefault("active_learning", {})
            al["strategy"] = "emc"
            if nodes_cap is None:
                al["emc_max_nodes_per_graph"] = None
            else:
                al["emc_max_nodes_per_graph"] = nodes_cap
            return c

        t0 = time.perf_counter()
        emc_acquisition(
            pool_ids=[sample_id],
            budget=1,
            rng=np.random.default_rng(0),
            cfg=e2e_cfg(32),
            checkpoint_path=args.checkpoint,
            device=device,
        )
        t_e2e_32 = time.perf_counter() - t0

        t0 = time.perf_counter()
        emc_acquisition(
            pool_ids=[sample_id],
            budget=1,
            rng=np.random.default_rng(0),
            cfg=e2e_cfg(None),
            checkpoint_path=args.checkpoint,
            device=device,
        )
        t_e2e_all = time.perf_counter() - t0

        print("End-to-end emc_acquisition (includes load_model + torch.load each run):")
        print(f"  emc_max_nodes_per_graph=32:  {t_e2e_32:.3f} s")
        print(f"  all nodes (null cap):        {t_e2e_all:.3f} s")
        print(f"  delta (all - 32):            {(t_e2e_all - t_e2e_32)*1000:.2f} ms")
    finally:
        tmp_splits.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
