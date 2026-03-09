"""Evaluate a trained student model and export per-protein predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from data.pyg_dataset import DistillationGraphDataset
from eval.metrics import compute_teacher_metrics
from models.factory import build_model
from train.config_utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SS8 distillation student")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    dataset = DistillationGraphDataset(
        processed_root=Path(cfg["paths"]["processed_root"]),
        dataset_name=cfg["data"]["dataset_name"],
        split_name=args.split,
        teacher_root=Path(cfg["paths"]["teacher_cache_root"]) / cfg["data"]["dataset_name"],
        cutoff=float(cfg["graph"].get("cutoff", 8.0)),
        max_neighbors=int(cfg["graph"].get("max_neighbors", 64)),
        cache_graphs=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pred_root = output_dir / "predictions" / args.split
    if args.save_predictions:
        pred_root.mkdir(parents=True, exist_ok=True)

    accum = {
        "teacher_ce": 0.0,
        "teacher_kl": 0.0,
        "teacher_top1_acc": 0.0,
        "dssp_acc_weighted": 0.0,
        "dssp_count": 0.0,
        "n_nodes": 0.0,
    }

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            teacher_probs = batch.teacher_probs
            dssp_idx = batch.dssp_idx

            m = compute_teacher_metrics(logits, teacher_probs, dssp_idx)
            n_nodes = float(logits.shape[0])
            accum["teacher_ce"] += m["teacher_ce"] * n_nodes
            accum["teacher_kl"] += m["teacher_kl"] * n_nodes
            accum["teacher_top1_acc"] += m["teacher_top1_acc"] * n_nodes
            accum["n_nodes"] += n_nodes

            if float(m.get("dssp_count", 0.0)) > 0:
                accum["dssp_acc_weighted"] += float(m["dssp_acc"]) * float(m["dssp_count"])
                accum["dssp_count"] += float(m["dssp_count"])

            if args.save_predictions:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                logits_np = logits.cpu().numpy()
                teacher_np = teacher_probs.cpu().numpy()
                dssp_np = dssp_idx.cpu().numpy()
                node_to_res = batch.node_to_residue.cpu().numpy()

                ptr = batch.ptr.cpu().numpy()
                sample_ids = batch.sample_id
                if isinstance(sample_ids, str):
                    sample_ids = [sample_ids]

                for i, sid in enumerate(sample_ids):
                    st, ed = int(ptr[i]), int(ptr[i + 1])
                    np.savez_compressed(
                        pred_root / f"{sid}.npz",
                        sample_id=sid,
                        logits=logits_np[st:ed],
                        probs=probs[st:ed],
                        teacher_probs=teacher_np[st:ed],
                        dssp_idx=dssp_np[st:ed],
                        node_to_residue=node_to_res[st:ed],
                    )

    denom = max(accum["n_nodes"], 1.0)
    summary = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "teacher_ce": accum["teacher_ce"] / denom,
        "teacher_kl": accum["teacher_kl"] / denom,
        "teacher_top1_acc": accum["teacher_top1_acc"] / denom,
        "dssp_acc": accum["dssp_acc_weighted"] / max(accum["dssp_count"], 1.0)
        if accum["dssp_count"] > 0
        else float("nan"),
        "n_nodes": accum["n_nodes"],
        "dssp_count": accum["dssp_count"],
    }

    (output_dir / f"eval_summary_{args.split}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
