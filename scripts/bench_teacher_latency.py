"""Per-query inference-latency benchmark for ESM3 teacher variants.

Motivates the active-learning story: if ESM3 takes X ms / protein, then
labeling the full 12k AL pool costs 12,000·X vs AL's 10k·X at the final
round — a constant saving that scales with teacher cost.  Larger ESM3
variants have larger X, so the AL saving scales with model size.

Loads N random proteins from a DISPEF-M split, rebuilds their sequence
strings from ``aa_idx`` (and optionally backbone coords from ``xyz``), and
times ``ESM3Teacher.predict_ss8_probs`` per protein on each requested model.
Reports median / p10 / p90 per-query latency and projects the total cost
for full-pool labeling vs the AL final labeled set.

Examples::

    # Local open-weight teacher only (what we actually use).
    python scripts/bench_teacher_latency.py --models esm3_sm_open_v1

    # Mix local + Forge-hosted variants.  Needs ESM_API_TOKEN in env.
    python scripts/bench_teacher_latency.py \\
        --models esm3_sm_open_v1 esm3-medium-2024-03 \\
        --backends local forge

    # With structure conditioning (matches how teacher labels were actually made).
    python scripts/bench_teacher_latency.py --models esm3_sm_open_v1 --use-structure
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.constants import INDEX_TO_AA1
from teacher.esm3_teacher import ESM3Teacher
from train.config_utils import load_config


# Default pool/AL budget numbers for the cost projection table.  Change via CLI.
DEFAULT_POOL_SIZE = 12_000
DEFAULT_AL_FINAL = 10_000
DEFAULT_AL_INITIAL = 2_500


def _aa_idx_to_sequence(aa_idx: np.ndarray) -> str:
    unknown = "X"
    return "".join(INDEX_TO_AA1.get(int(i), unknown) for i in aa_idx)


def _load_proteins(
    processed_root: Path,
    dataset_name: str,
    split_name: str,
    n: int,
    seed: int,
) -> List[Tuple[str, str, Optional[np.ndarray]]]:
    """Return a list of (sample_id, sequence, backbone_coords_ang_or_None)."""
    splits = json.loads((processed_root / dataset_name / "splits.json").read_text())
    pool = splits[split_name]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    proteins_dir = processed_root / dataset_name / "proteins"
    out: List[Tuple[str, str, Optional[np.ndarray]]] = []
    for i in idx:
        pid = pool[int(i)]
        data = np.load(proteins_dir / f"{pid}.npz")
        seq = _aa_idx_to_sequence(data["aa_idx"])
        # xyz layout is (L, 3_atoms, 3_coords) per data/preprocess_dispef.py.
        xyz = data["xyz"].astype(np.float32) if "xyz" in data.files else None
        out.append((pid, seq, xyz))
    return out


def _bench_one_model(
    model_name: str,
    backend: str,
    proteins: List[Tuple[str, str, Optional[np.ndarray]]],
    use_structure: bool,
    warmup: int,
) -> dict:
    print(f"\n=== {model_name}  (backend={backend}) ===", flush=True)
    teacher = ESM3Teacher(model_name=model_name, backend=backend)

    def _predict(seq: str, xyz: Optional[np.ndarray]) -> None:
        teacher.predict_ss8_probs(
            seq,
            backbone_coords_ang=(xyz if use_structure and xyz is not None else None),
        )

    # Warmup runs (not timed) to get weight loading + CUDA graphs + caches settled.
    for i in range(min(warmup, len(proteins))):
        _, seq, xyz = proteins[i]
        t0 = time.perf_counter()
        _predict(seq, xyz)
        print(f"  warmup {i+1}: {(time.perf_counter()-t0)*1000:.1f} ms  (L={len(seq)})", flush=True)

    # Timed runs.
    latencies_ms: List[float] = []
    lengths: List[int] = []
    for pid, seq, xyz in proteins:
        t0 = time.perf_counter()
        _predict(seq, xyz)
        dt_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(dt_ms)
        lengths.append(len(seq))
        print(f"  {pid}  L={len(seq):4d}  {dt_ms:7.1f} ms", flush=True)

    arr = np.asarray(latencies_ms)
    return {
        "model": model_name,
        "backend": backend,
        "use_structure": use_structure,
        "n_proteins": int(arr.size),
        "mean_length": float(np.mean(lengths)),
        "latency_ms": {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "std": float(arr.std()),
        },
    }


def _print_summary(results: List[dict], pool_size: int, al_final: int, al_initial: int) -> None:
    print(f"\n{'='*92}")
    print(f"{'model':<30} {'backend':>7} {'median ms':>11} {'p10 ms':>9} {'p90 ms':>9} "
          f"{'pool-label':>12} {'AL-final':>11} {'saving':>10}")
    print("-" * 92)
    for r in results:
        med_s = r["latency_ms"]["median"] / 1000.0
        pool_h = pool_size * med_s / 3600
        al_h = al_final * med_s / 3600
        savings_h = (pool_size - al_final) * med_s / 3600
        print(f"{r['model']:<30} {r['backend']:>7} "
              f"{r['latency_ms']['median']:>11.1f} "
              f"{r['latency_ms']['p10']:>9.1f} "
              f"{r['latency_ms']['p90']:>9.1f} "
              f"{pool_h:>10.2f} h "
              f"{al_h:>9.2f} h "
              f"{savings_h:>8.2f} h")
    print("=" * 92)
    print(f"Projection: labelling the full AL pool ({pool_size} proteins) vs the AL "
          f"final labeled set ({al_final} proteins); AL skips "
          f"{pool_size - al_final} queries "
          f"(= {(pool_size - al_final)/pool_size*100:.1f}% saving).")
    print(f"Initial cold-start labelling ({al_initial} proteins) still required.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/al_psc_dispef_m.yaml"),
                    help="Any config with paths.processed_root + data.dataset_name set.")
    ap.add_argument("--models", nargs="+", default=["esm3_sm_open_v1"],
                    help="Model names to benchmark (passed to ESM3Teacher).")
    ap.add_argument("--backends", nargs="+", default=None,
                    help="One backend per model (local|forge|auto). Default: 'auto' for all.")
    ap.add_argument("--split", type=str, default="train",
                    help="Split to draw proteins from.")
    ap.add_argument("--n-proteins", type=int, default=30,
                    help="Number of proteins to time per model.")
    ap.add_argument("--warmup", type=int, default=3,
                    help="Warmup forward passes per model (not timed).")
    ap.add_argument("--use-structure", action="store_true",
                    help="Condition on backbone coords (matches actual teacher labelling).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for protein selection.")
    ap.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    ap.add_argument("--al-final", type=int, default=DEFAULT_AL_FINAL)
    ap.add_argument("--al-initial", type=int, default=DEFAULT_AL_INITIAL)
    ap.add_argument("--output", type=Path, default=None,
                    help="Write JSON results here (default: stdout only).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    backends = args.backends or ["auto"] * len(args.models)
    if len(backends) != len(args.models):
        raise ValueError("--backends must have same length as --models (or be omitted).")

    print(f"Loading {args.n_proteins} proteins from {cfg['data']['dataset_name']} "
          f"({args.split} split), seed={args.seed}")
    proteins = _load_proteins(
        processed_root=Path(cfg["paths"]["processed_root"]),
        dataset_name=cfg["data"]["dataset_name"],
        split_name=args.split,
        n=args.n_proteins,
        seed=args.seed,
    )
    lens = [len(s) for _, s, _ in proteins]
    print(f"  sequence lengths: min={min(lens)} mean={np.mean(lens):.0f} "
          f"max={max(lens)} (n={len(proteins)})")
    print(f"  use_structure={args.use_structure}")

    if "HF_HOME" not in os.environ:
        print("  (HF_HOME not set — local weights may download to ~/.cache)")

    results: List[dict] = []
    for model_name, backend in zip(args.models, backends):
        try:
            r = _bench_one_model(
                model_name=model_name,
                backend=backend,
                proteins=proteins,
                use_structure=args.use_structure,
                warmup=args.warmup,
            )
            results.append(r)
        except Exception as exc:
            print(f"  ! {model_name} ({backend}) failed: {exc}", flush=True)

    _print_summary(results, args.pool_size, args.al_final, args.al_initial)

    if args.output is not None:
        args.output.write_text(json.dumps({
            "config": str(args.config),
            "split": args.split,
            "n_proteins": args.n_proteins,
            "use_structure": args.use_structure,
            "seed": args.seed,
            "pool_size": args.pool_size,
            "al_final": args.al_final,
            "al_initial": args.al_initial,
            "results": results,
        }, indent=2))
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
