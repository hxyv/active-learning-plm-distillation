"""Generate and cache per-protein teacher SS8 probabilities for DISPEF."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from teacher.base import FileTeacher
from teacher.esm3_teacher import ESM3Teacher
from teacher.label_cache import expand_residue_probs_to_nodes, save_teacher_probs

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ESM3 teacher SS8 probabilities")
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, default="dispef_m")
    parser.add_argument("--teacher-cache-root", type=Path, required=True)
    parser.add_argument("--provider", type=str, default="esm3", choices=["esm3", "file"])
    parser.add_argument("--provider-file-root", type=Path, default=None)
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "val", "test", "pool_unassigned"])
    parser.add_argument("--model-name", type=str, default="esm3_sm_open_v1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--esm-backend",
        type=str,
        default="auto",
        choices=["auto", "local", "forge"],
        help="ESM3 backend: local HF weights or EvolutionaryScale Forge API",
    )
    parser.add_argument("--forge-url", type=str, default="https://forge.evolutionaryscale.ai")
    parser.add_argument("--forge-token-env", type=str, default="ESM_API_TOKEN")
    parser.add_argument("--max-samples", type=int, default=0, help="Debug cap; 0 means all samples")
    parser.add_argument("--num-shards", type=int, default=1, help="Total parallel shards")
    parser.add_argument("--shard-id", type=int, default=0, help="Current shard id in [0, num_shards)")
    parser.add_argument(
        "--fetch-uniprot-sequences",
        action="store_true",
        help="If stored sequence is missing/unknown, fetch by sample accession from UniProt",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_sample_ids(dataset_root: Path, split: str) -> List[str]:
    splits = json.loads((dataset_root / "splits.json").read_text())
    if split == "all":
        union = set()
        for k in ["train", "val", "test", "pool_unassigned"]:
            union.update(splits.get(k, []))
        return sorted(union)
    return sorted(splits.get(split, []))


def fetch_uniprot_sequence(accession: str, cache: Dict[str, str]) -> str:
    acc = accession.strip()
    if acc in cache:
        return cache[acc]
    try:
        import requests
    except Exception as exc:
        raise RuntimeError(f"requests not available for UniProt fetch: {exc}")

    urls = [
        f"https://rest.uniprot.org/uniprotkb/{acc}.fasta",
        f"https://www.uniprot.org/uniprot/{acc}.fasta",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=20)
            if r.status_code != 200:
                continue
            lines = [line.strip() for line in r.text.splitlines() if line and not line.startswith(">")]
            seq = "".join(lines).upper()
            if seq:
                cache[acc] = seq
                return seq
        except Exception:
            continue
    raise RuntimeError(f"Failed to fetch UniProt sequence for accession: {acc}")


def load_sequence(sample_path: Path) -> Tuple[str, str]:
    with np.load(sample_path, allow_pickle=True) as arr:
        sample_id = str(arr["sample_id"]) if "sample_id" in arr else sample_path.stem
        file_base = str(arr["file_base"]) if "file_base" in arr else sample_id
        if "sequence" in arr:
            seq = "".join(str(x) for x in arr["sequence"].tolist())
            return seq, file_base
        aa_idx = arr["aa_idx"]
    # Last resort: unknown letters with same length.
    return "X" * int(len(aa_idx)), file_base


def main() -> None:
    args = parse_args()
    setup_logging()

    dataset_root = args.processed_root / args.dataset_name
    proteins_dir = dataset_root / "proteins"
    teacher_root = args.teacher_cache_root / args.dataset_name
    teacher_root.mkdir(parents=True, exist_ok=True)

    sample_ids = get_sample_ids(dataset_root, args.split)
    if not sample_ids:
        raise RuntimeError(f"No sample IDs found for split={args.split} at {dataset_root}")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard-id < num-shards")
    if args.num_shards > 1:
        sample_ids = [sid for i, sid in enumerate(sample_ids) if (i % args.num_shards) == args.shard_id]
    if args.max_samples > 0:
        sample_ids = sample_ids[: args.max_samples]

    if args.provider == "esm3":
        teacher = ESM3Teacher(
            model_name=args.model_name,
            device=args.device,
            temperature=args.temperature,
            backend=args.esm_backend,
            forge_url=args.forge_url,
            forge_token_env=args.forge_token_env,
        )
    else:
        if args.provider_file_root is None:
            raise ValueError("--provider-file-root is required when --provider=file")
        teacher = FileTeacher(probs_root=args.provider_file_root)

    seq_cache_path = teacher_root / "sequence_cache_uniprot.json"
    if seq_cache_path.exists():
        seq_cache: Dict[str, str] = json.loads(seq_cache_path.read_text())
    else:
        seq_cache = {}

    failures = []
    for sid in tqdm(sample_ids, desc="Teacher labeling"):
        sample_path = proteins_dir / f"{sid}.npz"
        out_path = teacher_root / f"{sid}.npz"

        if not sample_path.exists():
            failures.append((sid, "missing_sample"))
            continue

        if out_path.exists() and not args.overwrite:
            continue

        try:
            seq, accession = load_sequence(sample_path)
        except Exception as exc:
            failures.append((sid, f"sample_read_failed: {exc}"))
            logger.warning("Skipping unreadable sample %s: %s", sid, exc)
            continue
        if args.fetch_uniprot_sequences and (not seq or set(seq) == {"X"}):
            try:
                seq = fetch_uniprot_sequence(accession, seq_cache)
            except Exception as exc:
                failures.append((sid, f"sequence_fetch_failed: {exc}"))
                continue

        try:
            # Load backbone coords (nm) and convert to Angstrom for structure-conditioned ESM3.
            # This makes teacher SS8 predictions consistent with mdtraj DSSP labels
            # computed from the same DISPEF-M simulation coordinates (paper-faithful setup).
            backbone_coords_ang = None
            try:
                sample_data = np.load(sample_path, allow_pickle=True)
                if "coords" in sample_data:
                    backbone_coords_ang = sample_data["coords"].astype(np.float32) * 10.0  # nm -> Ang
            except Exception as exc:
                logger.debug("Could not load backbone coords for %s (%s); using sequence-only.", sid, exc)

            residue_probs = teacher.predict_ss8_probs(
                seq, sample_id=sid, backbone_coords_ang=backbone_coords_ang
            )
            residue_probs = residue_probs.astype(np.float32)
            residue_probs = residue_probs / np.clip(residue_probs.sum(axis=-1, keepdims=True), 1e-8, None)
            if residue_probs.shape != (len(seq), 8):
                raise ValueError(f"Expected {(len(seq), 8)}, got {residue_probs.shape}")

            node_probs = expand_residue_probs_to_nodes(residue_probs, atoms_per_residue=3)
            save_teacher_probs(
                out_path=out_path,
                sample_id=sid,
                residue_probs=residue_probs,
                node_probs=node_probs,
            )
        except Exception as exc:
            logger.exception("Teacher labeling failed for %s", sid)
            failures.append((sid, str(exc)))

    summary = {
        "dataset": args.dataset_name,
        "split": args.split,
        "provider": args.provider,
        "esm_backend": args.esm_backend if args.provider == "esm3" else None,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "total": len(sample_ids),
        "num_failures": len(failures),
        "failures": failures[:100],
    }
    (teacher_root / f"teacher_label_summary_{args.split}.json").write_text(json.dumps(summary, indent=2))
    seq_cache_path.write_text(json.dumps(seq_cache, indent=2))

    if failures:
        logger.warning("Teacher labeling finished with %d failures", len(failures))
    else:
        logger.info("Teacher labeling complete with no failures")


if __name__ == "__main__":
    main()
