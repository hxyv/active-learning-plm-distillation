"""Preprocess DISPEF into backbone-only NPZ files.

Supports two input modes:
1) Structure files (.pdb/.cif/.mmcif/.ent)
2) DISPEF .pt tensors (DISPEF_M_tr.pt / DISPEF_M_te.pt, etc.)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from data.constants import AA1_TO_INDEX, ATOM_TYPES, aa3_to_index
from data.dssp import compute_dssp_from_bb4_coords, compute_dssp_labels
from data.io_utils import (
    build_split_from_train_only,
    discover_split_ids,
    ensure_dir,
    map_split_ids_to_samples,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess DISPEF into backbone-only NPZ files")
    parser.add_argument("--raw-root", type=Path, required=True, help="DISPEF raw root")
    parser.add_argument("--processed-root", type=Path, required=True, help="Output root for processed files")
    parser.add_argument("--dataset-name", type=str, default="dispef_m", help="Dataset name")
    parser.add_argument("--input-format", type=str, default="auto", choices=["auto", "pt", "structure"])
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Val fraction from train if missing")
    parser.add_argument("--test-fraction", type=float, default=0.1, help="Fallback test fraction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-residues", type=int, default=10)
    parser.add_argument("--with-dssp", action="store_true", help="Use DSSP when parsing structure files")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument(
        "--no-clean-output",
        action="store_true",
        help="Do not clear existing processed protein NPZ files before writing new outputs",
    )
    parser.add_argument(
        "--fetch-uniprot-sequences",
        action="store_true",
        help="For .pt mode: fetch UniProt sequences by entry ID to populate amino-acid identities",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def save_sample(sample: Dict[str, object], proteins_dir: Path) -> Path:
    sample_id = str(sample["sample_id"])
    out_path = proteins_dir / f"{sample_id}.npz"
    np.savez_compressed(
        out_path,
        sample_id=sample_id,
        file_base=sample["file_base"],
        chain_id=sample["chain_id"],
        source_path=sample["source_path"],
        aa_idx=sample["aa_idx"],
        coords=sample["coords"],
        resseq=sample["resseq"],
        icode=sample["icode"],
        sequence=sample["sequence"],
        dssp_idx=sample["dssp_idx"],
        atom_order=np.asarray(ATOM_TYPES, dtype="U3"),
    )
    return out_path


def write_manifest(samples: List[Dict[str, object]], manifest_csv: Path) -> None:
    fields = ["sample_id", "file_base", "chain_id", "num_residues", "source_path"]
    with manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in samples:
            writer.writerow(
                {
                    "sample_id": s["sample_id"],
                    "file_base": s["file_base"],
                    "chain_id": s["chain_id"],
                    "num_residues": len(s["aa_idx"]),
                    "source_path": s["source_path"],
                }
            )


def fallback_split(ids: List[str], seed: int, test_fraction: float, val_fraction: float) -> Dict[str, object]:
    ids = sorted(set(ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * test_fraction))
    test_ids = sorted(ids[:n_test])
    remain = ids[n_test:]
    n_val = max(1, int(len(remain) * val_fraction))
    val_ids = sorted(remain[:n_val])
    train_ids = sorted(remain[n_val:])
    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "pool_unassigned": [],
        "split_source": "deterministic_fallback_no_official_split",
    }


def build_splits_structure(
    raw_root: Path,
    sample_ids: List[str],
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> Dict[str, object]:
    discovered = discover_split_ids(raw_root)
    if not discovered:
        logger.warning("No official split detected; using deterministic fallback split")
        return fallback_split(sample_ids, seed, test_fraction, val_fraction)

    mapped = map_split_ids_to_samples(discovered, sample_ids)
    train_ids = mapped.get("train", [])
    val_ids = mapped.get("val", [])
    test_ids = mapped.get("test", [])
    pool_ids = mapped.get("unassigned", [])

    if not train_ids or not test_ids:
        logger.warning("Could not map enough official split IDs; using fallback split")
        return fallback_split(sample_ids, seed, test_fraction, val_fraction)

    if not val_ids:
        train_ids, val_ids = build_split_from_train_only(train_ids, val_fraction, seed)
        split_source = "official_train_test_plus_val_from_train"
    else:
        split_source = "official_train_val_test"

    return {
        "train": sorted(set(train_ids)),
        "val": sorted(set(val_ids)),
        "test": sorted(set(test_ids)),
        "pool_unassigned": sorted(set(pool_ids)),
        "split_source": split_source,
    }


# ===== Structure-file mode =====

def load_structure(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".pdb", ".ent"}:
        from Bio.PDB import PDBParser

        parser = PDBParser(QUIET=True)
        return parser.get_structure(path.stem, str(path))
    if suffix in {".cif", ".mmcif"}:
        from Bio.PDB import MMCIFParser

        parser = MMCIFParser(QUIET=True)
        return parser.get_structure(path.stem, str(path))
    raise ValueError(f"Unsupported structure extension: {path}")


def iter_structure_files(raw_root: Path) -> Iterable[Path]:
    exts = ("*.pdb", "*.ent", "*.cif", "*.mmcif")
    paths: List[Path] = []
    for ext in exts:
        paths.extend(raw_root.rglob(ext))
    for path in sorted(set(paths)):
        try:
            if path.stat().st_size < 100:
                continue
        except OSError:
            continue
        yield path


def extract_chain_sample(
    structure,
    source_path: Path,
    chain,
    dssp_map: Dict[Tuple[str, int, str], int],
    min_residues: int,
) -> Optional[Dict[str, object]]:
    aa_idx: List[int] = []
    coords: List[np.ndarray] = []
    resseqs: List[int] = []
    icodes: List[str] = []
    seq_chars: List[str] = []
    dssp_idx: List[int] = []

    from data.constants import INDEX_TO_AA1

    for residue in chain.get_residues():
        hetflag, resseq, icode = residue.id
        if hetflag != " ":
            continue

        atom_bundle = []
        missing = False
        for atom_name in ATOM_TYPES:
            if atom_name not in residue:
                missing = True
                break
            atom_bundle.append(np.asarray(residue[atom_name].coord, dtype=np.float32))
        if missing:
            continue

        resname = residue.get_resname().upper()
        aa_i = aa3_to_index(resname)
        aa_idx.append(aa_i)
        coords.append(np.stack(atom_bundle, axis=0))
        resseqs.append(int(resseq))
        icode_str = str(icode).strip() or " "
        icodes.append(icode_str)
        seq_chars.append(INDEX_TO_AA1.get(aa_i, "X"))

        dkey = (chain.id, int(resseq), icode_str)
        dssp_idx.append(int(dssp_map.get(dkey, -100)))

    if len(aa_idx) < min_residues:
        return None

    file_base = source_path.stem
    chain_id = str(chain.id).strip() or "_"
    sample_id = f"{file_base}_{chain_id}"

    return {
        "sample_id": sample_id,
        "file_base": file_base,
        "chain_id": chain_id,
        "source_path": str(source_path),
        "aa_idx": np.asarray(aa_idx, dtype=np.int64),
        "coords": np.asarray(coords, dtype=np.float32),
        "resseq": np.asarray(resseqs, dtype=np.int32),
        "icode": np.asarray(icodes, dtype="U1"),
        "sequence": np.asarray(seq_chars, dtype="U1"),
        "dssp_idx": np.asarray(dssp_idx, dtype=np.int64),
    }


def preprocess_structure_mode(args: argparse.Namespace, dataset_root: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    proteins_dir = dataset_root / "proteins"
    ensure_dir(proteins_dir)

    samples: List[Dict[str, object]] = []
    n_files = 0
    n_chains = 0

    for struct_path in iter_structure_files(args.raw_root):
        n_files += 1
        if args.max_files > 0 and n_files > args.max_files:
            break

        try:
            structure = load_structure(struct_path)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", struct_path, exc)
            continue

        dssp_map: Dict[Tuple[str, int, str], int] = {}
        if args.with_dssp:
            dssp_map = compute_dssp_labels(structure, struct_path)

        try:
            model = structure[0]
        except Exception:
            logger.warning("Structure has no model 0: %s", struct_path)
            continue

        for chain in model.get_chains():
            n_chains += 1
            sample = extract_chain_sample(
                structure=structure,
                source_path=struct_path,
                chain=chain,
                dssp_map=dssp_map,
                min_residues=args.min_residues,
            )
            if sample is None:
                continue
            save_sample(sample, proteins_dir)
            samples.append(sample)

    if not samples:
        raise RuntimeError(
            "No valid chains extracted from structure files. "
            "If your DISPEF download contains *.pt files, rerun with --input-format pt."
        )

    splits = build_splits_structure(
        raw_root=args.raw_root,
        sample_ids=[str(s["sample_id"]) for s in samples],
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )

    meta = {
        "input_format": "structure",
        "num_files_seen": n_files,
        "num_chains_seen": n_chains,
        "num_samples_kept": len(samples),
        "with_dssp": bool(args.with_dssp),
    }
    return samples, {"splits": splits, "meta": meta}


# ===== PT mode =====

def find_dispef_pt_files(raw_root: Path, dataset_name: str) -> Tuple[Optional[Path], Optional[Path]]:
    name = dataset_name.lower()
    if name in {"dispef_m", "m", "dispef-m"}:
        tr_pat, te_pat = "*DISPEF_M_tr.pt", "*DISPEF_M_te.pt"
    elif name in {"dispef_s", "s", "dispef-s"}:
        tr_pat, te_pat = "*DISPEF_S_tr.pt", "*DISPEF_S_te.pt"
    else:
        tr_pat, te_pat = "*tr.pt", "*te.pt"

    tr_files = sorted(raw_root.rglob(tr_pat))
    te_files = sorted(raw_root.rglob(te_pat))
    tr = tr_files[0] if tr_files else None
    te = te_files[0] if te_files else None
    return tr, te


def load_dispef_dataset_obj(pt_path: Path, raw_root: Path):
    import torch

    sys.path.insert(0, str(raw_root))
    try:
        import dataset_prep  # type: ignore  # noqa
    except Exception as exc:
        raise RuntimeError(f"Failed importing dataset_prep.py from {raw_root}: {exc}")

    ds = torch.load(pt_path, weights_only=False)
    return ds, dataset_prep


def build_name_inverse(dataset_prep_module) -> Dict[int, str]:
    embedder = dataset_prep_module.get_embedding("name")
    classes = list(embedder.embed.classes_)
    return {i: str(name) for i, name in enumerate(classes)}


def fetch_uniprot_sequence(accession: str, cache: Dict[str, str]) -> Optional[str]:
    acc = accession.strip()
    if not acc:
        return None
    if acc in cache:
        return cache[acc]

    try:
        import requests
    except Exception:
        return None

    urls = [
        f"https://rest.uniprot.org/uniprotkb/{acc}.fasta",
        f"https://www.uniprot.org/uniprot/{acc}.fasta",
    ]

    seq = None
    for url in urls:
        try:
            r = requests.get(url, timeout=20)
            if r.status_code != 200:
                continue
            lines = [line.strip() for line in r.text.splitlines() if line and not line.startswith(">")]
            seq_candidate = "".join(lines).upper()
            seq_candidate = re.sub(r"[^A-Z]", "", seq_candidate)
            if seq_candidate:
                seq = seq_candidate
                break
        except Exception:
            continue

    if seq is not None:
        cache[acc] = seq
    return seq


def infer_aa_from_atom_block(atom_names: np.ndarray) -> str:
    """Infer one-letter residue identity from atom names in a residue block."""
    names = [str(x) for x in atom_names.tolist()]

    # Drop hydrogens and backbone atoms.
    side = {
        n
        for n in names
        if not n.startswith("H") and n not in {"N", "CA", "C", "O", "OXT"}
    }

    if not side:
        return "G"
    if "SG" in side:
        return "C"
    if "NZ" in side:
        return "K"
    if "SD" in side:
        return "M"
    if "OH" in side:
        return "Y"
    if "OG" in side:
        return "S"
    if "OG1" in side:
        return "T"
    if "ND2" in side:
        return "N"
    if "NE2" in side and "OE1" in side:
        return "Q"
    if "OD2" in side or {"OD1", "OD2"}.issubset(side):
        return "D"
    if "OE2" in side or ({"OE1", "OE2"}.issubset(side) and "NE2" not in side):
        return "E"
    if {"NH1", "NH2"}.intersection(side):
        return "R"
    if {"ND1", "NE2"}.intersection(side) and "NH1" not in side and "NH2" not in side:
        return "H"
    if {"CH2", "CZ2", "CZ3", "NE1"}.issubset(side):
        return "W"
    if {"CG1", "CG2", "CD1"}.issubset(side):
        return "I"
    if {"CG1", "CG2"}.issubset(side):
        return "V"
    if {"CD1", "CD2"}.issubset(side):
        if {"CE1", "CE2", "CZ"}.issubset(side):
            return "F"
        return "L"
    if "CD" in side and "CG" in side and "CB" in side and "CD1" not in side and "CD2" not in side:
        return "P"
    if side == {"CB"}:
        return "A"
    return "X"


def extract_bb4_from_pt_sample(
    coords: np.ndarray,
    atom_name_idx: np.ndarray,
    name_inv: Dict[int, str],
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """Extract backbone atoms from a DISPEF PT sample.

    Returns:
        bb3_coords: (L, 3, 3) float32 – [N, CA, C] per residue (model input).
        bb4_coords: (L, 4, 3) float32 – [N, CA, C, O] per residue (for DSSP).
        aa1_list:   list of L one-letter AA codes.
        num_res:    L.

    Residues missing N, CA, or C are dropped.  For residues missing O, the O
    coordinate is set to the zero vector (DSSP will produce -100 for those).
    """
    atom_names = np.asarray([name_inv.get(int(x), "UNK") for x in atom_name_idx.tolist()], dtype=object)

    ca_idx = np.where(atom_names == "CA")[0]
    if len(ca_idx) == 0:
        return (
            np.empty((0, 3, 3), dtype=np.float32),
            np.empty((0, 4, 3), dtype=np.float32),
            [],
            0,
        )

    bb3_residues: List[np.ndarray] = []
    bb4_residues: List[np.ndarray] = []
    aa1_list: List[str] = []

    for i, ca in enumerate(ca_idx):
        st = 0 if i == 0 else int(ca_idx[i - 1] + 1)
        ed = int(len(atom_names)) if i == len(ca_idx) - 1 else int(ca_idx[i + 1])
        block_names = atom_names[st:ed]

        n_rel = np.where(block_names == "N")[0]
        c_rel = np.where(block_names == "C")[0]
        if len(n_rel) == 0 or len(c_rel) == 0:
            continue

        n_abs_candidates = [st + int(x) for x in n_rel if st + int(x) < ca]
        c_abs_candidates = [st + int(x) for x in c_rel if st + int(x) > ca]
        if not n_abs_candidates or not c_abs_candidates:
            continue

        n_abs = n_abs_candidates[-1]
        c_abs = c_abs_candidates[0]

        # Locate the carbonyl O: first "O" atom after C within the residue block.
        o_rel = np.where(block_names == "O")[0]
        o_abs_candidates = [st + int(x) for x in o_rel if st + int(x) > c_abs]
        o_coord = coords[o_abs_candidates[0]] if o_abs_candidates else np.zeros(3, dtype=np.float32)

        bb3_residues.append(np.stack([coords[n_abs], coords[ca], coords[c_abs]], axis=0))
        bb4_residues.append(np.stack([coords[n_abs], coords[ca], coords[c_abs], o_coord], axis=0))
        aa1_list.append(infer_aa_from_atom_block(block_names))

    if not bb3_residues:
        return (
            np.empty((0, 3, 3), dtype=np.float32),
            np.empty((0, 4, 3), dtype=np.float32),
            [],
            0,
        )

    bb3 = np.asarray(bb3_residues, dtype=np.float32)
    bb4 = np.asarray(bb4_residues, dtype=np.float32)
    return bb3, bb4, aa1_list, int(bb3.shape[0])


def preprocess_pt_mode(args: argparse.Namespace, dataset_root: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    proteins_dir = dataset_root / "proteins"
    ensure_dir(proteins_dir)

    tr_path, te_path = find_dispef_pt_files(args.raw_root, args.dataset_name)
    if tr_path is None or te_path is None:
        raise RuntimeError(
            f"Could not locate train/test .pt files for {args.dataset_name} under {args.raw_root}."
        )

    logger.info("Using PT mode with train=%s test=%s", tr_path, te_path)

    tr_ds, dataset_prep_mod = load_dispef_dataset_obj(tr_path, args.raw_root)
    te_ds, _ = load_dispef_dataset_obj(te_path, args.raw_root)

    name_inv = build_name_inverse(dataset_prep_mod)

    seq_cache_path = dataset_root / "sequence_cache_uniprot.json"
    if seq_cache_path.exists():
        seq_cache: Dict[str, str] = json.loads(seq_cache_path.read_text())
    else:
        seq_cache = {}

    sample_counts = defaultdict(int)
    samples: List[Dict[str, object]] = []
    train_ids: List[str] = []
    test_ids: List[str] = []

    seen = 0
    for split_name, ds in [("train", tr_ds), ("test", te_ds)]:
        logger.info("Parsing %s split (%d samples)", split_name, len(ds))
        for i in range(len(ds)):
            if args.max_files > 0 and seen >= args.max_files:
                break
            tup = ds[i]
            if not isinstance(tup, (tuple, list)):
                continue

            if len(tup) >= 7:
                coords_t = tup[0]
                atom_name_t = tup[4]
                entry = str(tup[6])
            else:
                logger.warning("Unexpected sample format in %s at idx %d", split_name, i)
                continue

            coords = np.asarray(coords_t, dtype=np.float32)
            atom_name_idx = np.asarray(atom_name_t, dtype=np.int64)

            bb3_coords, bb4_coords, aa1_inferred, num_res = extract_bb4_from_pt_sample(
                coords, atom_name_idx, name_inv
            )
            if num_res < args.min_residues:
                continue

            # Ensure unique sample IDs across full dataset.
            sample_counts[entry] += 1
            suffix = sample_counts[entry]
            sample_id = entry if suffix == 1 else f"{entry}_{suffix}"

            seq_inferred = "".join(aa1_inferred) if aa1_inferred else ("X" * num_res)
            seq_str = None
            if args.fetch_uniprot_sequences:
                seq_str = fetch_uniprot_sequence(entry, seq_cache)

            if seq_str is not None and len(seq_str) == num_res:
                final_seq = seq_str
            else:
                final_seq = seq_inferred

            aa_idx = np.asarray([AA1_TO_INDEX.get(ch, AA1_TO_INDEX["X"]) for ch in final_seq], dtype=np.int64)
            seq_chars = np.asarray(list(final_seq), dtype="U1")

            # Compute 8-class DSSP from backbone N, CA, C, O coordinates.
            dssp_idx = compute_dssp_from_bb4_coords(bb4_coords, list(final_seq))

            sample = {
                "sample_id": sample_id,
                "file_base": entry,
                "chain_id": "_",
                "source_path": str(tr_path if split_name == "train" else te_path),
                "aa_idx": aa_idx,
                "coords": bb3_coords,
                "resseq": np.arange(num_res, dtype=np.int32),
                "icode": np.asarray([" "] * num_res, dtype="U1"),
                "sequence": seq_chars,
                "dssp_idx": dssp_idx,
            }

            save_sample(sample, proteins_dir)
            samples.append(sample)
            seen += 1
            if split_name == "train":
                train_ids.append(sample_id)
            else:
                test_ids.append(sample_id)
        if args.max_files > 0 and seen >= args.max_files:
            break

    if not samples:
        raise RuntimeError("No valid samples extracted from PT files")

    train_ids = sorted(set(train_ids))
    test_ids = sorted(set(test_ids))
    train_ids, val_ids = build_split_from_train_only(train_ids, args.val_fraction, args.seed)
    split_source = "pt_files_official_train_test"
    if val_ids:
        split_source = "pt_files_train_test_plus_val_from_train"

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "pool_unassigned": [],
        "split_source": split_source,
    }

    seq_cache_path.write_text(json.dumps(seq_cache, indent=2))

    meta = {
        "input_format": "pt",
        "train_pt": str(tr_path),
        "test_pt": str(te_path),
        "num_samples_kept": len(samples),
        "fetch_uniprot_sequences": bool(args.fetch_uniprot_sequences),
        "num_cached_sequences": len(seq_cache),
    }
    return samples, {"splits": splits, "meta": meta}


def main() -> None:
    args = parse_args()
    setup_logging()

    logger.info("Starting DISPEF preprocessing")
    logger.info("Raw root: %s", args.raw_root)
    logger.info("Processed root: %s", args.processed_root)

    dataset_root = args.processed_root / args.dataset_name
    ensure_dir(dataset_root)
    proteins_dir = dataset_root / "proteins"
    ensure_dir(proteins_dir)

    if not args.no_clean_output:
        removed = 0
        for old_npz in proteins_dir.glob("*.npz"):
            old_npz.unlink(missing_ok=True)
            removed += 1
        if removed > 0:
            logger.info("Cleared %d previous NPZ files from %s", removed, proteins_dir)

    pt_train, pt_test = find_dispef_pt_files(args.raw_root, args.dataset_name)
    structure_candidates = list(iter_structure_files(args.raw_root))

    if args.input_format == "pt":
        mode = "pt"
    elif args.input_format == "structure":
        mode = "structure"
    else:
        if pt_train is not None and pt_test is not None:
            mode = "pt"
        elif structure_candidates:
            mode = "structure"
        else:
            raise RuntimeError(
                "Could not detect supported DISPEF input format under raw-root. "
                "Expected either DISPEF_*_tr.pt / DISPEF_*_te.pt files or PDB/CIF structures."
            )

    logger.info("Selected preprocessing mode: %s", mode)

    if mode == "pt":
        samples, bundle = preprocess_pt_mode(args, dataset_root)
    else:
        samples, bundle = preprocess_structure_mode(args, dataset_root)

    manifest_csv = dataset_root / "manifest.csv"
    write_manifest(samples, manifest_csv)

    split_path = dataset_root / "splits.json"
    split_path.write_text(json.dumps(bundle["splits"], indent=2))

    meta = {
        "dataset_name": args.dataset_name,
        "raw_root": str(args.raw_root),
        "processed_root": str(dataset_root),
        "num_samples_kept": len(samples),
        "min_residues": args.min_residues,
        "split_source": bundle["splits"].get("split_source", "unknown"),
        "seed": args.seed,
        "notes": [
            "Only backbone atoms N, CA, C are retained.",
            "Per-sample output stores residue-level data; node-level duplication occurs in graph builder.",
        ],
    }
    meta.update(bundle["meta"])
    (dataset_root / "meta.json").write_text(json.dumps(meta, indent=2))

    logger.info("Preprocessing complete: %d samples", len(samples))
    logger.info("Manifest: %s", manifest_csv)
    logger.info("Splits: %s", split_path)


if __name__ == "__main__":
    main()
