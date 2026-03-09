"""I/O helpers for DISPEF preprocessing and split handling."""

from __future__ import annotations

import csv
import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_protein_id(raw_id: str) -> str:
    val = raw_id.strip().replace("\\", "/")
    val = val.split("/")[-1]
    val = val.replace(".pdb", "").replace(".cif", "").replace(".mmcif", "")
    val = re.sub(r"\s+", "", val)
    return val.upper()


def _read_id_list_text(path: Path) -> List[str]:
    ids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(normalize_protein_id(line.split()[0]))
    return ids


def _read_id_list_csv(path: Path) -> Tuple[Optional[Dict[str, List[str]]], List[str]]:
    """
    Returns either split dictionary or plain id list.
    Supported columns for split dictionary include: split + (id/protein/file/name).
    """
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None, []

    fieldnames = {name.lower(): name for name in (reader.fieldnames or [])}
    split_col = fieldnames.get("split") or fieldnames.get("set")
    id_col = (
        fieldnames.get("protein_id")
        or fieldnames.get("id")
        or fieldnames.get("name")
        or fieldnames.get("file")
        or fieldnames.get("pdb")
    )

    if split_col and id_col:
        out = {"train": [], "val": [], "test": []}
        for row in rows:
            split_raw = str(row.get(split_col, "")).strip().lower()
            split = "val" if split_raw in {"val", "valid", "validation", "dev"} else split_raw
            if split not in out:
                continue
            out[split].append(normalize_protein_id(str(row.get(id_col, ""))))
        if any(out.values()):
            return out, []

    if id_col:
        ids = [normalize_protein_id(str(row.get(id_col, ""))) for row in rows]
        return None, [x for x in ids if x]

    return None, []


def discover_split_ids(raw_root: Path) -> Dict[str, List[str]]:
    """
    Try to discover official DISPEF split files from the raw archive.
    If no split files are found, returns empty dict.
    """
    candidates = []
    patterns = [
        "*split*.json",
        "*split*.csv",
        "*split*.tsv",
        "*split*.txt",
        "*train*.txt",
        "*test*.txt",
        "*val*.txt",
        "*valid*.txt",
        "*train*.csv",
        "*test*.csv",
        "*val*.csv",
    ]

    for pattern in patterns:
        candidates.extend(raw_root.rglob(pattern))

    # Deterministic ordering for reproducibility.
    candidates = sorted({p.resolve() for p in candidates})
    if not candidates:
        return {}

    logger.info("Inspecting %d potential split files", len(candidates))

    split_dict = {"train": [], "val": [], "test": []}
    split_hits = 0

    for path in candidates:
        name = path.name.lower()
        try:
            if path.suffix.lower() == ".json":
                obj = json.loads(path.read_text())
                if isinstance(obj, dict):
                    local = {}
                    for key in ("train", "test", "val", "valid", "validation", "dev"):
                        if key in obj and isinstance(obj[key], list):
                            k = "val" if key in {"valid", "validation", "dev"} else key
                            local[k] = [normalize_protein_id(str(x)) for x in obj[key]]
                    if local:
                        split_hits += 1
                        for k, v in local.items():
                            split_dict[k].extend(v)
                        logger.info("Loaded split entries from JSON: %s", path)
                        continue

            if path.suffix.lower() in {".csv", ".tsv"}:
                found_split, ids = _read_id_list_csv(path)
                if found_split:
                    split_hits += 1
                    for k, v in found_split.items():
                        split_dict[k].extend(v)
                    logger.info("Loaded split entries from tabular file: %s", path)
                    continue
                if ids:
                    if "train" in name:
                        split_dict["train"].extend(ids)
                        split_hits += 1
                    elif "test" in name:
                        split_dict["test"].extend(ids)
                        split_hits += 1
                    elif any(x in name for x in ["val", "valid"]):
                        split_dict["val"].extend(ids)
                        split_hits += 1
                    if ids:
                        logger.info("Loaded %d IDs from %s", len(ids), path)
                    continue

            if path.suffix.lower() == ".txt":
                ids = _read_id_list_text(path)
                if ids:
                    if "train" in name:
                        split_dict["train"].extend(ids)
                        split_hits += 1
                    elif "test" in name:
                        split_dict["test"].extend(ids)
                        split_hits += 1
                    elif any(x in name for x in ["val", "valid"]):
                        split_dict["val"].extend(ids)
                        split_hits += 1
                    logger.info("Loaded %d IDs from %s", len(ids), path)
                    continue

        except Exception as exc:
            logger.warning("Skipping split candidate %s due to parse error: %s", path, exc)

    if split_hits == 0:
        return {}

    for key in split_dict:
        split_dict[key] = sorted(set(x for x in split_dict[key] if x))
    return split_dict


def build_split_from_train_only(
    train_ids: Iterable[str],
    val_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    train_ids = sorted(set(train_ids))
    rng = random.Random(seed)
    rng.shuffle(train_ids)
    n_val = max(1, int(len(train_ids) * val_fraction))
    val_ids = sorted(train_ids[:n_val])
    train_ids = sorted(train_ids[n_val:])
    return train_ids, val_ids


def map_split_ids_to_samples(
    split_ids: Dict[str, List[str]],
    sample_ids: Iterable[str],
) -> Dict[str, List[str]]:
    sample_ids = sorted(set(sample_ids))
    id_to_samples: Dict[str, Set[str]] = {}

    for sid in sample_ids:
        norm = normalize_protein_id(sid)
        base = normalize_protein_id(sid.split("_")[0])
        id_to_samples.setdefault(norm, set()).add(sid)
        id_to_samples.setdefault(base, set()).add(sid)

    out = {"train": [], "val": [], "test": []}
    used = set()

    for split_name in ["train", "val", "test"]:
        for raw in split_ids.get(split_name, []):
            norm = normalize_protein_id(raw)
            matches = id_to_samples.get(norm, set())
            for sid in matches:
                out[split_name].append(sid)
                used.add(sid)

    # remove duplicates while preserving deterministic order
    for k in out:
        out[k] = sorted(set(out[k]))

    out["unassigned"] = sorted(set(sample_ids) - used)
    return out
