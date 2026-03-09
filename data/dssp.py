"""DSSP utility functions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

from data.constants import SS8_TO_INDEX

logger = logging.getLogger(__name__)


def compute_dssp_labels(structure, structure_path: Path) -> Dict[Tuple[str, int, str], int]:
    """
    Compute DSSP labels for a Biopython structure.

    Returns mapping keyed by (chain_id, residue_number, insertion_code) -> SS8 index.
    If DSSP is unavailable, returns an empty dict.
    """
    try:
        from Bio.PDB.DSSP import DSSP
    except Exception as exc:
        logger.warning("Bio.PDB.DSSP import failed (%s); skipping DSSP labels", exc)
        return {}

    try:
        model = structure[0]
        dssp = DSSP(model, str(structure_path))
    except Exception as exc:
        logger.warning(
            "DSSP computation failed for %s (%s). "
            "Install `dssp`/`mkdssp` or disable DSSP in config.",
            structure_path,
            exc,
        )
        return {}

    mapping: Dict[Tuple[str, int, str], int] = {}
    for key, value in dssp.property_dict.items():
        # key: (chain_id, (' ', resseq, icode))
        chain_id = key[0]
        hetflag, resseq, icode = key[1]
        if hetflag != " ":
            continue
        ss = value[1] if len(value) > 1 else "C"
        ss = "C" if ss in {"-", " "} else ss
        ss = ss if ss in SS8_TO_INDEX else "C"
        mapping[(chain_id, int(resseq), str(icode).strip() or " ")] = SS8_TO_INDEX[ss]
    return mapping
