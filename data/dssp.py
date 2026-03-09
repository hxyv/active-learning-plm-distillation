"""DSSP utility functions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

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


def compute_dssp_from_bb4_coords(
    bb4_coords: np.ndarray,
    sequence: List[str],
) -> np.ndarray:
    """Compute 8-class DSSP labels from backbone N, CA, C, O coordinates.

    Uses mdtraj (version 1.10.1, as in the paper) with backbone-only topology.
    Coordinates must be in **nanometers** (mdtraj native unit).

    Args:
        bb4_coords: float32 array, shape (L, 4, 3).  Atom order: [N, CA, C, O].
                    Coordinates must be in nanometers.
        sequence:   List of L one-letter amino-acid codes (used for residue names).

    Returns:
        dssp_idx: int64 array, shape (L,).  Values 0-7 map to SS8_CLASSES
                  ["G","H","I","T","E","B","S","C"].
                  Returns all -100 if mdtraj is unavailable or L < 5.
    """
    L = int(bb4_coords.shape[0])
    fallback = np.full(L, -100, dtype=np.int64)
    if L < 5:
        return fallback

    try:
        import mdtraj as md
    except Exception as exc:
        logger.warning("mdtraj unavailable (%s); DSSP labels will be -100", exc)
        return fallback

    try:
        # Build a minimal backbone topology: N, CA, C, O per residue.
        topology = md.Topology()
        chain = topology.add_chain()
        coords_flat: List[np.ndarray] = []

        for res_i in range(L):
            residue = topology.add_residue("ALA", chain)
            for aname, coord in zip(["N", "CA", "C", "O"], bb4_coords[res_i]):
                topology.add_atom(
                    aname,
                    md.element.Element.getBySymbol(aname[0]),
                    residue,
                )
                coords_flat.append(coord)

        # xyz shape: (1, L*4, 3) in nm — mdtraj native unit.
        xyz = np.array(coords_flat, dtype=np.float32)[np.newaxis, :, :]
        traj = md.Trajectory(xyz, topology)

        # simplified=False → 8-class (G, H, I, T, E, B, S, ' ').
        dssp_chars = md.compute_dssp(traj, simplified=False)[0]  # (L,)

        # Map mdtraj character codes to SS8_CLASSES indices.
        # mdtraj uses ' ' for coil, 'NA' for terminal/missing.
        char_map = {
            "G": SS8_TO_INDEX["G"],
            "H": SS8_TO_INDEX["H"],
            "I": SS8_TO_INDEX["I"],
            "T": SS8_TO_INDEX["T"],
            "E": SS8_TO_INDEX["E"],
            "B": SS8_TO_INDEX["B"],
            "S": SS8_TO_INDEX["S"],
            " ": SS8_TO_INDEX["C"],
            "C": SS8_TO_INDEX["C"],
            "NA": SS8_TO_INDEX["C"],
        }
        return np.array(
            [char_map.get(str(c), SS8_TO_INDEX["C"]) for c in dssp_chars],
            dtype=np.int64,
        )

    except Exception as exc:
        logger.warning("mdtraj DSSP computation failed (%s); labels will be -100", exc)
        return fallback
