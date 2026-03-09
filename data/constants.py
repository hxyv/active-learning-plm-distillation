"""Constants and mappings used across DISPEF preprocessing and training."""

from __future__ import annotations

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}

AA1_TO_INDEX = {
    aa: idx
    for idx, aa in enumerate(
        [
            "A",
            "R",
            "N",
            "D",
            "C",
            "Q",
            "E",
            "G",
            "H",
            "I",
            "L",
            "K",
            "M",
            "F",
            "P",
            "S",
            "T",
            "W",
            "Y",
            "V",
            "X",  # unknown
        ]
    )
}

INDEX_TO_AA1 = {v: k for k, v in AA1_TO_INDEX.items()}

# Paper baseline uses backbone atoms N, CA, C.
ATOM_TYPES = ["N", "CA", "C"]
ATOM_TYPE_TO_INDEX = {atom: idx for idx, atom in enumerate(ATOM_TYPES)}

# DSSP SS8 order used throughout the project.
# Order matches the helper in Schake v2: G, H, I, T, E, B, S, C.
SS8_CLASSES = ["G", "H", "I", "T", "E", "B", "S", "C"]
SS8_TO_INDEX = {ss: idx for idx, ss in enumerate(SS8_CLASSES)}
INDEX_TO_SS8 = {v: k for k, v in SS8_TO_INDEX.items()}


def aa3_to_index(aa3: str) -> int:
    aa1 = AA3_TO_1.get(aa3.upper(), "X")
    return AA1_TO_INDEX.get(aa1, AA1_TO_INDEX["X"])
