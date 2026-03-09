"""Abstract teacher interface for per-residue SS8 probability prediction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class TeacherBase(ABC):
    @abstractmethod
    def predict_ss8_probs(self, sequence: str, sample_id: Optional[str] = None) -> np.ndarray:
        """Return [L, 8] float32 probabilities for a sequence of length L."""


class FileTeacher(TeacherBase):
    """Loads precomputed per-residue SS8 probabilities from disk."""

    def __init__(self, probs_root):
        from pathlib import Path

        self.probs_root = Path(probs_root)

    def predict_ss8_probs(self, sequence: str, sample_id: Optional[str] = None) -> np.ndarray:
        if sample_id is None:
            raise ValueError("FileTeacher requires sample_id to locate cached probabilities")

        import numpy as np

        path = self.probs_root / f"{sample_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing precomputed teacher file: {path}")
        with np.load(path, allow_pickle=True) as arr:
            if "teacher_probs_residue" in arr:
                probs = arr["teacher_probs_residue"]
            elif "probs" in arr:
                probs = arr["probs"]
            else:
                raise KeyError(f"Expected teacher_probs_residue/probs in {path}")

        if probs.shape[0] != len(sequence):
            raise ValueError(
                f"Teacher length mismatch for {sample_id}: sequence={len(sequence)}, probs={probs.shape}"
            )
        if probs.shape[1] != 8:
            raise ValueError(f"Teacher probabilities must have 8 classes; got shape {probs.shape}")

        probs = probs.astype(np.float32)
        probs = probs / np.clip(probs.sum(axis=-1, keepdims=True), 1e-8, None)
        return probs
