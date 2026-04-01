"""Pool manager for active learning split state."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ALPoolManager:
    """Manages per-round pool/train split state for active learning.

    Splits files are stored under ``al_output_dir/splits/`` so the original
    ``splits.json`` in the processed-data directory is never modified.

    State is persisted to ``al_output_dir/al_state.json`` after every mutation
    so a job can be safely resumed after interruption.

    Round numbering starts at 0 (initial labeled set, no acquisition yet).
    Round 1 is the first acquisition round, etc.
    """

    STATE_FILE = "al_state.json"
    SPLITS_DIR = "splits"

    def __init__(
        self,
        processed_root: Path,
        dataset_name: str,
        al_output_dir: Path,
        pool_size: int,
        initial_train_size: int,
        budget_per_round: int,
        seed: int = 42,
    ) -> None:
        self.processed_root = Path(processed_root)
        self.dataset_name = dataset_name
        self.al_output_dir = Path(al_output_dir)
        self.pool_size = pool_size
        self.initial_train_size = initial_train_size
        self.budget_per_round = budget_per_round
        self.seed = seed

        self.splits_dir = self.al_output_dir / self.SPLITS_DIR
        self.state_file = self.al_output_dir / self.STATE_FILE

        self._state: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def initialized(self) -> bool:
        return self.state_file.exists()

    @property
    def current_round(self) -> int:
        return self._ensure_state()["round"]

    def get_train_ids(self) -> List[str]:
        return list(self._ensure_state()["train_ids"])

    def get_pool_ids(self) -> List[str]:
        return list(self._ensure_state()["pool_ids"])

    def get_val_ids(self) -> List[str]:
        return list(self._ensure_state()["val_ids"])

    def get_test_ids(self) -> List[str]:
        return list(self._ensure_state()["test_ids"])

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create the round-0 split from the original splits.json.

        Takes the first ``pool_size`` training IDs as the candidate pool,
        randomly assigns ``initial_train_size`` to train, and keeps the rest
        as the unlabeled pool.  Val/test are passed through unchanged.
        """
        if self.initialized:
            logger.warning("ALPoolManager already initialized; skipping. Use resume=True to continue.")
            return

        original_splits_path = self.processed_root / self.dataset_name / "splits.json"
        if not original_splits_path.exists():
            raise FileNotFoundError(f"Original splits.json not found: {original_splits_path}")

        original_splits = json.loads(original_splits_path.read_text())
        all_train_ids = sorted(original_splits.get("train", []))

        if len(all_train_ids) < self.pool_size:
            raise ValueError(
                f"pool_size={self.pool_size} exceeds available training IDs ({len(all_train_ids)})"
            )
        if self.initial_train_size >= self.pool_size:
            raise ValueError(
                f"initial_train_size={self.initial_train_size} must be < pool_size={self.pool_size}"
            )

        # Use the first pool_size IDs as the candidate pool (deterministic)
        pool_candidates = all_train_ids[: self.pool_size]

        rng = np.random.default_rng(self.seed)
        chosen_indices = rng.choice(len(pool_candidates), size=self.initial_train_size, replace=False)
        chosen_indices_set = set(chosen_indices.tolist())

        initial_train = sorted([pool_candidates[i] for i in chosen_indices_set])
        initial_pool = sorted([pool_candidates[i] for i in range(len(pool_candidates)) if i not in chosen_indices_set])

        val_ids = sorted(original_splits.get("val", []))
        test_ids = sorted(original_splits.get("test", []))

        state = {
            "round": 0,
            "train_ids": initial_train,
            "pool_ids": initial_pool,
            "val_ids": val_ids,
            "test_ids": test_ids,
            "seed": self.seed,
            "pool_size": self.pool_size,
            "initial_train_size": self.initial_train_size,
            "budget_per_round": self.budget_per_round,
        }

        self.al_output_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self._state = state
        self._save_state()
        self._write_splits_file(0)

        logger.info(
            "ALPoolManager initialized: round=0, train=%d, pool=%d, val=%d, test=%d",
            len(initial_train), len(initial_pool), len(val_ids), len(test_ids),
        )

    # ------------------------------------------------------------------
    # Round advancement
    # ------------------------------------------------------------------

    def advance_round(self, selected_ids: List[str]) -> None:
        """Move ``selected_ids`` from pool → train and write the next splits file."""
        state = self._ensure_state()

        pool_set = set(state["pool_ids"])
        bad = [sid for sid in selected_ids if sid not in pool_set]
        if bad:
            raise ValueError(f"IDs not in pool: {bad[:5]} ...")

        for sid in selected_ids:
            pool_set.discard(sid)

        state["train_ids"] = sorted(state["train_ids"] + selected_ids)
        state["pool_ids"] = sorted(pool_set)
        state["round"] = state["round"] + 1

        self._save_state()
        self._write_splits_file(state["round"])

        logger.info(
            "Advance to round=%d: added %d proteins, train=%d, pool=%d",
            state["round"], len(selected_ids), len(state["train_ids"]), len(state["pool_ids"]),
        )

    # ------------------------------------------------------------------
    # Splits file helpers
    # ------------------------------------------------------------------

    def get_current_splits_file(self, round_idx: Optional[int] = None) -> Path:
        if round_idx is None:
            round_idx = self.current_round
        return self.splits_dir / f"splits_round_{round_idx:02d}.json"

    def _write_splits_file(self, round_idx: int) -> None:
        state = self._ensure_state()
        splits = {
            "train": state["train_ids"],
            "val": state["val_ids"],
            "test": state["test_ids"],
            # pool_unassigned kept for dataset compatibility (dataset won't load it during AL)
            "pool_unassigned": state["pool_ids"],
        }
        path = self.get_current_splits_file(round_idx)
        path.write_text(json.dumps(splits, indent=2))
        logger.debug("Wrote splits file: %s", path)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _ensure_state(self) -> Dict:
        if self._state is not None:
            return self._state
        if not self.state_file.exists():
            raise RuntimeError(
                f"ALPoolManager not initialized. Call initialize() first or check {self.state_file}"
            )
        self._state = json.loads(self.state_file.read_text())
        return self._state

    def _save_state(self) -> None:
        self.state_file.write_text(json.dumps(self._state, indent=2))
