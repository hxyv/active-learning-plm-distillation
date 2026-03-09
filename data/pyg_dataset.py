"""PyTorch Geometric dataset for DISPEF distillation graphs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from torch.utils.data import Dataset

from data.graph_builder import build_graph_data


class DistillationGraphDataset(Dataset):
    """
    One chain/protein per graph sample.

    Splits are loaded from processed `<dataset_root>/splits.json`.
    """

    def __init__(
        self,
        processed_root: Path,
        dataset_name: str,
        split_name: str,
        teacher_root: Optional[Path],
        cutoff: float,
        max_neighbors: int,
        cache_graphs: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_root = processed_root / dataset_name
        self.proteins_dir = self.dataset_root / "proteins"
        self.teacher_root = teacher_root
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.cache_graphs = cache_graphs
        self._cache: Dict[str, object] = {}

        split_path = self.dataset_root / "splits.json"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")

        splits = json.loads(split_path.read_text())
        if split_name not in splits:
            valid = ", ".join(sorted(splits.keys()))
            raise KeyError(f"Split '{split_name}' not found in {split_path}. Valid keys: {valid}")

        self.sample_ids: List[str] = sorted(list(splits[split_name]))
        if not self.sample_ids:
            raise RuntimeError(f"Split '{split_name}' is empty for dataset {self.dataset_root}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _sample_paths(self, sample_id: str):
        sample_path = self.proteins_dir / f"{sample_id}.npz"
        teacher_path = None
        if self.teacher_root is not None:
            teacher_path = self.teacher_root / f"{sample_id}.npz"
        return sample_path, teacher_path

    def __getitem__(self, idx: int):
        sample_id = self.sample_ids[idx]
        if self.cache_graphs and sample_id in self._cache:
            return self._cache[sample_id]

        sample_path, teacher_path = self._sample_paths(sample_id)
        if not sample_path.exists():
            raise FileNotFoundError(f"Missing preprocessed sample: {sample_path}")

        data = build_graph_data(
            sample_path=sample_path,
            teacher_path=teacher_path,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            center_coords=True,
        )

        if self.cache_graphs:
            self._cache[sample_id] = data
        return data


class SplitIndex:
    """Utility for future active-learning split updates without changing training code."""

    def __init__(self, split_file: Path) -> None:
        self.split_file = split_file
        self.splits = json.loads(split_file.read_text())

    def save(self) -> None:
        self.split_file.write_text(json.dumps(self.splits, indent=2))

    def move_ids(self, ids: List[str], src: str, dst: str) -> None:
        src_set = set(self.splits.get(src, []))
        dst_set = set(self.splits.get(dst, []))
        for sid in ids:
            if sid in src_set:
                src_set.remove(sid)
                dst_set.add(sid)
        self.splits[src] = sorted(src_set)
        self.splits[dst] = sorted(dst_set)
