"""Acquisition functions for pool-based active learning.

Current implementations
-----------------------
random_acquisition  -- passive baseline (uniform random sampling)

Planned (not yet implemented)
------------------------------
mc_dropout_acquisition  -- MC Dropout uncertainty sampling on output MLP head
emc_acquisition         -- Expected Model Change via output-layer gradient norms
diversity_acquisition   -- Agglomerative hierarchical clustering on node embeddings
"""

from __future__ import annotations

from typing import List

import numpy as np


def random_acquisition(pool_ids: List[str], budget: int, rng: np.random.Generator) -> List[str]:
    """Select ``budget`` proteins uniformly at random from the pool (passive baseline).

    Args:
        pool_ids: IDs of all currently unlabeled proteins.
        budget:   Number of proteins to select.
        rng:      Seeded numpy Generator for reproducibility.

    Returns:
        Sorted list of selected protein IDs (length = min(budget, len(pool_ids))).
    """
    n = min(budget, len(pool_ids))
    indices = rng.choice(len(pool_ids), size=n, replace=False)
    return sorted([pool_ids[i] for i in indices])
