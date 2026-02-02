from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Sequence

# Add parent directory to path to import Model package
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import pairwise_distances
from Model.model_config import SEED


def k_center_greedy_from_uncertain(
    X_unlabeled: np.ndarray,
    uncertain_idxs: Sequence[int],
    k: int,
    metric: str = "euclidean",
    random_state: int = SEED,
) -> List[int]:
    """
    Run k-center greedy on a restricted pool of indices (uncertain_idxs).
    Returns selected indices (w.r.t original X_unlabeled indexing).
    """
    idx_list = np.asarray(list(uncertain_idxs), dtype=int)

    if k <= 0:
        raise ValueError("k must be positive.")
    if len(idx_list) == 0:
        return []
    if k >= len(idx_list):
        return idx_list.tolist()

    X_uncertain = X_unlabeled[idx_list]
    m = len(idx_list)

    rng = np.random.default_rng(random_state)
    first_idx = int(rng.integers(0, m))

    selected_local = [first_idx]

    min_d = pairwise_distances(
        X_uncertain,
        X_uncertain[first_idx].reshape(1, -1),
        metric=metric,
    ).flatten()

    for _ in range(1, k):
        next_local = int(np.argmax(min_d))
        selected_local.append(next_local)

        d_new = pairwise_distances(
            X_uncertain,
            X_uncertain[next_local].reshape(1, -1),
            metric=metric,
        ).flatten()

        min_d = np.minimum(min_d, d_new)

    return idx_list[np.array(selected_local)].tolist()
