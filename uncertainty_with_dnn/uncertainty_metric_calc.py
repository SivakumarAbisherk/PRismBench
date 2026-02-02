from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# Add parent directory to path to import Model package
sys.path.insert(0, str(Path(__file__).parent.parent))

from Model.model_definition import MLP
from Model.tabular_dataset import TabularDataset
from torch.utils.data import DataLoader
from Model.model_utils import get_prediction_probs
from Model.model_config import BATCH_SIZE

def calculate_prediction_entropy(
    model: MLP ,
    unlabeled_X: pd.DataFrame,
    label_columns: Sequence[str],
) -> np.ndarray:
    """
    Prediction entropy based on mean predicted probability from the final model of each Active learning loop.
    Returns array shape (n_samples, n_labels).
    """
    # DataLoader expects label columns thus preparing dummy column
    dummy_y = pd.DataFrame(np.zeros((len(unlabeled_X), len(label_columns)), dtype=np.float32))

    dataset = TabularDataset(unlabeled_X, dummy_y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # probs shape = (n_samples, n_labels)
    probs, _ = get_prediction_probs(model, loader)
    

    # avoid 0 prob values
    eps = 1e-7
    probs = np.clip(probs, eps, 1.0 - eps)

    # entropy calculation
    entropy = -(
        probs * np.log(probs) +
        (1.0 - probs) * np.log(1.0 - probs)
    )
    
    # Replace NaN values (from 0*log(0) or 1*log(1)) with 0
    entropy = np.nan_to_num(entropy, nan=0.0)
    
    # Aggregate entropy across all labels to get single score per sample
    # Use mean to get average uncertainty across all labels
    mean_entropy = entropy.mean(axis=1)

    return mean_entropy

