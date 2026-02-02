from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def resolve_dir(data_root: Path, loop_number: int) -> Path:
    return data_root / f"loop_{loop_number}_data"


def load_labeled_train_data(loop_dir: Path) -> pd.DataFrame:
    """
    Load labeled train data CSV for a given loop.

    Expects files:
      - labeled_train_data.csv
    """
    train_path = loop_dir / f"labeled_train_data.csv"
   

    if not train_path.exists():
        raise FileNotFoundError(f"Missing labeled train CSV: {train_path}")

    labeled_train = pd.read_csv(train_path)

    return labeled_train

def load_unlabeled_data(loop_dir: Path) -> pd.DataFrame:
    """
    Load unlabeled data CSV for a given loop.

    Expects files:
      - unlabeled_data.csv
    """

    unlabeled_path = loop_dir / f"unlabeled_data.csv"

    if not unlabeled_path.exists():
        raise FileNotFoundError(f"Missing unlabeled CSV: {unlabeled_path}")

    unlabeled = pd.read_csv(unlabeled_path)

    return unlabeled

def load_labeled_test_data(loop_dir: Path) -> pd.DataFrame:
    """
    Load labeled test data CSV for a given loop.

    Expects files:
      - labeled_test_data.csv
    """

    labeled_test_path = loop_dir / f"labeled_test_data.csv"

    if not labeled_test_path.exists():
        raise FileNotFoundError(f"Missing unlabeled CSV: {labeled_test_path}")

    labeled_test = pd.read_csv(labeled_test_path)

    return labeled_test
