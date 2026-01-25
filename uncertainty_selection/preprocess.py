from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def drop_columns_safe(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Drop columns that exist; ignore missing (more robust than notebook)."""
    existing = [c for c in cols if c in df.columns]
    return df.drop(columns=existing)


def split_features_labels(
    labeled_train: pd.DataFrame,
    # labeled_test: pd.DataFrame,
    unlabeled: pd.DataFrame,
    label_columns: Sequence[str],
    drop_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      X_train, y_train, X_test, y_test, X_unlabeled

    Notes:
      - We drop configured columns from all sets.
      - Feature columns are inferred from labeled_train after dropping.
    """
    lt = labeled_train.copy()
    # lte = labeled_test.copy()
    ul = unlabeled.copy()

    lt = drop_columns_safe(lt, drop_columns)
    # lte = drop_columns_safe(lte, drop_columns)
    ul = drop_columns_safe(ul, drop_columns)

    missing_labels = [c for c in label_columns if c not in lt.columns]
    if missing_labels:
        raise ValueError(f"labeled_train is missing label columns: {missing_labels}")

    feature_columns = [c for c in lt.columns if c not in label_columns]

    X_train = lt[feature_columns].copy()
    y_train = lt[list(label_columns)].copy()

    # X_test = lte[feature_columns].copy() if all(c in lte.columns for c in feature_columns) else lte.drop(columns=list(label_columns), errors="ignore")
    # y_test = lte[list(label_columns)].copy() if all(c in lte.columns for c in label_columns) else pd.DataFrame()

    X_unlabeled = ul[feature_columns].copy() if all(c in ul.columns for c in feature_columns) else ul.copy()

    # return X_train, y_train, X_test, y_test, X_unlabeled
    return X_train, y_train, X_unlabeled


def log1p_and_standardize_numeric(
    X_train: pd.DataFrame,
    # X_test: pd.DataFrame,
    X_unlabeled: pd.DataFrame,
    numeric_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Applies log1p transform then standard scaling to numeric columns (fit on train).

    This replicates the notebook behavior but adds guards for missing columns.
    """
    X_train = X_train.copy()
    # X_test = X_test.copy()
    X_unlabeled = X_unlabeled.copy()

    num_cols = [c for c in numeric_columns if c in X_train.columns]
    if not num_cols:
        raise ValueError("No numeric columns found in X_train; check numeric_columns config.")

    # log1p
    for df in (X_train, X_unlabeled):
        cols = [c for c in num_cols if c in df.columns]
        df[cols] = np.log1p(df[cols].astype(float))

    scaler = StandardScaler().set_output(transform="pandas").fit(X_train[num_cols])

    X_train[num_cols] = scaler.transform(X_train[num_cols])
    # if len(X_test) > 0 and all(c in X_test.columns for c in num_cols):
    #     X_test[num_cols] = scaler.transform(X_test[num_cols])
    if len(X_unlabeled) > 0 and all(c in X_unlabeled.columns for c in num_cols):
        X_unlabeled[num_cols] = scaler.transform(X_unlabeled[num_cols])

    # return X_train, X_test, X_unlabeled, scaler
    return X_train, X_unlabeled, scaler
