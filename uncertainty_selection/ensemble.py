from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier
except Exception as e:  # pragma: no cover
    XGBClassifier = None  # type: ignore


EnsembleDict = Dict[int, Dict[str, Dict[str, object]]]


def create_bootstrap_indices(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(low=0, high=n_samples, size=n_samples)


def create_bootstrap_samples(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_sets: int,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    bootstrap_sets_X: List[pd.DataFrame] = []
    bootstrap_sets_y: List[pd.DataFrame] = []

    for i in range(n_sets):
        idx = create_bootstrap_indices(X.shape[0], seed=i)
        bootstrap_sets_X.append(X.iloc[idx])
        bootstrap_sets_y.append(y.iloc[idx])

    return bootstrap_sets_X, bootstrap_sets_y


def create_bootstrap_classifiers(
    n_sets: int,
    label_columns: Sequence[str],
) -> EnsembleDict:
    if XGBClassifier is None:
        raise ImportError(
            "xgboost is not available. Install it (pip install xgboost) "
            "or remove XGBoost from the ensemble."
        )

    classifiers: EnsembleDict = {}

    for bootstrap_idx in range(n_sets):
        bootstrap_model_set: Dict[str, Dict[str, object]] = {}

        for label in label_columns:
            bootstrap_model_set[label] = {
                "XGBoost": XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                ),
                "LogisticRegression": LogisticRegression(
                    C=0.1,
                    penalty="l1",
                    solver="liblinear",
                    max_iter=1000,
                ),
                "SVM": LinearSVC(
                    C=0.001,
                    loss="squared_hinge",
                    dual=False,
                    max_iter=5000,
                ),
            }
        classifiers[bootstrap_idx] = bootstrap_model_set

    return classifiers


def train_all_models(
    classifiers: EnsembleDict,
    bootstrap_X: Sequence[pd.DataFrame],
    bootstrap_y: Sequence[pd.DataFrame],
    label_columns: Sequence[str],
    verbose: bool = True,
) -> EnsembleDict:
    
    trained_classifiers:EnsembleDict=classifiers

    for bootstrap_idx in list(trained_classifiers.keys()):
        X_train = bootstrap_X[int(bootstrap_idx)]
        y_train = bootstrap_y[int(bootstrap_idx)]

        for label_col in label_columns:
            y_binary = y_train[label_col]

            # imbalance ratio for this label
            pos = int(np.sum(y_binary == 1))
            neg = int(np.sum(y_binary == 0))
            scale_pos_weight = neg / max(pos, 1)

            for model_name in ["XGBoost", "LogisticRegression", "SVM"]:
                model = trained_classifiers[bootstrap_idx][label_col][model_name]

                if model_name == "XGBoost":
                    # type: ignore[attr-defined]
                    model.set_params(scale_pos_weight=scale_pos_weight)

                # type: ignore[call-arg]
                model.fit(X_train, y_binary)

                if verbose:
                    print(
                        f"Trained {model_name} | bootstrap={bootstrap_idx} | label={label_col} "
                        f"(pos={pos}, neg={neg}, scale_pos_weight={scale_pos_weight:.3f})"
                    )

    return trained_classifiers


def predict_hard_labels(
    classifiers: EnsembleDict,
    unlabeled_X: pd.DataFrame,
    label_columns: Sequence[str],
) -> Dict[str, np.ndarray]:
    """
    Returns dict[label] -> array shape (n_samples, n_ensemble_members) with 0/1 predictions.
    """
    all_label_ensemble_prediction: Dict[str, np.ndarray] = {}

    for label_col in label_columns:
        label_by_ensemble: List[np.ndarray] = []

        for bootstrap_idx in list(classifiers.keys()):
            for model_name in ["XGBoost", "LogisticRegression", "SVM"]:
                model = classifiers[bootstrap_idx][label_col][model_name]
                # type: ignore[call-arg]
                y_pred = model.predict(unlabeled_X)
                label_by_ensemble.append(np.asarray(y_pred).reshape(-1, 1))

        all_label_ensemble_prediction[label_col] = np.concatenate(label_by_ensemble, axis=1)

    return all_label_ensemble_prediction
