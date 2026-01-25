from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit

from .ensemble import EnsembleDict


def calculate_voting_entropy(ensemble_predictions: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Voting entropy per label.
    Returns array shape (n_samples, n_labels).
    """
    all_voting_entropy_for_label = []

    for key in list(ensemble_predictions.keys()):
        predictions_for_label = ensemble_predictions[key].astype(int)
        total_pr_count = predictions_for_label.shape[0]
        n_members = predictions_for_label.shape[1]

        positive_count = np.sum(predictions_for_label, axis=1)
        frac_pos = positive_count / n_members
        frac_neg = 1 - frac_pos

        ent = np.zeros(total_pr_count)
        mask = (frac_pos > 0) & (frac_pos < 1)

        ent[mask] = -(
            frac_pos[mask] * np.log2(frac_pos[mask])
            + frac_neg[mask] * np.log2(frac_neg[mask])
        )

        all_voting_entropy_for_label.append(ent.reshape(-1, 1))

    return np.concatenate(all_voting_entropy_for_label, axis=1)


def calculate_prediction_entropy(
    classifiers: EnsembleDict,
    unlabeled_X: pd.DataFrame,
    label_columns: Sequence[str],
) -> np.ndarray:
    """
    Prediction entropy based on mean predicted probability across ensemble members.
    Returns array shape (n_samples, n_labels).
    """
    all_label_ensemble_probabilities: Dict[str, np.ndarray] = {}

    for label_col in label_columns:
        probs_by_ensemble = []

        for bootstrap_idx in list(classifiers.keys()):
            for model_name in ["XGBoost", "LogisticRegression", "SVM"]:
                model = classifiers[bootstrap_idx][label_col][model_name]

                if model_name in ["XGBoost", "LogisticRegression"]:
                    # type: ignore[attr-defined]
                    y_probs = model.predict_proba(unlabeled_X)[:, 1]
                else:  # SVM
                    # type: ignore[attr-defined]
                    y_probs = expit(model.decision_function(unlabeled_X))

                probs_by_ensemble.append(np.asarray(y_probs).reshape(-1, 1))

        all_label_ensemble_probabilities[label_col] = np.concatenate(probs_by_ensemble, axis=1)

    all_prediction_entropy_for_label = []

    for key in label_columns:
        probs = all_label_ensemble_probabilities[key]
        avg_p = np.mean(probs, axis=1)

        ent = np.zeros(unlabeled_X.shape[0])
        mask = (avg_p > 0) & (avg_p < 1)
        p = avg_p[mask]
        ent[mask] = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

        all_prediction_entropy_for_label.append(ent.reshape(-1, 1))

    return np.concatenate(all_prediction_entropy_for_label, axis=1)


def combined_entropy_scores(
    voting_entropy: np.ndarray,
    prediction_entropy: np.ndarray,
) -> np.ndarray:
    return (voting_entropy + prediction_entropy) / 2.0


def pr_uncertainty_score(
    combined_entropy: np.ndarray,
    top_k_labels_for_score: int = 3,
) -> np.ndarray:
    """
    Notebook logic: sum of the top-k label entropies per PR.
    """
    if top_k_labels_for_score <= 0:
        raise ValueError("top_k_labels_for_score must be positive.")
    if top_k_labels_for_score > combined_entropy.shape[1]:
        raise ValueError(
            f"top_k_labels_for_score={top_k_labels_for_score} exceeds #labels={combined_entropy.shape[1]}"
        )
    return np.partition(combined_entropy, -top_k_labels_for_score, axis=1)[:, -top_k_labels_for_score:].sum(axis=1)
