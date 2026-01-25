from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SelectionConfig
from .data import (
    resolve_loop_dir, 
    load_loop_labeled_data,
    load_loop_unlabeled_data,
)
from .preprocess import split_features_labels, log1p_and_standardize_numeric
from .ensemble import (
    create_bootstrap_samples,
    create_bootstrap_classifiers,
    train_all_models,
    predict_hard_labels,
)
from .uncertainty import (
    calculate_voting_entropy,
    calculate_prediction_entropy,
    combined_entropy_scores,
    pr_uncertainty_score,
)
from .diversity import k_center_greedy_from_uncertain


def run_uncertainty_selection(
    ml_features_csv:str,
    loop_number: int = 1,
    project: str = "kafka",
    data_root: Path = Path("SamplingLoopData"),
    output_dir: Path = Path("UncertainPoints"),
    n_bootstrap: int = 20,
    top_uncertain: int = 100,
    k_diverse: int = 25,
    top_k_labels: int = 3,
    metric: str = "euclidean",
    random_state: int = 42,
    verbose: bool = False,
) -> pd.DataFrame | None:
    """
    Run uncertainty sampling with bootstrap ensemble, entropy, and k-center greedy diversity.
    
    Args:
        loop_number: Loop/iteration number
        project: Project name (e.g., "kafka")
        data_root: Root directory containing loop data
        output_dir: Directory to write output files
        n_bootstrap: Number of bootstrap samples
        top_uncertain: Number of top uncertain points to consider for diversification
        k_diverse: Number of diverse points to select
        top_k_labels: Top k labels for uncertainty score calculation
        metric: Distance metric for k-center greedy
        random_state: Random seed
        write_full_rows: If True, also output selected rows with all original columns
        verbose: Enable verbose logging
    
    Returns:
        Tuple of (selected_df, merged_df or None) containing uncertainty scores and optionally full rows
    """

    cfg = SelectionConfig(
        loop_number=loop_number,
        project_name=project,
        data_root=data_root,
        output_dir=output_dir,
        n_bootstrap_sets=n_bootstrap,
        top_uncertain_pool=top_uncertain,
        k_diverse=k_diverse,
        top_k_labels_for_score=top_k_labels,
        metric=metric,
        random_state=random_state,
    )

    loop_unlabeled_dir = resolve_loop_dir(cfg.data_root, cfg.loop_number-1)
    unlabeled = load_loop_unlabeled_data(loop_unlabeled_dir)
    # h=unlabeled.shape  

    # filtering only the prs with szz issues
    # read pr data with szz
    szz_issue_path = Path(__file__).parent.parent / ml_features_csv
    szz_origin_check = pd.read_csv(szz_issue_path)
    # prs with szz extracted from the szz data
    pr_with_szz = szz_origin_check.loc[~szz_origin_check["szz_origin_issues"].isna(),"pr_number"]

    # unlabeld prs with szz issue tickes linked
    unlabeled = unlabeled[unlabeled["pr_number"].isin(pr_with_szz)]

    # Load labeled data from all previous loops (exclude current loop)
    labeled_dfs = []
    for prev_loop in range(0, cfg.loop_number):
        prev_loop_dir = resolve_loop_dir(cfg.data_root, prev_loop)
        prev_labeled = load_loop_labeled_data(prev_loop_dir)
        labeled_dfs.append(prev_labeled)

    labeled_train = pd.concat(labeled_dfs, ignore_index=True)
    
    if "pr_number" not in unlabeled.columns:
        raise ValueError("Unlabeled CSV must contain 'pr_number' column (needed for output mapping).")

    # Split and preprocess
    X_train, y_train, X_unlabeled = split_features_labels(
        labeled_train=labeled_train,
        # labeled_test=labeled_test,
        unlabeled=unlabeled,
        label_columns=cfg.label_columns,
        drop_columns=cfg.drop_columns,
    )

    X_train, X_unlabeled, _scaler = log1p_and_standardize_numeric(
        X_train=X_train,
        # X_test=X_test,
        X_unlabeled=X_unlabeled,
        numeric_columns=cfg.numeric_columns,
    )

    # Bootstrap ensemble
    bootstrap_X, bootstrap_y = create_bootstrap_samples(X_train, y_train, cfg.n_bootstrap_sets)

    classifiers = create_bootstrap_classifiers(cfg.n_bootstrap_sets, cfg.label_columns)

    classifiers = train_all_models(
        classifiers=classifiers,
        bootstrap_X=bootstrap_X,
        bootstrap_y=bootstrap_y,
        label_columns=cfg.label_columns,
        verbose=verbose,
    )
    # Uncertainty scoring
    hard_preds = predict_hard_labels(classifiers, X_unlabeled, cfg.label_columns)
    v_entropy = calculate_voting_entropy(hard_preds)
    p_entropy = calculate_prediction_entropy(classifiers, X_unlabeled, cfg.label_columns)
    comb = combined_entropy_scores(v_entropy, p_entropy)
    pr_scores = pr_uncertainty_score(comb, cfg.top_k_labels_for_score)

    pr_uncertainty_df = pd.DataFrame(
        {"pr_number": unlabeled["pr_number"].values, "uncertainty_score": pr_scores}
    ).sort_values("uncertainty_score", ascending=False)

    # Pool then diversify
    pool_df = pr_uncertainty_df.head(cfg.top_uncertain_pool)
    pool_idxs = pool_df.index.to_numpy()

    selected_idxs = k_center_greedy_from_uncertain(
        X_unlabeled=X_unlabeled.to_numpy(),
        uncertain_idxs=pool_idxs,
        k=cfg.k_diverse,
        metric=cfg.metric,
        random_state=cfg.random_state,
    )
    
    selected_df = pr_uncertainty_df.loc[selected_idxs].sort_values("uncertainty_score", ascending=False)

    # Output
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_scores_path = cfg.output_dir / f"loop_{cfg.loop_number}_selected.csv"
    selected_df.to_csv(out_scores_path, index=False)

    # Prepare unlabeled data for next loop by omitting selected rows in this loop
    next_loop_unlabeled_df = unlabeled[~unlabeled["pr_number"].isin(selected_df["pr_number"])]
    # l=next_loop_unlabeled_df.shape
    
    # Write next loop unlabeled data
    current_loop_dir = cfg.data_root / f"loop_{cfg.loop_number}_data"
    current_loop_dir.mkdir(parents=True, exist_ok=True)
    current_loop_unlabeled_path = current_loop_dir / f"unlabeled_data.csv"

    # write the unlabeld data for next loop inside curretn folder
    next_loop_unlabeled_df.to_csv(current_loop_unlabeled_path, index=False)

    if verbose:
        print(f"Wrote: {out_scores_path}")
        print(f"Wrote: {current_loop_unlabeled_path}")
        # print(h, l)
        # print("BBBB:", labeled_train.shape)

    return selected_df

