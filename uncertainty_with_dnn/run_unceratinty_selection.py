from __future__ import annotations

import os
import sys
from pathlib import Path

# Add parent directory to path to import Model package
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch

from .prepare_data import (
    resolve_dir, 
    load_labeled_train_data,
    load_unlabeled_data,
    load_labeled_test_data
)
from .uncertainty_metric_calc import calculate_prediction_entropy
from .k_center_greedy import k_center_greedy_from_uncertain
from Model.model_train import train_final_MLP, train_mlp_with_cv, calculate_evaluation_metrics
from Model.data_config import LABEL_COLS
from Model.model_config import DEVICE, BATCH_SIZE, OPTIMIZERS, EPOCHS, LABEL_THRESHOLD, SEED
from Model.model_utils import get_prediction_probs, make_data_loaders
from Model.scale_numeric_features import scale_and_transform

from .uncertainty_metric_calc import calculate_prediction_entropy


def run_uncertainty_selection(
    ml_features_csv:str,
    loop_number: int = 1,
    data_root: Path = Path("SamplingLoopData"),
    output_dir: Path = Path("UncertainPoints"),
    model_monitor_dir: Path = Path("ModelMonitoring"),
    n_top_uncertain: int = 100,
    k_diverse: int = 25,
    metric: str = "euclidean",
    verbose: bool = False,
) -> pd.DataFrame | None:
    """
    Run uncertainty sampling with bootstrap ensemble, entropy, and k-center greedy diversity.
    
    Args:
        loop_number: Loop/iteration number
        data_root: Root directory containing loop data
        output_dir: Directory to write output files
        n_bootstrap: Number of bootstrap samples
        n_top_uncertain: Number of top uncertain points to consider for diversification
        k_diverse: Number of diverse points to select
        metric: Distance metric for k-center greedy
        write_full_rows: If True, also output selected rows with all original columns
        verbose: Enable verbose logging
    Returns:
        Tuple of (selected_df, merged_df or None) containing uncertainty scores and optionally full rows
    """
########################## Load unlabeled train data ################################################################
    loop_unlabeled_dir = resolve_dir(data_root, loop_number-1)
    unlabeled = load_unlabeled_data(loop_unlabeled_dir)

    # filtering only the prs with szz issues
    # read pr data with szz
    szz_issue_path = Path(__file__).parent.parent / ml_features_csv
    szz_origin_check = pd.read_csv(szz_issue_path)
    # prs with szz extracted from the szz data
    prs_with_szz = szz_origin_check.loc[~szz_origin_check["szz_origin_issues"].isna(),"pr_number"]

    # unlabeld prs with szz issue tickes linked
    unlabeled = unlabeled[unlabeled["pr_number"].isin(prs_with_szz)]

########################## Load labeled train data ###################################################################
    # Load labeled data from all previous loops (exclude current loop)
    labeled_dfs = []
    for prev_loop in range(0, loop_number):
        prev_loop_dir = resolve_dir(data_root, prev_loop)
        prev_labeled = load_labeled_train_data(prev_loop_dir)
        labeled_dfs.append(prev_labeled)

    labeled_train = pd.concat(labeled_dfs, ignore_index=True)

########################## Load labeled test data ####################################################################

    labeled_test_dir = resolve_dir(data_root, 0)
    labeled_test = load_labeled_test_data(labeled_test_dir)

#####################################################################################################################    

    if "pr_number" not in unlabeled.columns:
        raise ValueError("Unlabeled CSV must contain 'pr_number' column (needed for output mapping).")
    
    # X_unlabeled = unlabeled.drop(["pr_number"], axis=1)
    X_unlabeled = unlabeled.copy() 
    
########################## Train MLP with K-fold CV to pick best hparams #############################################
    # labeled_train_copy = labeled_train.copy()
    
    y=labeled_train[LABEL_COLS]
    X= labeled_train.drop(columns=LABEL_COLS)

    if "pr_number" in X.columns:
        X = X.drop(["pr_number"], axis=1)

    # labeled_train is scaled inside training function & "pr_number" is dropped inside
    _, average_f1, best_hparams, _ = train_mlp_with_cv(X, y, "adam")

########################## Train final MLP with best hparams #########################################################

    final_model, t_scaler = train_final_MLP(X_train=X, y_train=y,
                                          hidden_dims=best_hparams["hidden_dims"], dropout=best_hparams["dropout"],
                                          lr=best_hparams["lr"], weight_decay=best_hparams["weight_decay"],
                                          batch_size=BATCH_SIZE, optimizer_definition=OPTIMIZERS["adam"],
                                          epochs=EPOCHS, device=DEVICE)
    
########################## Store the final trained Model in each loop ################################################
    
    model_store_path = model_monitor_dir / f"model_store" / f"final_model_{loop_number}.pt"

    torch.save(final_model.state_dict(), model_store_path) 

########################## Test final MLP on Golden Seed test set #####################################################
    # scaling is done outside the training function
    labeled_test, t_scaler = scale_and_transform(labeled_test, t_scaler)

    y_test = labeled_test[LABEL_COLS]
    # X_test = labeled_test.drop(columns=LABEL_COLS+["pr_number"])
    X_test = labeled_test.drop(columns=LABEL_COLS)

    if "pr_number" in X_test.columns:
        X_test = X_test.drop(["pr_number"], axis=1)

    test_loader, _ = make_data_loaders(X_test, y_test, BATCH_SIZE)
    test_prob, test_label = get_prediction_probs(final_model, test_loader)

    test_eval_metric = calculate_evaluation_metrics(test_prob, test_label, LABEL_THRESHOLD)

########################## Visualize evaluation results of testing of final MLP #######################################

    eval_metric_csv = model_monitor_dir / f"eval_metrics" / f"final_model_evaluation.csv"
    eval_metric_row = {
        "loop_number": loop_number,
        "accuracy": test_eval_metric["accuracy"],
        "precision": test_eval_metric["precision"],
        "recall": test_eval_metric["recall"],
        "micro_f1": test_eval_metric["micro_f1"],
    }
    pd.DataFrame([eval_metric_row]).to_csv(
        eval_metric_csv, 
        mode="a", 
        header=not os.path.exists(eval_metric_csv), 
        index=False
    )

########################## Calculate prediction entropy of Unlabeled Set using final MLP ##############################

    entropy_values = calculate_prediction_entropy(final_model, X_unlabeled, LABEL_COLS)

    assert len(unlabeled) == len(entropy_values), "Unlabeled Data Frame and entropy value list has different lengths."

    pr_entropy_df = pd.DataFrame(
        {"pr_number": unlabeled["pr_number"].values, "uncertainty_score": entropy_values}
    ).sort_values("uncertainty_score", ascending=False)

    # Pool then diversify
    pool_df = pr_entropy_df.head(n_top_uncertain)
    pool_idxs = pool_df.index.to_numpy()

########################## Apply k-center greedy algorithm to pick diverse points #####################################

    selected_idxs = k_center_greedy_from_uncertain(X_unlabeled=X_unlabeled.values,
                                                   uncertain_idxs=pool_idxs,
                                                   metric=metric,
                                                   k=k_diverse)

    selected_pr_df = pr_entropy_df.loc[selected_idxs].sort_values("uncertainty_score", ascending=False)

    # Output
    output_dir.mkdir(parents=True, exist_ok=True)
    out_scores_path = output_dir / f"loop_{loop_number}_selected.csv"
    selected_pr_df.to_csv(out_scores_path, index=False)

########################## Prepare next loop unlabeled data ###########################################################

    # Prepare unlabeled data for next loop by omitting selected rows in this loop
    next_loop_unlabeled_df = unlabeled[~unlabeled["pr_number"].isin(selected_pr_df["pr_number"])]
    # l=next_loop_unlabeled_df.shape
    
    # Write next loop unlabeled data
    current_loop_dir = data_root / f"loop_{loop_number}_data"
    current_loop_dir.mkdir(parents=True, exist_ok=True)
    current_loop_unlabeled_path = current_loop_dir / f"unlabeled_data.csv"

    # write the unlabeld data for next loop inside curretn folder
    next_loop_unlabeled_df.to_csv(current_loop_unlabeled_path, index=False)

    if verbose:
        print(f"Wrote: {out_scores_path}")
        print(f"Wrote: {current_loop_unlabeled_path}")

    print(f"Lenth of unlabled df: {len(unlabeled)}")
    print(f"Lenth of next unlabeled df: {len(next_loop_unlabeled_df)}")

    return selected_pr_df
