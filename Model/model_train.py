import copy 
import pandas as pd
import numpy as np
import itertools

from Model.model_definition import MLP
from Model.model_config import (
    SEED, 
    DEVICE, 
    EPOCHS, 
    BATCH_SIZE, 
    OPTIMIZERS, 
    LABEL_THRESHOLD, 
    PARAM_GRID, 
)
from Model.model_utils import(
    set_seed,
    make_data_loaders,
    get_prediction_probs,
    calc_pos_class_weight,
    calculate_evaluation_metrics,
)

import torch.nn as nn
from sklearn.model_selection import KFold

from Model.scale_numeric_features import scale_and_log_transform
from typing import Tuple, List, Callable
from sklearn.preprocessing import StandardScaler
import torch

def train_final_MLP(X_train: pd.DataFrame, y_train: pd.DataFrame, 
              hidden_dims: List[int], dropout: float, 
              lr: float, weight_decay: float, batch_size: int, optimizer_definition: Callable, 
              epochs: int, device: torch.device) -> Tuple[MLP, StandardScaler]:

    print(f"\nFinal Model Training with HParams: LR={lr}, WeightDecay={weight_decay}, Dropout={dropout}, HiddenDims={hidden_dims}")

    # scale numeric data
    X_train_scaled, scaler = scale_and_log_transform(X_train, train_scaler=None)

    input_dim = X_train_scaled.shape[1]
    output_dim = y_train.shape[1]

    model = MLP(in_dim=input_dim, hidden_dim=hidden_dims, dropout=dropout, out_dim=output_dim).to(device)

    optimizer = optimizer_definition(model.parameters(), lr=lr, weight_decay=weight_decay)

    pos_weight = calc_pos_class_weight(y_train)

    # pos weight for handling class imabalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader, _ = make_data_loaders(X=X_train_scaled,y=y_train, batch_size=batch_size)
    
    for epoch in range(epochs):
        model.train()
            
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} completed.")
    return model, scaler


def train_mlp_with_cv(X: pd.DataFrame, y: pd.DataFrame, optimizer_choice: str, k: int = 5, **kwargs) -> Tuple[dict, float, dict, StandardScaler]:
    """
    Perform k-fold cross validation using train_mlp as helper function.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.DataFrame): Label matrix
        optimizer_choice (str): Optimizer to use ('adam' or 'adamw')
        k (int): Number of folds for cross validation
        **kwargs: Additional arguments passed to train_mlp
        
    Returns:
        tuple: (best_model_state, average_f1, best_hparams) where best is from the fold with highest F1
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    
    fold_f1_scores = []
    fold_hparams = []
    fold_model_states = []
    
    best_overall_f1 = -np.inf
    best_model_state = None
    best_hparams = None

    X_scaled, scaler = scale_and_log_transform(X, train_scaler=None)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/{k}")
        print(f"{'='*50}")
        
        X_train = X_scaled.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X_scaled.iloc[val_idx]
        y_val = y.iloc[val_idx]

        # Call tune_hyper_param for this fold
        model_state, f1_score, hparams = tune_hyper_param(
            X_train, y_train, X_val, y_val, optimizer_choice, k_i=fold+1, **kwargs
        )
        
        fold_f1_scores.append(f1_score)
        fold_hparams.append(hparams)
        fold_model_states.append(model_state)
        
        if f1_score > best_overall_f1:
            best_overall_f1 = f1_score
            best_model_state = model_state
            best_hparams = hparams
    
    average_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)
    
    print(f"\n{'='*50}")
    print("Cross Validation Results:")
    print(f"{'='*50}")
    print(f"Average F1-micro: {average_f1:.4f} ± {std_f1:.4f}")
    print(f"Best Fold F1-micro: {best_overall_f1:.4f}")
    print(f"Best Hyperparameters: {best_hparams}")
    print(f"Fold F1 scores: {fold_f1_scores}")
    
    return best_model_state, average_f1, best_hparams, scaler


def tune_hyper_param(X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame, 
                     optimizer_choice: str, k_i: int, 
                     optimizers: dict = OPTIMIZERS, param_grid: dict = PARAM_GRID, epochs: int = EPOCHS,
                     device: torch.device = DEVICE, batch_size: int = BATCH_SIZE, 
                     label_threshold: float = LABEL_THRESHOLD, seed: int = SEED) -> Tuple[dict, float, dict]:
    # set seed for any random initialization
    set_seed(seed=seed)

    # Dynamically adjust batch size to avoid batch size of 1 during k-fold CV
    # Batch norm requires batch size > 1 during training
    min_fold_size = min(len(X_train), len(X_val))
    adjusted_batch_size = max(1, min(batch_size, max(2, min_fold_size // 2)))

    # micro f1 is more statistically stable than macro f1
    # as micro pools all the TP,FP, FN together for calc --> increased sample size
    best_f1_micro = -np.inf
    best_hparams = None
    best_model_state = None

    for hparams in itertools.product(
        param_grid["lr"],
        param_grid["weight_decay"],
        param_grid["hidden_dims"],
        param_grid['dropout'],
    ):
        lr, weight_decay, hidden_dims, dropout = hparams

        print(f"\nTraining with HParams(k={k_i}): LR={lr}, WeightDecay={weight_decay}, Dropout={dropout}, HiddenDims={hidden_dims}")

        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]

        model = MLP(in_dim=input_dim, 
                      hidden_dim=hidden_dims, 
                      dropout=dropout, 
                      out_dim=output_dim).to(device)

        optimizer = optimizers[optimizer_choice](model.parameters(), lr=lr, weight_decay=weight_decay)

        pos_weight = calc_pos_class_weight(y_train)

        # pos weight for handling class imabalance
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_loader, _ = make_data_loaders(X=X_train,y=y_train, batch_size=adjusted_batch_size)
        val_loader, _ = make_data_loaders(X=X_val,y=y_val, batch_size=adjusted_batch_size)

        for epoch in range(epochs):
            model.train()

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            validation_probs, validation_labels = get_prediction_probs(model, val_loader)
            
            eval_metrics = calculate_evaluation_metrics(probs=validation_probs, y_true=validation_labels, threshold=label_threshold)

            val_accuracy, val_precision, val_recall, val_micro_f1 = eval_metrics.values()

            print(f"Epoch {epoch+1}/{epochs}, Val F1-micro: {val_micro_f1:.4f}  Val Recall: {val_recall:.4f}  Val Precision: {val_precision:.4f}  Val Acc: {val_accuracy:.4f}")

        
            if val_micro_f1 > best_f1_micro:
                best_f1_micro = val_micro_f1
            
                best_hparams = {
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "hidden_dims": hidden_dims,
                    "dropout": dropout,
                }
                best_model_state = copy.deepcopy(model.state_dict())

    print("\nHyperparameter tuning complete!")
    print(f"Best Hyperparameters: {best_hparams}")
    print(f"Best Validation F1-micro: {best_f1_micro:.4f}")

    return best_model_state, best_f1_micro, best_hparams


