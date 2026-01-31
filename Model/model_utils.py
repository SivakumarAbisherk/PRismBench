import numpy as np
import pandas as pd

from model_config import DEVICE

import torch
from torch.utils.data import DataLoader
from tabular_dataset import TabularDataset

from sklearn.utils.class_weight import compute_class_weight

def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size: int):

    train_ds = TabularDataset(X_train, y_train)
    val_ds   = TabularDataset(X_val, y_val)
    test_ds  = TabularDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, len(train_ds)



@torch.no_grad()
def get_probs(model, loader):
    """
    Function get_probs computes the raw logits and compute & return the probabilities, truth labels as ndarrys .
    """
    # set model for inference mode: stops batch-norm and dropouts
    model.eval()
    probalities, y_source = [], []

    # process data in batches: xb, yb
    for xb, yb in loader:

        xb = xb.to(DEVICE)
        yb=yb.to(DEVICE)

        # raw output from model
        logits = model(xb)

        # sigmoid to convert logits to prob
        prob = torch.sigmoid(logits).cpu().numpy()

       
        probalities.append(prob)
        y_source.append(yb.cpu().numpy()) # Removed .ravel() for multi-label

    return np.concatenate(probalities, axis=0), np.concatenate(y_source, axis=0)


def calc_pos_class_weight(y:pd.DataFrame):
    """
    Function calc_pos_class_weight returns pos/neg class ratio.
    """

    n_labels = y.shape[1]
    pos_weights_list = []

    for i in range(n_labels):

        y_i = y_i.loc[:, i]
        # unique classes in the given y
        unique_classes=np.unique(y_i)

        if len(unique_classes)==1:
            pos_weight=1
        else:
            class_weights = compute_class_weight( 
                class_weight="balanced",
                classes=unique_classes,
                y=y_i
            ) 
            # class_weights --> [class_weight_for_class 0, class_weight_for_class 1]
            pos_weight = class_weights[1]

        pos_weights_list.append(pos_weight)

    return torch.tensor(pos_weights_list, dtype=torch.float32).to(DEVICE)


