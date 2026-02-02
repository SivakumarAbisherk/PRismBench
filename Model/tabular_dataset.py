import numpy as np
import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(self, X_df, y_df):
        # pytorch expects float32 for dnn modeling
        X = X_df.to_numpy(dtype=np.float32, copy=True)
        y = y_df.to_numpy(dtype=np.float32, copy=True)

        # loads data into self.X, self.y without copying: memory efficient
        # uses the same object created in the memory, just points to the location instead of creating another copy of X, y
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]