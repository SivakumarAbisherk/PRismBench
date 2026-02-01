import pandas as pd
import numpy as np
from model_train import train_mlp_with_cv

from pathlib import Path

file_path = Path("D:\\FYP\\DataMultiLabeling\\Model")

train_df = pd.read_csv(file_path / "train_set.csv")
val_df = pd.read_csv(file_path / "val_set.csv")

df = pd.concat([train_df, val_df], axis=0)

label_col = ["bug", "security", "performance" ,"code_quality_or_maintenability"]
y=df[label_col]
X= df.drop(columns=label_col)

train_mlp_with_cv(X, y, "adam")