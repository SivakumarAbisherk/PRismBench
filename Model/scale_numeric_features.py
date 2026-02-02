from Model.data_config import NUMERIC_COLS
import numpy as np
import pandas as pd
from typing import Tuple, Optional

from sklearn.preprocessing import StandardScaler

def scale_and_log_transform(df: pd.DataFrame, train_scaler: Optional[StandardScaler] = None) -> Tuple[pd.DataFrame, StandardScaler]:

    # get numeric features
    df_numeric = df[NUMERIC_COLS]
    
    # train scaler is not there (train_df)
    if not train_scaler:
        train_scaler = StandardScaler().set_output(transform="pandas").fit(df_numeric)

    scaled_df = pd.DataFrame(train_scaler.transform(df_numeric))

    for col in NUMERIC_COLS:
        df[col] = scaled_df[col]

    return df, train_scaler




