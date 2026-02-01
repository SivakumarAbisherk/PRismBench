from data_config import NUMERIC_COLS
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

def apply_log_transform(df:pd.DataFrame):

    # drop pr_number from the training features
    df = df.drop("pr_number", axis=1)

    df = df.drop(NUMERIC_COLS, axis=1)

    for col in NUMERIC_COLS:
        assert df[col].min() >= 0, "Negative values provided for log transform"

    # apply log transform
    df[NUMERIC_COLS] = np.log1p(df[NUMERIC_COLS])

    return df

def scale_log_transformed(df:pd.DataFrame, train_scaler=None):
    # train scaler is not there (train_df)
    if not train_scaler:
        train_scaler = StandardScaler().set_output(transform="pandas").fit(df)

    scaled_df = pd.DataFrame(train_scaler.transform(df))

    return df, train_scaler




