from data_config import NUMERIC_COLS
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

def scale_and_log_transform(df:pd.DataFrame, train_scaler=None):

    # get numeric features
    df_numeric = df[NUMERIC_COLS]

    for col in NUMERIC_COLS:
        assert df_numeric[col].min() >= 0, "Negative values provided for log transform"

    # apply log transform
    df_log = pd.DataFrame()
    df_log[NUMERIC_COLS] = np.log1p(df_numeric[NUMERIC_COLS])

    
    # train scaler is not there (train_df)
    if not train_scaler:
        train_scaler = StandardScaler().set_output(transform="pandas").fit(df_log)

    scaled_df = pd.DataFrame(train_scaler.transform(df_log))

    for col in NUMERIC_COLS:
        df[col] = scaled_df[col]

    return df, train_scaler




