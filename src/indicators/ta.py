import pandas as pd
import numpy as np

def stochastic_kd(high: pd.Series, low: pd.Series, close: pd.Series,
                  n: int = 9, k_smooth: int = 3, d_period: int = 3):
    """
    Slow Stochastic KD (n,k_smooth,d_period).
    Returns (K, D) as pandas Series aligned to input index.
    """
    if len(close) < n + k_smooth + d_period:
        return None, None

    lowest_low = low.rolling(window=n, min_periods=n).min()
    highest_high = high.rolling(window=n, min_periods=n).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)

    k_raw = (close - lowest_low) / denom * 100.0
    K = k_raw.rolling(window=k_smooth, min_periods=k_smooth).mean()
    D = K.rolling(window=d_period, min_periods=d_period).mean()
    return K, D
