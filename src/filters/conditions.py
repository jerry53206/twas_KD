import numpy as np
import pandas as pd

def golden_cross_in_window(K: pd.Series, D: pd.Series, window: int = 3,
                           zone_low: float = 40.0, zone_high: float = 80.0,
                           require_both_in_zone: bool = True):
    """
    檢查最後 window 根內（含當日）是否出現 KD 黃金交叉：
      - 交叉定義：K_{t-1} <= D_{t-1} 且 K_t > D_t
      - 若 require_both_in_zone 為 True，交叉當日 K、D 皆需落在 [zone_low, zone_high]
        否則只要求 K 在區間內
    回傳：交叉發生日的整數索引（相對於 K/D 的位置），或 None
    """
    n = len(K)
    if n < 2:
        return None
    start = max(1, n - window)
    for i in range(start, n):
        k_prev, d_prev = K.iloc[i-1], D.iloc[i-1]
        k_curr, d_curr = K.iloc[i],   D.iloc[i]
        if np.isnan([k_prev, d_prev, k_curr, d_curr]).any():
            continue
        crossed = (k_prev <= d_prev) and (k_curr > d_curr)
        if not crossed:
            continue
        if require_both_in_zone:
            in_zone = (zone_low <= k_curr <= zone_high) and (zone_low <= d_curr <= zone_high)
        else:
            in_zone = (zone_low <= k_curr <= zone_high)
        if in_zone:
            return i
    return None

def golden_cross_recent_no_zone(K: pd.Series, D: pd.Series, window: int = 3):
    """
    不檢查區間的 KD 黃金交叉偵測：
    回傳最近 window 根內（含當日）第一次出現黃金交叉的索引；沒有則回傳 None。
    """
    n = len(K)
    if n < 2:
        return None
    start = max(1, n - window)
    for i in range(start, n):
        k_prev, d_prev = K.iloc[i-1], D.iloc[i-1]
        k_curr, d_curr = K.iloc[i],   D.iloc[i]
        if np.isnan([k_prev, d_prev, k_curr, d_curr]).any():
            continue
        crossed = (k_prev <= d_prev) and (k_curr > d_curr)
        if crossed:
            return i
    return None

def volume_spike(volume: pd.Series, lookback: int = 20, multiplier: float = 1.5):
    """
    今日量 / 過去 lookback 日平均量 >= multiplier -> True
    回傳 (bool_ok, ratio)
    """
    if len(volume) < lookback + 1:
        return False, np.nan
    v_today = volume.iloc[-1]
    v_ref = volume.iloc[-(lookback+1):-1].mean()
    if v_ref == 0 or np.isnan(v_today) or np.isnan(v_ref):
        return False, np.nan
    ratio = float(v_today) / float(v_ref)
    return (ratio >= multiplier), ratio
