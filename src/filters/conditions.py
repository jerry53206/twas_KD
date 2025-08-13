import numpy as np
import pandas as pd


def golden_cross_in_window(
    K: pd.Series,
    D: pd.Series,
    window: int = 3,
    zone_low: float = 40.0,
    zone_high: float = 55.0,
    require_both_in_zone: bool = False,
):
    """
    檢查最後 window 根內是否出現 KD 黃金交叉；
    如 require_both_in_zone，則交叉當日 K、D 皆需落在 [zone_low, zone_high]。
    回傳交叉當日的整數索引（相對於 series 的位置），或 None。
    """
    n = len(K)
    if n < 2:
        return None
    start = max(1, n - window)
    for i in range(start, n):
        k_prev, d_prev = K.iloc[i - 1], D.iloc[i - 1]
        k_curr, d_curr = K.iloc[i], D.iloc[i]
        if np.isnan([k_prev, d_prev, k_curr, d_curr]).any():
            continue
        crossed = (k_prev <= d_prev) and (k_curr > d_curr)
        if not crossed:
            continue
        if require_both_in_zone:
            in_zone = (zone_low <= k_curr <= zone_high) and (zone_low <= d_curr <= zone_high)
        else:
            in_zone = True
        if in_zone:
            return i
    return None


def volume_today_over_ma20(volume: pd.Series, lookback: int = 20, multiplier: float = 1.5):
    """
    今日量 / 過去 lookback(不含今日) 平均量 >= multiplier -> True
    回傳 (bool, ratio, v_today, v_avg20)
    """
    if len(volume) < lookback + 1:
        return False, np.nan, np.nan, np.nan
    v_today = float(volume.iloc[-1])
    v_ref = float(volume.iloc[-(lookback + 1):-1].mean())
    if v_ref <= 0 or np.isnan(v_today) or np.isnan(v_ref):
        return False, np.nan, v_today, v_ref
    ratio = v_today / v_ref
    return (ratio >= multiplier), ratio, v_today, v_ref


def volume_min_last_n(volume: pd.Series, n: int = 10, min_shares: int = 1_000_000) -> bool:
    """
    近 n 日，每一日成交量皆 >= min_shares 才回 True
    """
    if len(volume) < n:
        return False
    lastn = volume.iloc[-n:]
    return bool((lastn >= min_shares).all())
