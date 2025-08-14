# ====================== 1) evaluate_signals_for_ticker（整段覆蓋） ======================
def evaluate_signals_for_ticker(
    df: pd.DataFrame, params: dict, market_cap: float = None
) -> Dict:
    """
    給定單一標的的日線 OHLCV（含 Volume），檢查是否通過所有「海選」條件；
    通過則回傳訊號資訊 dict，否則回傳 None。

    依據條件：
    A. 市值 >= MARK​ET_CAP_MIN（若有取到市值）
    B. 結構/均線：
       - MA5 > MA20
       - 當日 Open 或 Close >= MA20
       - 近 5 日中，收盤價 < MA10 的天數 <= 3
       - 黑K 限制：若 Close < Open，則 Close >= Open * BLACK_CANDLE_MAX_DROP（預設 0.95）
    C. 訊號：
       - 最近 KD_CROSS_WINDOW 日內出現 K 向上穿越 D，且「當下（最後一日）K > D」
       - （可選）若 KD_REQUIRE_ZONE=True，則交叉當日 K、D 都在 [KD_ZONE_LOW, KD_ZONE_HIGH]
    D. 價量：
       - 今日量 / 過去 20 日均量 >= VOLUME_MULTIPLIER（預設 1.5）
    E. 流動性：
       - 最近 10 個交易日，每一日成交量 >= 1,000,000

    回傳欄位將包含：close, K, D, vol_ratio, cross_day, ma20, px_vs_ma20 等
    """

    if df is None or df.empty:
        return None

    # 確保 Volume 有意義
    if df["Volume"].fillna(0).sum() == 0:
        return None

    # ---- 市值門檻 ----
    if market_cap is not None and market_cap < float(params.get("MARKET_CAP_MIN", 1e10)):
        return None

    # ---- 基本資料與均線 ----
    close = df["Close"]
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()

    # 歷史足夠性檢查
    need_len = max(
        int(params.get("VOLUME_LOOKBACK", 20)) + 1,
        30,
        int(params.get("KD_N", 9)) + int(params.get("KD_K_SMOOTH", 3)) + int(params.get("KD_D_PERIOD", 3)),
    )
    if len(df) < need_len:
        return None

    # ---- KD 計算 ----
    K, D = stochastic_kd(
        high, low, close,
        n=int(params.get("KD_N", 9)),
        k_smooth=int(params.get("KD_K_SMOOTH", 3)),
        d_period=int(params.get("KD_D_PERIOD", 3))
    )
    if K is None or D is None or np.isnan(K.iloc[-1]) or np.isnan(D.iloc[-1]):
        return None

    # ---- KD 黃金交叉 within window + 當下 K > D +（可選）區間限制 ----
    window = int(params.get("KD_CROSS_WINDOW", 3))
    require_zone = bool(str(params.get("KD_REQUIRE_ZONE", "false")).lower() == "true")
    zone_low = float(params.get("KD_ZONE_LOW", 40.0))
    zone_high = float(params.get("KD_ZONE_HIGH", 80.0))

    def _find_cross_idx_in_last_n(Ks: pd.Series, Ds: pd.Series, n: int) -> int | None:
        start = max(1, len(Ks) - n)
        for i in range(start, len(Ks)):
            k_prev, d_prev = Ks.iloc[i - 1], Ds.iloc[i - 1]
            k_curr, d_curr = Ks.iloc[i], Ds.iloc[i]
            if np.isnan([k_prev, d_prev, k_curr, d_curr]).any():
                continue
            crossed = (k_prev <= d_prev) and (k_curr > d_curr)
            if not crossed:
                continue
            if require_zone:
                in_zone = (zone_low <= k_curr <= zone_high) and (zone_low <= d_curr <= zone_high)
                if not in_zone:
                    continue
            return i
        return None

    cross_idx = _find_cross_idx_in_last_n(K, D, window)
    if cross_idx is None:
        return None

    # 「當下」K > D（最後一根）
    if not (K.iloc[-1] > D.iloc[-1]):
        return None

    # ---- 放量規則：今日量 / 過去 20 日均量 >= multiplier（母體為前 20 日，不含今日） ----
    lookback = int(params.get("VOLUME_LOOKBACK", 20))
    multiplier = float(params.get("VOLUME_MULTIPLIER", 1.5))
    if len(vol) < lookback + 1:
        return None
    v_today = float(vol.iloc[-1])
    v20 = float(vol.iloc[-(lookback + 1):-1].mean())  # 不含今日
    if v20 <= 0:
        return None
    vol_ratio = v_today / v20
    if vol_ratio < multiplier:
        return None

    # ---- 流動性門檻：最近 10 日，每一日成交量 >= 1,000,000 ----
    liq_n = int(params.get("LIQ_MIN_VOLUME_N", 10))
    liq_min = int(params.get("LIQ_MIN_SHARES", 1_000_000))
    if len(vol) < liq_n:
        return None
    if not (vol.iloc[-liq_n:] >= liq_min).all():
        return None

    # ---- 均線/結構規則 ----
    ma5_gt_ma20_en = bool(str(params.get("ENABLE_RULE_MA5_GT_MA20", "true")).lower() == "true")
    if ma5_gt_ma20_en and not (ma5.iloc[-1] > ma20.iloc[-1]):
        return None

    oc_above_ma20_en = bool(str(params.get("ENABLE_RULE_OC_ABOVE_MA20", "true")).lower() == "true")
    if oc_above_ma20_en and not ((open_.iloc[-1] >= ma20.iloc[-1]) or (close.iloc[-1] >= ma20.iloc[-1])):
        return None

    last5_ma10_en = bool(str(params.get("ENABLE_RULE_LAST5_MA10_THRESHOLD", "true")).lower() == "true")
    max_below_days = int(params.get("MAX_DAYS_BELOW_MA10_IN_5", 3))
    if last5_ma10_en:
        recent5 = close.iloc[-5:]
        ma10_5 = ma10.iloc[-5:]
        below_days = int((recent5 < ma10_5).sum())
        if below_days > max_below_days:
            return None

    # ---- 黑K 限制 ----
    black_en = bool(str(params.get("ENABLE_RULE_BLACK_CANDLE_LIMIT", "true")).lower() == "true")
    black_drop = float(params.get("BLACK_CANDLE_MAX_DROP", 0.95))
    if black_en and (close.iloc[-1] < open_.iloc[-1]):
        if not (close.iloc[-1] >= open_.iloc[-1] * black_drop):
            return None

    # ---- 回傳訊息（供排序與輸出）----
    last_idx = df.index[-1]
    ma20_today = float(ma20.iloc[-1]) if not np.isnan(ma20.iloc[-1]) else None
    px_vs_ma20 = (float(close.iloc[-1]) - ma20_today) / ma20_today if (ma20_today and ma20_today != 0) else None

    res = {
        "date": pd.to_datetime(last_idx).date().isoformat(),
        "close": float(close.iloc[-1]),
        "K": float(K.iloc[-1]) if not np.isnan(K.iloc[-1]) else None,
        "D": float(D.iloc[-1]) if not np.isnan(D.iloc[-1]) else None,
        "vol_ratio": float(vol_ratio),
        "cross_day": pd.to_datetime(df.index[cross_idx]).date().isoformat(),
        "ma20": ma20_today,
        "px_vs_ma20": px_vs_ma20,  # 排序因子 c 會用到
    }
    return res


# ====================== 2) 排序/權重區塊（放在 run_once() 末端，覆蓋原排序） ======================
    # ---- Ranking system (updated factors a, b, c) ----
    if rows:
        df_out = pd.DataFrame(rows)
        # 動能擴散（K-D）、量比名次
        df_out["k_d_spread"] = df_out["K"] - df_out["D"]
        # 名次從 1 起算（由高到低）
        df_out["n_k"] = df_out["k_d_spread"].rank(ascending=False, method="min").astype(int)
        df_out["n_v"] = df_out["vol_ratio"].rank(ascending=False, method="min").astype(int)

        # 因子 a、b
        df_out["a"] = 2.0 - 0.02 * df_out["n_k"]
        df_out["b"] = 2.0 - 0.02 * df_out["n_v"]
        # 若你擔心樣本數很大導致 a/b 變負，可解開下面兩行做裁切：
        # df_out["a"] = df_out["a"].clip(lower=0)
        # df_out["b"] = df_out["b"].clip(lower=0)

        # 因子 c = (Close - MA20) / MA20
        if "px_vs_ma20" in df_out.columns and df_out["px_vs_ma20"].notna().any():
            df_out["c"] = df_out["px_vs_ma20"]
        elif {"close", "ma20"}.issubset(df_out.columns):
            df_out["c"] = (df_out["close"] - df_out["ma20"]) / df_out["ma20"]
        else:
            df_out["c"] = np.nan
            logger.warning("px_vs_ma20 / ma20 欄位缺少，c 因子無法完整計算。")

        # 最終分數
        df_out["final_score"] = df_out["a"] * df_out["b"] * df_out["c"]

        # 依 final_score（高→低）排序；同分再看 vol_ratio
        df_out = df_out.sort_values(["final_score", "vol_ratio"], ascending=[False, False])

        # 只保留你需要輸出的欄位
        cols = [
            "date","code","name","close","K","D","vol_ratio","cross_day",
            "ma20","px_vs_ma20","a","b","c","final_score","market_cap"
        ]
        df_out = df_out[[c for c in cols if c in df_out.columns]]

        # 取 Top N
        top_n = int(params.get("TOP_N", 20))
        df_out = df_out.head(top_n)
    else:
        df_out = pd.DataFrame(columns=[
            "date","code","name","close","K","D","vol_ratio","cross_day",
            "ma20","px_vs_ma20","a","b","c","final_score","market_cap"
        ])
