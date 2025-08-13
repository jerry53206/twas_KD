#!/usr/bin/env python3
import os
import sys
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from universe.twse_listed import fetch_twse_listed_equities  # TWSE+TPEX
from data_sources.yahoo import download_ohlcv_batches
from data_sources.yahoo_meta import get_market_caps
from indicators.ta import stochastic_kd
from filters.conditions import (
    golden_cross_in_window,
    volume_today_over_ma20,
    volume_min_last_n,
)

# --------------------- 基礎設定 ---------------------
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
LOG_DIR = ROOT / "logs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("twse_kd_screener")


def _bool_env(v: str, default=False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def get_env_params() -> Dict:
    load_dotenv(ROOT / ".env", override=True)
    return dict(
        # KD 參數 & 條件
        KD_N=int(os.getenv("KD_N", "9")),
        KD_K_SMOOTH=int(os.getenv("KD_K_SMOOTH", "3")),
        KD_D_PERIOD=int(os.getenv("KD_D_PERIOD", "3")),
        KD_CROSS_WINDOW=int(os.getenv("KD_CROSS_WINDOW", "3")),
        KD_REQUIRE_ZONE=_bool_env(os.getenv("KD_REQUIRE_ZONE", "false")),
        KD_ZONE_LOW=float(os.getenv("KD_ZONE_LOW", "40")),
        KD_ZONE_HIGH=float(os.getenv("KD_ZONE_HIGH", "80")),

        # 量能規則（當日量 > 20MA × 倍數）
        VOLUME_LOOKBACK=int(os.getenv("VOLUME_LOOKBACK", "20")),
        VOLUME_MULTIPLIER=float(os.getenv("VOLUME_MULTIPLIER", "1.5")),

        # 黑K 限制
        ENABLE_RULE_BLACK_CANDLE_LIMIT=_bool_env(os.getenv("ENABLE_RULE_BLACK_CANDLE_LIMIT", "true")),
        BLACK_CANDLE_MAX_DROP=float(os.getenv("BLACK_CANDLE_MAX_DROP", "0.95")),

        # MA 結構
        ENABLE_RULE_OC_ABOVE_MA20=_bool_env(os.getenv("ENABLE_RULE_OC_ABOVE_MA20", "true")),
        ENABLE_RULE_LAST5_MA10_THRESHOLD=_bool_env(os.getenv("ENABLE_RULE_LAST5_MA10_THRESHOLD", "true")),
        MAX_DAYS_BELOW_MA10_IN_5=int(os.getenv("MAX_DAYS_BELOW_MA10_IN_5", "3")),
        ENABLE_RULE_MA5_GT_MA20=_bool_env(os.getenv("ENABLE_RULE_MA5_GT_MA20", "true")),

        # 流動性門檻（新增）
        MIN_LIQ_N=int(os.getenv("MIN_LIQ_N", "10")),
        MIN_LIQ_SHARES=int(os.getenv("MIN_LIQ_SHARES", "1000000")),

        # 市值門檻
        MARKET_CAP_MIN=float(os.getenv("MARKET_CAP_MIN", "10000000000")),  # 100 億

        # 報表與批次
        TOP_N=int(os.getenv("TOP_N", "20")),     # 供摘要展示用
        FINAL_TOP=int(os.getenv("FINAL_TOP", "5")),  # 最終操作標的數
        BATCH_SIZE=int(os.getenv("BATCH_SIZE", "120")),

        # Telegram（可留空）
        TELEGRAM_BOT_TOKEN=os.getenv("TELEGRAM_BOT_TOKEN"),
        TELEGRAM_CHAT_ID=os.getenv("TELEGRAM_CHAT_ID"),
    )


def build_universe(params: Dict) -> pd.DataFrame:
    """
    回傳欄位：code, name, exchange, yahoo
    上市(TWSE) 用 .TW，上櫃(TPEX) 用 .TWO
    """
    df = fetch_twse_listed_equities()  # 已含 TWSE+TPEX
    suffix = df["exchange"].map({"TWSE": ".TW", "TPEX": ".TWO"}).fillna(".TW")
    df["yahoo"] = df["code"].astype(str).str.zfill(4) + suffix
    return df[["code", "name", "exchange", "yahoo"]]


def _moving_averages(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    return ma5, ma10, ma20


def _send_telegram(bot_token: str, chat_id: str, text: str):
    """
    極簡 Telegram 發送（用 requests）。如果未設定，直接略過。
    """
    if not bot_token or not chat_id:
        logger.info("Telegram not configured; skip Telegram sending.")
        return
    import requests
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(url, data={"chat_id": chat_id, "text": text})
        if resp.status_code != 200:
            logger.warning(f"Telegram send failed: HTTP {resp.status_code} {resp.text}")
        else:
            logger.info("Telegram message sent.")
    except Exception as e:
        logger.warning(f"Telegram send exception: {e}")


def evaluate_signals_for_ticker(
    df: pd.DataFrame, params: Dict, market_cap: float = None
) -> Dict | None:
    """
    給單一標的的 OHLCV（欄位：Open, High, Low, Close, Volume；日線），檢核是否通過『海選』。
    若通過，回傳包含必要數值（供後續排序）之 dict；否則回傳 None。
    """
    if df is None or df.empty:
        return None
    if df["Volume"].fillna(0).sum() == 0:
        return None

    # 市值門檻
    if market_cap is not None and market_cap < params["MARKET_CAP_MIN"]:
        return None

    close = df["Close"]
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # 需至少有足夠歷史長度
    need_len = max(60, params["VOLUME_LOOKBACK"] + params["KD_N"] + params["KD_K_SMOOTH"] + params["KD_D_PERIOD"] + 5)
    if len(df) < need_len:
        return None

    # 計算 KD
    K, D = stochastic_kd(
        high, low, close,
        n=params["KD_N"],
        k_smooth=params["KD_K_SMOOTH"],
        d_period=params["KD_D_PERIOD"]
    )
    if K is None or D is None:
        return None

    # C1: 最近 3 日曾黃金交叉；且「當下」K > D；（可選）交叉當日要在區間
    cross_idx = golden_cross_in_window(
        K, D,
        window=params["KD_CROSS_WINDOW"],
        zone_low=params["KD_ZONE_LOW"],
        zone_high=params["KD_ZONE_HIGH"],
        require_both_in_zone=params["KD_REQUIRE_ZONE"]
    )
    if cross_idx is None:
        return None
    if not (K.iloc[-1] > D.iloc[-1]):
        return None

    # D1: 當日放量 ≥ 20MA × 倍數
    vol_ok, vol_ratio, v_today, v20avg = volume_today_over_ma20(
        volume, lookback=params["VOLUME_LOOKBACK"], multiplier=params["VOLUME_MULTIPLIER"]
    )
    if not vol_ok:
        return None

    # E1: 近 10 日，每一日成交量皆 ≥ 100萬股
    if not volume_min_last_n(volume, n=params["MIN_LIQ_N"], min_shares=params["MIN_LIQ_SHARES"]):
        return None

    # B: 結構與保護
    ma5, ma10, ma20 = _moving_averages(close)
    if params["ENABLE_RULE_MA5_GT_MA20"]:
        if not (ma5.iloc[-1] > ma20.iloc[-1]):
            return None

    if params["ENABLE_RULE_OC_ABOVE_MA20"]:
        if not ((open_.iloc[-1] >= ma20.iloc[-1]) or (close.iloc[-1] >= ma20.iloc[-1])):
            return None

    if params["ENABLE_RULE_LAST5_MA10_THRESHOLD"]:
        last5_close = close.iloc[-5:]
        last5_ma10 = ma10.iloc[-5:]
        days_below = int((last5_close < last5_ma10).sum())
        if days_below > params["MAX_DAYS_BELOW_MA10_IN_5"]:
            return None

    # D2: 黑K 限制
    if params["ENABLE_RULE_BLACK_CANDLE_LIMIT"]:
        if close.iloc[-1] < open_.iloc[-1]:
            if not (close.iloc[-1] >= open_.iloc[-1] * params["BLACK_CANDLE_MAX_DROP"]):
                return None

    # 供排序用的指標
    kd_spread = float(K.iloc[-1] - D.iloc[-1])
    c_trend = float((close.iloc[-1] - ma20.iloc[-1]) / ma20.iloc[-1]) if ma20.iloc[-1] > 0 else np.nan

    return {
        "date": pd.to_datetime(df.index[-1]).date().isoformat(),
        "close": float(close.iloc[-1]),
        "open": float(open_.iloc[-1]),
        "K": float(K.iloc[-1]) if not np.isnan(K.iloc[-1]) else None,
        "D": float(D.iloc[-1]) if not np.isnan(D.iloc[-1]) else None,
        "kd_spread": kd_spread,          # 因子 a 的排序依據
        "vol_ratio": float(vol_ratio),   # 因子 b 的排序依據
        "ma20": float(ma20.iloc[-1]),
        "trend_c": c_trend,              # 因子 c 直接取值
        "cross_day": pd.to_datetime(df.index[cross_idx]).date().isoformat(),
        "v_today": float(v_today) if not np.isnan(v_today) else None,
        "v20avg": float(v20avg) if not np.isnan(v20avg) else None,
    }


def run_once():
    params = get_env_params()
    logger.info(
        "Params loaded: KD=(%d,%d,%d), window=%d, zone=%s[%g~%g], "
        "VOL: N=%d x%.2f, MIN_LIQ=%dd≥%d, TOP_N=%d, FINAL_TOP=%d, MCAP_MIN=%d",
        params["KD_N"], params["KD_K_SMOOTH"], params["KD_D_PERIOD"],
        params["KD_CROSS_WINDOW"],
        "ON" if params["KD_REQUIRE_ZONE"] else "OFF",
        params["KD_ZONE_LOW"], params["KD_ZONE_HIGH"],
        params["VOLUME_LOOKBACK"], params["VOLUME_MULTIPLIER"],
        params["MIN_LIQ_N"], params["MIN_LIQ_SHARES"],
        params["TOP_N"], params["FINAL_TOP"], int(params["MARKET_CAP_MIN"])
    )

    logger.info("Building universe from TWSE/TPEX ISIN page...")
    uni = build_universe(params)
    logger.info(f"Universe size: {len(uni)}")

    tickers = uni["yahoo"].tolist()
    code_map = dict(zip(uni["yahoo"], uni["code"]))
    name_map = dict(zip(uni["yahoo"], uni["name"]))

    logger.info("Downloading OHLCV from Yahoo in batches...")
    data_map = download_ohlcv_batches(
        tickers, period="6mo", interval="1d",
        batch_size=params["BATCH_SIZE"], retries=2, sleep_sec=1.0
    )

    # 先做「放量 & 基本結構」的初篩，再查市值（減少 meta 呼叫量）
    prelim = {}
    for ysym, df in data_map.items():
        try:
            # 先不帶市值，僅做基本檢核：量能/MA/KD；通過者再補市值檢核
            sig = evaluate_signals_for_ticker(df, {**params, "MARKET_CAP_MIN": -1})
            if sig:
                prelim[ysym] = sig
        except Exception as e:
            logger.warning(f"Preliminary evaluation failed for {ysym}: {e}")

    # 取得市值，僅限 prelim 候選名單
    logger.info(f"Fetching market caps from Yahoo for {len(prelim)} candidates...")
    mc_map = get_market_caps(list(prelim.keys()), retries=1, sleep=0.05)

    # 補上市值門檻，再形成 rows
    rows: List[Dict] = []
    for ysym, sig in prelim.items():
        mcap = mc_map.get(ysym)
        if mcap is None or mcap < params["MARKET_CAP_MIN"]:
            continue
        rows.append({
            "date": sig["date"],
            "code": code_map.get(ysym, ""),
            "name": name_map.get(ysym, ""),
            "close": sig["close"],
            "K": sig["K"],
            "D": sig["D"],
            "kd_spread": sig["kd_spread"],     # a 的排序依據
            "vol_ratio": sig["vol_ratio"],     # b 的排序依據
            "ma20": sig["ma20"],
            "trend_c": sig["trend_c"],         # c 的原始值
            "cross_day": sig["cross_day"],
            "market_cap": float(mcap),
        })

    if rows:
        df_out = pd.DataFrame(rows)
        # ------- 排序因子 -------
        # a: 依 kd_spread 由高到低排名 -> a = 1.1 - 0.02 * n
        df_out["rank_a"] = df_out["kd_spread"].rank(method="min", ascending=False).astype(int)
        df_out["a"] = 1.1 - 0.02 * df_out["rank_a"]

        # b: 依 vol_ratio 由高到低排名 -> b = 1.1 - 0.02 * n
        df_out["rank_b"] = df_out["vol_ratio"].rank(method="min", ascending=False).astype(int)
        df_out["b"] = 1.1 - 0.02 * df_out["rank_b"]

        # c: 直接取 trend_c = (Close - MA20)/MA20
        df_out["c"] = df_out["trend_c"]

        # 最終分數
        df_out["score"] = df_out["a"] * df_out["b"] * df_out["c"]

        # 依 score 由高到低
        df_out = df_out.sort_values(["score"], ascending=False, kind="mergesort")
    else:
        df_out = pd.DataFrame(
            columns=["date","code","name","close","K","D","kd_spread","vol_ratio","ma20","trend_c","cross_day","market_cap","a","b","c","score"]
        )

    today = datetime.now().strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"picks_{today}.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved results to {out_path} (count={len(df_out)})")

    # -------- Telegram 摘要（僅顯示：今日收盤、KD、放量倍數） --------
    try:
        if len(df_out) == 0:
            msg = f"【TWSE/TPEX KD Screener】{today}\n今日無符合條件之個股。"
        else:
            top_n = min(params["TOP_N"], len(df_out))
            head = df_out.head(top_n)
            lines = [f"【TWSE/TPEX KD Screener】{today} 前{top_n}（依 score）"]
            for _, r in head.iterrows():
                k = r['K'] if pd.notna(r['K']) else np.nan
                d = r['D'] if pd.notna(r['D']) else np.nan
                lines.append(
                    f"{r['code']} {r['name']} | 收盤 {r['close']:.2f} | KD {k:.2f}/{d:.2f} | 量能倍數 {r['vol_ratio']:.2f}"
                )
            msg = "\n".join(lines)

        _send_telegram(params.get("TELEGRAM_BOT_TOKEN"), params.get("TELEGRAM_CHAT_ID"), msg)

    except Exception as e:
        logger.error(f"Telegram sending failed: {e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    run_once()
