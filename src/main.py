#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import math
import time
import logging
import traceback
from datetime import datetime, date
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# ====== 專案內模組（沿用你現有的取數邏輯） ======
# - TWSE/TPEX 名單：請確保 fetch_twse_listed_equities() 回傳欄位至少包含 code/name/exchange
#   並已同時納入 上市(TWSE) + 上櫃(TPEX) 普通股
from universe.twse_listed import fetch_twse_listed_equities
# - Yahoo 歷史價量
from data_sources.yahoo import download_ohlcv_batches
# - Yahoo 市值（回傳新台幣）
from data_sources.yahoo_meta import get_market_caps


# ====== 目錄與記錄檔 ======
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
LOG_DIR = ROOT / "logs"
STATE_DIR = ROOT / "state"                   # 用來存連續出現的 streaks
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("twse_tpex_kd_screener")


# ====== 讀取 .env ======
def _to_bool(s: Optional[str], default: bool) -> bool:
    if s is None:
        return default
    return s.strip().lower() in ("1", "true", "yes", "y", "on")

def get_env_params() -> Dict:
    load_dotenv(ROOT / ".env", override=True)

    params = dict(
        # --- 產出數量 ---
        TOP_N=int(os.getenv("TOP_N", "20")),

        # --- KD ---
        KD_N=int(os.getenv("KD_N", "9")),
        KD_K_SMOOTH=int(os.getenv("KD_K_SMOOTH", "3")),
        KD_D_PERIOD=int(os.getenv("KD_D_PERIOD", "3")),
        KD_CROSS_WINDOW=int(os.getenv("KD_CROSS_WINDOW", "3")),

        # --- 價量 ---
        VOLUME_LOOKBACK=int(os.getenv("VOLUME_LOOKBACK", "20")),
        VOLUME_MULTIPLIER=float(os.getenv("VOLUME_MULTIPLIER", "1.5")),

        # --- 趨勢規則（開關 + 參數） ---
        ENABLE_RULE_MA5_GT_MA20=_to_bool(os.getenv("ENABLE_RULE_MA5_GT_MA20", "true"), True),
        ENABLE_RULE_OC_ABOVE_MA20=_to_bool(os.getenv("ENABLE_RULE_OC_ABOVE_MA20", "true"), True),
        ENABLE_RULE_LAST5_MA10_THRESHOLD=_to_bool(os.getenv("ENABLE_RULE_LAST5_MA10_THRESHOLD", "true"), True),
        MAX_DAYS_BELOW_MA10_IN_5=int(os.getenv("MAX_DAYS_BELOW_MA10_IN_5", "3")),

        # --- 黑K 限制 ---
        ENABLE_RULE_BLACK_CANDLE_LIMIT=_to_bool(os.getenv("ENABLE_RULE_BLACK_CANDLE_LIMIT", "true"), True),
        BLACK_CANDLE_MAX_DROP=float(os.getenv("BLACK_CANDLE_MAX_DROP", "0.95")),  # Close >= Open * 0.95

        # --- 流通性 ---
        LIQ_MIN_DAYS=int(os.getenv("LIQ_MIN_DAYS", "10")),
        LIQ_MIN_SHARES=int(os.getenv("LIQ_MIN_SHARES", "1000000")),

        # --- 市值（TWD） ---
        MARKET_CAP_MIN=float(os.getenv("MARKET_CAP_MIN", "10000000000")),

        # --- Yahoo 抓取 ---
        BATCH_SIZE=int(os.getenv("BATCH_SIZE", "120")),

        # --- Telegram ---
        TELEGRAM_BOT_TOKEN=os.getenv("TELEGRAM_BOT_TOKEN"),
        TELEGRAM_CHAT_ID=os.getenv("TELEGRAM_CHAT_ID"),

        # --- 連續出現 key ---
        CONTINUATION_KEY=os.getenv("CONTINUATION_KEY", "yahoo"),
    )
    logger.info(
        "Params loaded: KD=(%d,%d,%d), window=%d, VOL: N=%d x%.2f, "
        "MA rules: 5>20=%s, OC>=MA20=%s, 5d<MA10<=%d=%s, BlackK<=5%%=%s, "
        "LIQ: days=%d >= %d, TOP_N=%d, MCAP_MIN=%.0f",
        params["KD_N"], params["KD_K_SMOOTH"], params["KD_D_PERIOD"], params["KD_CROSS_WINDOW"],
        params["VOLUME_LOOKBACK"], params["VOLUME_MULTIPLIER"],
        params["ENABLE_RULE_MA5_GT_MA20"], params["ENABLE_RULE_OC_ABOVE_MA20"],
        params["MAX_DAYS_BELOW_MA10_IN_5"], params["ENABLE_RULE_LAST5_MA10_THRESHOLD"],
        params["ENABLE_RULE_BLACK_CANDLE_LIMIT"],
        params["LIQ_MIN_DAYS"], params["LIQ_MIN_SHARES"],
        params["TOP_N"], params["MARKET_CAP_MIN"]
    )
    return params


# ====== 工具：SMA / KD / 交叉偵測 ======
def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

def stochastic_kd(high: pd.Series, low: pd.Series, close: pd.Series,
                  n: int = 9, k_smooth: int = 3, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    low_n = low.rolling(n, min_periods=n).min()
    high_n = high.rolling(n, min_periods=n).max()
    denom = (high_n - low_n)
    rsv = (close - low_n) / denom.replace(0, np.nan) * 100.0
    rsv = rsv.fillna(50.0)
    K = rsv.rolling(k_smooth, min_periods=k_smooth).mean()
    D = K.rolling(d_period, min_periods=d_period).mean()
    return K, D

def golden_cross_in_window(K: pd.Series, D: pd.Series, window: int) -> Optional[int]:
    """
    回傳最近一次黃金交叉的索引位置（距今 window 之內），否則 None。
    黃金交叉定義：前一日 K<=D 且當日 K>D。
    """
    if len(K) < 2 or len(D) < 2:
        return None
    cross_mask = (K.shift(1) <= D.shift(1)) & (K > D)
    cross_idx = np.where(cross_mask.values)[0]
    if cross_idx.size == 0:
        return None
    last_i = len(K) - 1
    # 檢查是否有交叉點落在 [last_i-window+1, last_i]
    lo = max(0, last_i - window + 1)
    recent = cross_idx[cross_idx >= lo]
    if recent.size == 0:
        return None
    return int(recent[-1])  # 最近一次


# ====== 名單：上市+上櫃 → Yahoo 代碼 ======
def build_universe() -> pd.DataFrame:
    """
    需要回傳欄位：code, name, exchange, yahoo
    exchange ∈ {"TWSE","TPEX"} → Yahoo 後綴分別為 .TW / .TWO
    """
    logger.info("Building universe from TWSE/TPEX ISIN page...")
    df = fetch_twse_listed_equities()
    if "exchange" not in df.columns:
        df["exchange"] = "TWSE"  # 舊版 fallback

    def to_yahoo(row):
        code4 = str(row["code"]).zfill(4)
        suf = ".TW" if str(row["exchange"]).upper() in ("TWSE", "TSE", "上市") else ".TWO"
        return f"{code4}{suf}"

    df["yahoo"] = df.apply(to_yahoo, axis=1)
    return df[["code", "name", "exchange", "yahoo"]]


# ====== 送 Telegram ======
def send_telegram_message(token: Optional[str], chat_id: Optional[str], text: str) -> None:
    if not token or not chat_id:
        logger.info("Telegram not configured; skip sending.")
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, data={"chat_id": chat_id, "text": text})
        if resp.status_code != 200:
            logger.warning("Telegram send failed: %s %s", resp.status_code, resp.text)
        else:
            logger.info("Telegram sent: %d chars", len(text))
    except Exception as e:
        logger.warning("Telegram error: %s", e)


# ====== 連續出現（streaks）讀寫 ======
STREAKS_PATH = STATE_DIR / "streaks.json"

def load_streaks() -> Dict[str, Dict]:
    if STREAKS_PATH.exists():
        try:
            return json.loads(STREAKS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_streaks(streaks: Dict[str, Dict]) -> None:
    try:
        STREAKS_PATH.write_text(json.dumps(streaks, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Save streaks failed: %s", e)


# ====== 個股評估 ======
def evaluate_one(ysym: str, raw_df: pd.DataFrame, params: Dict, market_cap: Optional[float]) -> Optional[Dict]:
    """
    傳回滿足條件之摘要資料 dict，否則 None。
    需要欄位：Open/High/Low/Close/Volume
    """
    if raw_df is None or raw_df.empty:
        return None
    df = raw_df.copy()

    # 市值門檻
    if (market_cap is not None) and (market_cap < params["MARKET_CAP_MIN"]):
        return None

    # 需要最少的歷史長度（KD、MA20、量能計算）
    need_len = max(30, params["KD_N"] + params["KD_K_SMOOTH"] + params["KD_D_PERIOD"] + 5, params["VOLUME_LOOKBACK"] + 2)
    if len(df) < need_len:
        return None

    # 均線
    close = df["Close"]
    df["ma5"] = sma(close, 5)
    df["ma10"] = sma(close, 10)
    df["ma20"] = sma(close, 20)
    if math.isnan(df["ma20"].iloc[-1]) or math.isnan(df["ma10"].iloc[-1]):
        return None

    # KD
    K, D = stochastic_kd(df["High"], df["Low"], close,
                         n=params["KD_N"],
                         k_smooth=params["KD_K_SMOOTH"],
                         d_period=params["KD_D_PERIOD"])
    if math.isnan(K.iloc[-1]) or math.isnan(D.iloc[-1]):
        return None

    # ---- 海選條件 ----
    o, c, v = float(df["Open"].iloc[-1]), float(df["Close"].iloc[-1]), float(df["Volume"].iloc[-1])

    # E. 流通性：近 LIQ_MIN_DAYS 每日量 >= LIQ_MIN_SHARES
    liq_days = params["LIQ_MIN_DAYS"]
    liq_min = params["LIQ_MIN_SHARES"]
    if len(df) < liq_days:
        return None
    if (df["Volume"].iloc[-liq_days:] < liq_min).any():
        return None

    # D. 放量：今日量 / (過去20日均量，不含今日) >= 倍數
    look = params["VOLUME_LOOKBACK"]
    past20 = df["Volume"].iloc[-(look+1):-1]  # 不含今日
    if len(past20) < look:
        return None
    v20 = float(past20.mean())
    if v20 <= 0:
        return None
    vol_ratio = v / v20
    if vol_ratio < params["VOLUME_MULTIPLIER"]:
        return None

    # D2. 黑K 限制：若 c<o，則 c >= o * 0.95
    if params["ENABLE_RULE_BLACK_CANDLE_LIMIT"]:
        if c < o and c < o * params["BLACK_CANDLE_MAX_DROP"]:
            return None

    # B1. MA5 > MA20
    if params["ENABLE_RULE_MA5_GT_MA20"]:
        if not (df["ma5"].iloc[-1] > df["ma20"].iloc[-1]):
            return None

    # B2. 開盤 or 收盤 >= MA20
    if params["ENABLE_RULE_OC_ABOVE_MA20"]:
        if not (o >= df["ma20"].iloc[-1] or c >= df["ma20"].iloc[-1]):
            return None

    # B3. 近5日 收盤<MA10 的天數 <= 閾值
    if params["ENABLE_RULE_LAST5_MA10_THRESHOLD"]:
        last5 = df.iloc[-5:]
        cnt_below = int((last5["Close"] < last5["ma10"]).sum())
        if cnt_below > params["MAX_DAYS_BELOW_MA10_IN_5"]:
            return None

    # C. 近3日 KD 黃金交叉，且當下 K>D
    if not (K.iloc[-1] > D.iloc[-1]):
        return None
    cross_idx = golden_cross_in_window(K, D, params["KD_CROSS_WINDOW"])
    if cross_idx is None:
        return None

    # ---- 通過：計算輸出因子 ----
    kd_spread = float(K.iloc[-1] - D.iloc[-1])
    price_ma20_pct = float((c - df["ma20"].iloc[-1]) / df["ma20"].iloc[-1])

    return dict(
        date=pd.to_datetime(df.index[-1]).date().isoformat(),
        open=o,
        close=c,
        volume=v,
        ma5=float(df["ma5"].iloc[-1]),
        ma10=float(df["ma10"].iloc[-1]),
        ma20=float(df["ma20"].iloc[-1]),
        K=float(K.iloc[-1]),
        D=float(D.iloc[-1]),
        kd_spread=kd_spread,
        kd_cross_day=pd.to_datetime(df.index[cross_idx]).date().isoformat(),
        volume_ratio=float(vol_ratio),
        price_ma20_pct=price_ma20_pct,
    )


# ====== 主要流程 ======
def run_once():
    params = get_env_params()

    # 1) 建宇宙
    uni = build_universe()
    logger.info("Universe size: %d", len(uni))
    tickers: List[str] = uni["yahoo"].tolist()
    code_map = dict(zip(uni["yahoo"], uni["code"]))
    name_map = dict(zip(uni["yahoo"], uni["name"]))
    exch_map = dict(zip(uni["yahoo"], uni["exchange"]))

    # 2) 下載價量
    logger.info("Downloading OHLCV from Yahoo in batches...")
    data_map: Dict[str, pd.DataFrame] = download_ohlcv_batches(
        tickers, period="6mo", interval="1d",
        batch_size=params["BATCH_SIZE"], retries=2, sleep_sec=1.0
    )

    # 檢查是否為「非交易日」
    # 取得所有有資料標的的最新日期，與台北當日比較（若小於今日 → 視為非交易日）
    dates = []
    ref_prev = None
    for df in data_map.values():
        if df is not None and not df.empty:
            dates.append(pd.to_datetime(df.index[-1]).date())
            if ref_prev is None and len(df) >= 2:
                ref_prev = pd.to_datetime(df.index[-2]).date()
    if not dates:
        dates = []
    latest_trade_date = max(dates) if dates else None
    today_tpe = datetime.now(ZoneInfo("Asia/Taipei")).date()

    if (latest_trade_date is None) or (latest_trade_date < today_tpe):
        # 建立空 CSV 以便 workflow 上傳 artifact
        out_empty = OUTPUT_DIR / f"picks_{today_tpe.strftime('%Y%m%d')}.csv"
        pd.DataFrame(columns=[
            "date","code","name","exchange","yahoo","close","K","D","volume_ratio",
            "kd_spread","price_ma20_pct","score","continuation_days"
        ]).to_csv(out_empty, index=False, encoding="utf-8-sig")
        # 友善訊息
        send_telegram_message(params["TELEGRAM_BOT_TOKEN"], params["TELEGRAM_CHAT_ID"],
                              "今日為非交易日，請開心過好每一天")
        logger.info("No today bars -> Non-trading day. Wrote empty CSV: %s", out_empty)
        return

    # 3) 市值
    logger.info("Fetching market caps from Yahoo...")
    mc_map = get_market_caps(tickers, retries=1, sleep=0.05)

    # 4) 評估全部
    rows = []
    for ysym, df in data_map.items():
        try:
            sig = evaluate_one(ysym, df, params, market_cap=mc_map.get(ysym))
            if sig:
                rows.append({
                    "date": sig["date"],
                    "code": code_map.get(ysym, ""),
                    "name": name_map.get(ysym, ""),
                    "exchange": exch_map.get(ysym, ""),
                    "yahoo": ysym,
                    "open": sig["open"],
                    "close": sig["close"],
                    "volume": sig["volume"],
                    "ma5": sig["ma5"],
                    "ma10": sig["ma10"],
                    "ma20": sig["ma20"],
                    "K": sig["K"],
                    "D": sig["D"],
                    "kd_spread": sig["kd_spread"],
                    "kd_cross_day": sig["kd_cross_day"],
                    "volume_ratio": sig["volume_ratio"],
                    "price_ma20_pct": sig["price_ma20_pct"],
                    "market_cap": mc_map.get(ysym)
                })
        except Exception as e:
            logger.warning("Signal evaluation failed for %s: %s", ysym, e)

    # 無標的 → 仍輸出空檔
    today_str = latest_trade_date.strftime("%Y%m%d")
    base_cols = ["date","code","name","exchange","yahoo","close","K","D","volume_ratio",
                 "kd_spread","price_ma20_pct","rank_kd","rank_vol","a","b","c","score","continuation_days"]
    if not rows:
        out_path = OUTPUT_DIR / f"picks_{today_str}.csv"
        pd.DataFrame(columns=base_cols).to_csv(out_path, index=False, encoding="utf-8-sig")
        send_telegram_message(params["TELEGRAM_BOT_TOKEN"], params["TELEGRAM_CHAT_ID"], "今日無符合條件之個股。")
        logger.info("Saved empty results to %s", out_path)
        return

    df_all = pd.DataFrame(rows)

    # 5) 三因子排名與分數
    # 依 kd_spread、volume_ratio 由高到低排序取得名次（1=最好）
    df_all["rank_kd"] = df_all["kd_spread"].rank(method="min", ascending=False).astype(int)
    df_all["rank_vol"] = df_all["volume_ratio"].rank(method="min", ascending=False).astype(int)

    # a=(2-0.02*n_k)，b=(2-0.02*n_v)；避免為負，做下限 0
    df_all["a"] = (2.0 - 0.02 * df_all["rank_kd"]).clip(lower=0.0)
    df_all["b"] = (2.0 - 0.02 * df_all["rank_vol"]).clip(lower=0.0)
    df_all["c"] = df_all["price_ma20_pct"]
    df_all["score"] = df_all["a"] * df_all["b"] * df_all["c"]

    # 6) 取前 TOP_N
    df_all = df_all.sort_values("score", ascending=False).reset_index(drop=True)
    df_top = df_all.head(params["TOP_N"]).copy()

    # 7) 連續出現（以昨日交易日為準）
    prev_trade_date = ref_prev or (latest_trade_date)  # 若缺，就無法判斷嚴謹連續，視為首次
    key_field = params.get("CONTINUATION_KEY", "yahoo")
    if key_field not in df_top.columns:
        key_field = "yahoo"

    streaks = load_streaks()  # { key: {"last_date":"YYYY-MM-DD","streak":int} }
    updated = {}

    cont_days = []
    for _, r in df_top.iterrows():
        key = str(r[key_field])
        prev = streaks.get(key)
        if prev and prev.get("last_date") == str(prev_trade_date):
            days = int(prev.get("streak", 1)) + 1
        else:
            days = 1
        cont_days.append(days)
        updated[key] = {"last_date": str(latest_trade_date), "streak": days}
    df_top["continuation_days"] = cont_days

    # 清理舊 streaks：僅保留今日榜單中的 key（避免無限膨脹）
    save_streaks(updated)

    # 8) 依「是否連續 >=2」分兩張表
    df_cont = df_top[df_top["continuation_days"] >= 2].copy()
    df_fresh = df_top[df_top["continuation_days"] == 1].copy()

    # 9) 輸出 CSV
    out_all = OUTPUT_DIR / f"picks_{today_str}.csv"
    out_cont = OUTPUT_DIR / f"picks_{today_str}_top_continuous.csv"
    out_fresh = OUTPUT_DIR / f"picks_{today_str}_top_fresh.csv"
    # 統一欄位順序
    order_cols = ["date","code","name","exchange","yahoo","close","K","D","volume_ratio",
                  "kd_spread","price_ma20_pct","rank_kd","rank_vol","a","b","c","score","continuation_days"]
    df_top[order_cols].to_csv(out_all, index=False, encoding="utf-8-sig")
    df_cont[order_cols].to_csv(out_cont, index=False, encoding="utf-8-sig")
    df_fresh[order_cols].to_csv(out_fresh, index=False, encoding="utf-8-sig")
    logger.info("Saved results: all=%s (count=%d), cont=%s (%d), fresh=%s (%d)",
                out_all, len(df_top), out_cont, len(df_cont), out_fresh, len(df_fresh))

    # 10) Telegram 摘要（只顯示：收盤、KD、放量倍數；連續清單加上「連N」）
    def fmt_row(r) -> str:
        return f"{r['code']} {r['name']} | 收 {r['close']:.2f} | KD {r['K']:.1f}/{r['D']:.1f} | 量 {r['volume_ratio']:.2f}x"

    lines: List[str] = []
    lines.append(f"📈 {latest_trade_date} KD選股（前{params['TOP_N']}）")

    if len(df_cont) > 0:
        lines.append("— 連續出現（≥2天） —")
        for _, r in df_cont.iterrows():
            lines.append(fmt_row(r) + f" | 連{int(r['continuation_days'])}")
    else:
        lines.append("— 連續出現（≥2天） — 無")

    if len(df_fresh) > 0:
        lines.append("— 非連續 —")
        for _, r in df_fresh.iterrows():
            lines.append(fmt_row(r))
    else:
        lines.append("— 非連續 — 無")

    msg = "\n".join(lines)
    send_telegram_message(params["TELEGRAM_BOT_TOKEN"], params["TELEGRAM_CHAT_ID"], msg)


if __name__ == "__main__":
    try:
        run_once()
    except Exception as e:
        logger.error("Fatal error: %s\n%s", e, traceback.format_exc())
        # 盡量回報錯誤到 Telegram，方便遠端看
        try:
            p = get_env_params()
            send_telegram_message(p.get("TELEGRAM_BOT_TOKEN"), p.get("TELEGRAM_CHAT_ID"),
                                  f"❌ Screener 失敗：{e}")
        except Exception:
            pass
        sys.exit(1)
