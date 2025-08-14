#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytz
from dotenv import load_dotenv

# ----------------- 專案內模組 -----------------
from universe.twse_listed import fetch_twse_listed_equities
from data_sources.yahoo import download_ohlcv_batches
from data_sources.yahoo_meta import get_market_caps
from indicators.ta import stochastic_kd

# Telegram 發送（若無 notify.telegram 模組則用後援）
try:
    from notify.telegram import send_telegram_message  # def send_telegram_message(bot_token, chat_id, text)
except Exception:
    import requests
    def send_telegram_message(bot_token: str, chat_id: str, text: str):
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            r = requests.post(url, data={"chat_id": chat_id, "text": text})
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Telegram API error: {e}")

# ----------------- 目錄與記錄 -----------------
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
LOG_DIR = ROOT / "logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("twse_kd_screener")


# ----------------- 讀取 .env 參數 -----------------
def get_env_params() -> Dict:
    load_dotenv(ROOT / ".env", override=True)
    params = dict(
        # KD
        KD_N=int(os.getenv("KD_N", "9")),
        KD_K_SMOOTH=int(os.getenv("KD_K_SMOOTH", "3")),
        KD_D_PERIOD=int(os.getenv("KD_D_PERIOD", "3")),
        KD_CROSS_WINDOW=int(os.getenv("KD_CROSS_WINDOW", "3")),
        KD_REQUIRE_ZONE=os.getenv("KD_REQUIRE_ZONE", "false").lower() == "true",
        KD_ZONE_LOW=float(os.getenv("KD_ZONE_LOW", "40")),
        KD_ZONE_HIGH=float(os.getenv("KD_ZONE_HIGH", "80")),

        # Volume today vs 20-day avg
        VOLUME_LOOKBACK=int(os.getenv("VOLUME_LOOKBACK", "20")),
        VOLUME_MULTIPLIER=float(os.getenv("VOLUME_MULTIPLIER", "1.5")),

        # 10日流動性下限（每日皆 >= 100萬股）
        MIN_DAILY_VOLUME_10D=int(os.getenv("MIN_DAILY_VOLUME_10D", "1000000")),

        # 價格/MAs 規則
        ENABLE_RULE_BLACK_CANDLE_LIMIT=os.getenv("ENABLE_RULE_BLACK_CANDLE_LIMIT", "true").lower() == "true",
        BLACK_CANDLE_MAX_DROP=float(os.getenv("BLACK_CANDLE_MAX_DROP", "0.95")),  # 收黑K時 C >= O*0.95
        ENABLE_RULE_OC_ABOVE_MA20=os.getenv("ENABLE_RULE_OC_ABOVE_MA20", "true").lower() == "true",
        ENABLE_RULE_LAST5_MA10_THRESHOLD=os.getenv("ENABLE_RULE_LAST5_MA10_THRESHOLD", "true").lower() == "true",
        MAX_DAYS_BELOW_MA10_IN_5=int(os.getenv("MAX_DAYS_BELOW_MA10_IN_5", "3")),
        ENABLE_RULE_MA5_GT_MA20=os.getenv("ENABLE_RULE_MA5_GT_MA20", "true").lower() == "true",

        # 市值
        MARKET_CAP_MIN=float(os.getenv("MARKET_CAP_MIN", "10000000000")),  # 100億

        # 取資料
        BATCH_SIZE=int(os.getenv("BATCH_SIZE", "120")),

        # 排名與輸出
        TOP_N=int(os.getenv("TOP_N", "20")),

        # 通知
        TELEGRAM_BOT_TOKEN=os.getenv("TELEGRAM_BOT_TOKEN"),
        TELEGRAM_CHAT_ID=os.getenv("TELEGRAM_CHAT_ID"),
    )
    zone_flag = "ON" if params["KD_REQUIRE_ZONE"] else "OFF"
    logger.info(
        "Params loaded: KD=(%d,%d,%d), window=%d, zone=%s[%.0f~%.0f], VOL: N=%d x%.2f, "
        "MIN_VOL_10D=%d, TOP_N=%d, MCAP_MIN=%d",
        params["KD_N"], params["KD_K_SMOOTH"], params["KD_D_PERIOD"], params["KD_CROSS_WINDOW"],
        zone_flag, params["KD_ZONE_LOW"], params["KD_ZONE_HIGH"],
        params["VOLUME_LOOKBACK"], params["VOLUME_MULTIPLIER"],
        params["MIN_DAILY_VOLUME_10D"], params["TOP_N"], int(params["MARKET_CAP_MIN"])
    )
    return params


# ----------------- 工具 -----------------
def today_str_tpe() -> str:
    tz = pytz.timezone("Asia/Taipei")
    return datetime.now(tz).strftime("%Y%m%d")


def is_non_trading_today() -> bool:
    """
    用 2330.TW 判斷是否有今日日K。收盤前執行通常無今日K，會判非交易日。
    """
    try:
        tz = pytz.timezone("Asia/Taipei")
        today_tpe = datetime.now(tz).date()
        m = download_ohlcv_batches(
            tickers=["2330.TW"],
            period="1mo",
            interval="1d",
            batch_size=1,
            retries=1,
            sleep_sec=0.5
        )
        df = m.get("2330.TW")
        if df is None or df.empty:
            return True
        last_date = pd.to_datetime(df.index[-1]).date()
        return last_date != today_tpe
    except Exception as e:
        logger.warning(f"Trading-day check failed: {e}")
        return False  # 檢查失敗時當交易日處理，避免誤判


def build_universe(params: Dict) -> pd.DataFrame:
    """
    回傳：code, name, exchange, yahoo
    exchange: 'TWSE' or 'TPEX'；yahoo: .TW / .TWO
    """
    df = fetch_twse_listed_equities()  # 需回傳 exchange 欄位（TWSE/TPEX）
    df = df.copy()

    def to_yahoo(row):
        code = str(row["code"]).zfill(4)
        ex = str(row.get("exchange", "")).upper()
        if ex in ("TPEX", "OTC", "TWO"):
            return f"{code}.TWO"
        return f"{code}.TW"

    df["yahoo"] = df.apply(to_yahoo, axis=1)
    return df[["code", "name", "exchange", "yahoo"]]


def golden_cross_recent(K: pd.Series,
                        D: pd.Series,
                        window: int,
                        require_zone: bool,
                        zone_low: float,
                        zone_high: float) -> Optional[int]:
    """
    近 window 日內，存在某日 (K_prev <= D_prev) 且 (K_curr > D_curr)，
    目前 K > D；若 require_zone=True，交叉日 K/D 要落在區間內。
    回傳交叉日索引（相對 series），否則 None。
    """
    n = len(K)
    if n < 2:
        return None
    start = max(1, n - window)
    last_k, last_d = K.iloc[-1], D.iloc[-1]
    if not (pd.notna(last_k) and pd.notna(last_d) and (last_k > last_d)):
        return None

    for i in range(start, n):
        k_prev, d_prev = K.iloc[i-1], D.iloc[i-1]
        k_curr, d_curr = K.iloc[i], D.iloc[i]
        if np.isnan([k_prev, d_prev, k_curr, d_curr]).any():
            continue
        crossed = (k_prev <= d_prev) and (k_curr > d_curr)
        if not crossed:
            continue
        if require_zone:
            if not (zone_low <= k_curr <= zone_high and zone_low <= d_curr <= zone_high):
                continue
        return i
    return None


def volume_today_over_ma(volume: pd.Series, lookback: int, multiplier: float):
    """今日量 / 過去 lookback 日均量（不含今日） >= multiplier"""
    if len(volume) < lookback + 1:
        return False, np.nan, np.nan, np.nan
    v_today = float(volume.iloc[-1])
    v_ref = float(pd.Series(volume.iloc[-(lookback+1):-1]).mean())
    if not (math.isfinite(v_today) and math.isfinite(v_ref)) or v_ref <= 0:
        return False, np.nan, v_today, v_ref
    ratio = v_today / v_ref
    return (ratio >= multiplier), ratio, v_today, v_ref


def liquidity_10d_ok(volume: pd.Series, min_share: int) -> bool:
    """最近10日每日成交量皆 >= min_share"""
    if len(volume) < 10:
        return False
    last10 = volume.iloc[-10:]
    return bool((last10 >= min_share).all())


def compute_streak_days(code: str, today: datetime.date, lookback_days: int = 14) -> int:
    """
    計算「連續出現天數」：含今天。
    逐日回看 picks_YYYYMMDD.csv，直到缺檔或未出現即中斷。
    """
    streak = 1
    for d in range(1, lookback_days + 1):
        dte = (datetime.combine(today, datetime.min.time()) - timedelta(days=d)).date()
        fname = OUTPUT_DIR / f"picks_{dte.strftime('%Y%m%d')}.csv"
        if not fname.exists():
            break
        try:
            df_prev = pd.read_csv(fname, dtype={"code": str})
            if not (df_prev["code"].astype(str) == str(code)).any():
                break
            streak += 1
        except Exception:
            break
    return streak


# ----------------- 個股海選（不含市值） -----------------
def evaluate_signals_for_ticker(df: pd.DataFrame, params: Dict) -> Optional[Dict]:
    if df is None or df.empty or {"Open","High","Low","Close","Volume"} - set(df.columns):
        return None

    min_len = max(
        params["VOLUME_LOOKBACK"] + 1,
        params["KD_N"] + params["KD_K_SMOOTH"] + params["KD_D_PERIOD"] + 5,
        30
    )
    if len(df) < min_len:
        return None

    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    v = df["Volume"].fillna(0).astype(float)
    if v.sum() == 0:
        return None

    ma5 = c.rolling(5).mean()
    ma10 = c.rolling(10).mean()
    ma20 = c.rolling(20).mean()

    K, D = stochastic_kd(h, l, c,
                         n=params["KD_N"],
                         k_smooth=params["KD_K_SMOOTH"],
                         d_period=params["KD_D_PERIOD"])
    if K is None or D is None or len(K) != len(c):
        return None

    # KD 黃金交叉（近 N 日）且目前 K > D（必要）
    cross_idx = golden_cross_recent(
        K, D,
        window=params["KD_CROSS_WINDOW"],
        require_zone=params["KD_REQUIRE_ZONE"],
        zone_low=params["KD_ZONE_LOW"],
        zone_high=params["KD_ZONE_HIGH"]
    )
    if cross_idx is None:
        return None

    # 放量：今日量 >= 20MA量 * 倍數
    vol_ok, vol_ratio, v_today, v20_avg = volume_today_over_ma(
        v, params["VOLUME_LOOKBACK"], params["VOLUME_MULTIPLIER"]
    )
    if not vol_ok:
        return None

    # 10日流動性：最近10日每日成交量皆 >= 100萬股
    if not liquidity_10d_ok(v, params["MIN_DAILY_VOLUME_10D"]):
        return None

    # 多頭排列：MA5 > MA20
    if params["ENABLE_RULE_MA5_GT_MA20"]:
        if not (ma5.iloc[-1] > ma20.iloc[-1]):
            return None

    # MA20 保護：當日 O 或 C >= MA20
    if params["ENABLE_RULE_OC_ABOVE_MA20"]:
        if not ( (o.iloc[-1] >= ma20.iloc[-1]) or (c.iloc[-1] >= ma20.iloc[-1]) ):
            return None

    # MA10 穩健度：近5日，收盤 < MA10 的天數 <= 門檻
    if params["ENABLE_RULE_LAST5_MA10_THRESHOLD"]:
        last5 = c.iloc[-5:]
        last5_ma10 = ma10.iloc[-5:]
        if last5_ma10.isna().any():
            return None
        days_below = int((last5 < last5_ma10).sum())
        if days_below > params["MAX_DAYS_BELOW_MA10_IN_5"]:
            return None

    # 黑K 限制：若 C < O，則 C >= O * 0.95
    if params["ENABLE_RULE_BLACK_CANDLE_LIMIT"]:
        if c.iloc[-1] < o.iloc[-1]:
            if c.iloc[-1] < (o.iloc[-1] * params["BLACK_CANDLE_MAX_DROP"]):
                return None

    return dict(
        date=pd.to_datetime(df.index[-1]).date().isoformat(),
        close=float(c.iloc[-1]),
        K=float(K.iloc[-1]) if pd.notna(K.iloc[-1]) else np.nan,
        D=float(D.iloc[-1]) if pd.notna(D.iloc[-1]) else np.nan,
        k_d_spread=float(K.iloc[-1] - D.iloc[-1]) if pd.notna(K.iloc[-1]) and pd.notna(D.iloc[-1]) else np.nan,
        vol_ratio=float(vol_ratio),
        ma20=float(ma20.iloc[-1]),
        trend_strength=float((c.iloc[-1] - ma20.iloc[-1]) / ma20.iloc[-1]) if ma20.iloc[-1] and not np.isnan(ma20.iloc[-1]) else np.nan,
        cross_day=pd.to_datetime(df.index[cross_idx]).date().isoformat()
    )


def format_summary_line(code: str, name: str, r: pd.Series, with_streak_icon: bool = True) -> str:
    """摘要：收盤、KD、放量倍數；可選擇是否顯示🔥連N"""
    kd_part = f"K {r['K']:.2f} / D {r['D']:.2f}" if pd.notna(r["K"]) and pd.notna(r["D"]) else "KD N/A"
    vol_part = f"{r['vol_ratio']:.2f}x" if pd.notna(r["vol_ratio"]) else "N/A"
    price_part = f"{r['close']:.2f}"
    streak_tag = ""
    if with_streak_icon and (("streak_days" in r) and pd.notna(r["streak_days"]) and int(r["streak_days"]) >= 2):
        streak_tag = f" 🔥連{int(r['streak_days'])}"
    return f"{code} {name}{streak_tag} | 收盤 {price_part} | {kd_part} | 放量 {vol_part}"


# ----------------- 主流程 -----------------
def run_once():
    params = get_env_params()

    # 非交易日
    if is_non_trading_today():
        msg = "今日為非交易日，請開心過好每一天"
        bot, chat = params.get("TELEGRAM_BOT_TOKEN"), params.get("TELEGRAM_CHAT_ID")
        if bot and chat:
            try:
                send_telegram_message(bot, chat, msg)
                logger.info("Sent non-trading-day notice via Telegram.")
            except Exception as e:
                logger.warning(f"Telegram notice failed: {e}")
        else:
            logger.info("Telegram not configured; skip non-trading-day notice.")
        return

    # 建立市場清單
    logger.info("Building universe from TWSE/TPEX ISIN page...")
    uni = build_universe(params)
    logger.info(f"Universe size: {len(uni)}")

    # 下載 OHLCV
    tickers = uni["yahoo"].tolist()
    code_map = dict(zip(uni["yahoo"], uni["code"]))
    name_map = dict(zip(uni["yahoo"], uni["name"]))

    logger.info("Downloading OHLCV from Yahoo in batches...")
    data_map = download_ohlcv_batches(
        tickers,
        period="6mo",
        interval="1d",
        batch_size=params["BATCH_SIZE"],
        retries=2,
        sleep_sec=1.0
    )

    # 海選（不含市值）
    prelim_rows = []
    for ysym, df in data_map.items():
        try:
            sig = evaluate_signals_for_ticker(df, params)
            if sig:
                prelim_rows.append({
                    "ysym": ysym,
                    "code": str(code_map.get(ysym, "")),
                    "name": name_map.get(ysym, ""),
                    **sig
                })
        except Exception as e:
            logger.warning(f"Signal evaluation failed for {ysym}: {e}")

    today = today_str_tpe()

    if not prelim_rows:
        logger.info("No candidates after pre-screen. Saving empty CSV and notifying...")
        out_path = OUTPUT_DIR / f"picks_{today}.csv"
        pd.DataFrame(columns=["date","code","name","close","K","D","vol_ratio","cross_day","market_cap",
                              "k_d_spread","trend_strength","rank_k","rank_v","a","b","score","streak_days"]
                     ).to_csv(out_path, index=False, encoding="utf-8-sig")
        bot, chat = params.get("TELEGRAM_BOT_TOKEN"), params.get("TELEGRAM_CHAT_ID")
        if bot and chat:
            try:
                send_telegram_message(bot, chat, f"[KD Screener] {today} 今日無符合條件之個股。")
            except Exception as e:
                logger.warning(f"Telegram send failed: {e}")
        return

    df_cand = pd.DataFrame(prelim_rows)

    # 市值過濾
    cand_syms = df_cand["ysym"].dropna().unique().tolist()
    logger.info(f"Fetching market caps from Yahoo for {len(cand_syms)} candidates...")
    mc_map = get_market_caps(cand_syms, retries=1, sleep=0.05)
    df_cand["market_cap"] = df_cand["ysym"].map(mc_map).astype(float)
    df_cand = df_cand[df_cand["market_cap"] >= params["MARKET_CAP_MIN"]].copy()

    if df_cand.empty:
        logger.info("No candidates after market cap filter. Saving empty CSV and notifying...")
        out_path = OUTPUT_DIR / f"picks_{today}.csv"
        pd.DataFrame(columns=["date","code","name","close","K","D","vol_ratio","cross_day","market_cap",
                              "k_d_spread","trend_strength","rank_k","rank_v","a","b","score","streak_days"]
                     ).to_csv(out_path, index=False, encoding="utf-8-sig")
        bot, chat = params.get("TELEGRAM_BOT_TOKEN"), params.get("TELEGRAM_CHAT_ID")
        if bot and chat:
            try:
                send_telegram_message(bot, chat, f"[KD Screener] {today} 今日無符合條件之個股。")
            except Exception as e:
                logger.warning(f"Telegram send failed: {e}")
        return

    # 排名因子與分數
    df_cand["rank_k"] = df_cand["k_d_spread"].rank(method="first", ascending=False)
    df_cand["rank_v"] = df_cand["vol_ratio"].rank(method="first", ascending=False)
    df_cand["a"] = 2 - 0.02 * df_cand["rank_k"]
    df_cand["b"] = 2 - 0.02 * df_cand["rank_v"]
    df_cand["score"] = df_cand["a"] * df_cand["b"] * df_cand["trend_strength"]
    df_cand.sort_values(["score", "vol_ratio", "k_d_spread"], ascending=False, inplace=True)

    # 連續出現標記
    tz = pytz.timezone("Asia/Taipei")
    today_date = datetime.now(tz).date()
    df_cand["streak_days"] = df_cand["code"].apply(lambda x: compute_streak_days(str(x), today_date))

    # 只留 TOP_N
    topn = max(1, int(params["TOP_N"]))
    df_out = df_cand.head(topn).copy()

    # 存檔（單一 CSV 保留）
    out_path = OUTPUT_DIR / f"picks_{today}.csv"
    cols = ["date","code","name","close","K","D","vol_ratio","cross_day","market_cap",
            "k_d_spread","trend_strength","rank_k","rank_v","a","b","score","streak_days"]
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved results to {out_path} (count={len(df_out)})")

    # ===== Telegram：分成兩個表區塊 =====
    bot, chat = params.get("TELEGRAM_BOT_TOKEN"), params.get("TELEGRAM_CHAT_ID")
    if bot and chat:
        try:
            if df_out.empty:
                text = f"[KD Screener] {today} 今日無符合條件之個股。"
            else:
                header = f"[TWSE/TPEX KD Screener] {today} 前{len(df_out)}名（依 score）"
                df_hot = df_out[df_out["streak_days"] >= 2].copy()
                df_new = df_out[df_out["streak_days"] < 2].copy()

                lines = [header]

                # 表一：連續出現（>=2日）
                if not df_hot.empty:
                    lines.append("")
                    lines.append("🔥 連續出現（≥2日）")
                    for _, r in df_hot.iterrows():
                        lines.append(" - " + format_summary_line(str(r["code"]), str(r["name"]), r, with_streak_icon=True))
                else:
                    lines.append("")
                    lines.append("🔥 連續出現（≥2日）：無")

                # 表二：非連續（今日首次或中斷）
                if not df_new.empty:
                    lines.append("")
                    lines.append("— 非連續（今日首次或中斷） —")
                    for _, r in df_new.iterrows():
                        # 非連續段不顯示🔥標籤
                        lines.append(" - " + format_summary_line(str(r["code"]), str(r["name"]), r, with_streak_icon=False))
                else:
                    lines.append("")
                    lines.append("— 非連續（今日首次或中斷） — 無")

                text = "\n".join(lines)

            send_telegram_message(bot, chat, text)
            logger.info("Telegram sent.")
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")
    else:
        logger.info("Telegram not configured; skip Telegram sending.")


if __name__ == "__main__":
    try:
        run_once()
    except Exception as e:
        logger.error(f"Fatal error: {e}\n{traceback.format_exc()}")
        try:
            load_dotenv(ROOT / ".env", override=True)
            bot = os.getenv("TELEGRAM_BOT_TOKEN")
            chat = os.getenv("TELEGRAM_CHAT_ID")
            if bot and chat:
                send_telegram_message(bot, chat, f"❌ Screener 例外：{e}")
        except Exception:
            pass
        sys.exit(1)
