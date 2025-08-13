#!/usr/bin/env python3
import os
import sys
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from universe.twse_listed import fetch_twse_listed_equities
from data_sources.yahoo import download_ohlcv_batches
from data_sources.yahoo_meta import get_market_caps
from indicators.ta import stochastic_kd
from filters.conditions import (
    golden_cross_in_window,
    golden_cross_recent_no_zone,
    volume_spike,
)
from notify.emailer import send_email_with_attachments

# --------- Config & Logging ---------
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

# --------- Utilities ---------
def bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y", "on")

def send_telegram_message(token: Optional[str], chat_id: Optional[str], text: str) -> None:
    if not token or not chat_id:
        logger.info("Telegram not configured; skip Telegram sending.")
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, data={"chat_id": chat_id, "text": text})
        if resp.status_code != 200:
            logger.error(f"Telegram send failed: {resp.status_code} {resp.text}")
        else:
            logger.info("Telegram message sent.")
    except Exception as e:
        logger.error(f"Telegram sending exception: {e}")

# --------- Parameters ---------
def get_env_params():
    # 從 repo 根目錄的 .env 讀取，Actions 會在前一步把 secrets 寫進這個檔案
    load_dotenv(ROOT / ".env", override=True)
    params = dict(
        KD_N=int(os.getenv("KD_N", "9")),
        KD_K_SMOOTH=int(os.getenv("KD_K_SMOOTH", "3")),
        KD_D_PERIOD=int(os.getenv("KD_D_PERIOD", "3")),
        KD_ZONE_LOW=float(os.getenv("KD_ZONE_LOW", "40")),
        KD_ZONE_HIGH=float(os.getenv("KD_ZONE_HIGH", "80")),
        KD_CROSS_WINDOW=int(os.getenv("KD_CROSS_WINDOW", "3")),
        KD_REQUIRE_ZONE=os.getenv("KD_REQUIRE_ZONE", "false").lower() == "true",

        VOLUME_LOOKBACK=int(os.getenv("VOLUME_LOOKBACK", "20")),
        VOLUME_MULTIPLIER=float(os.getenv("VOLUME_MULTIPLIER", "1.5")),

        ENABLE_RULE_BLACK_CANDLE_LIMIT=os.getenv("ENABLE_RULE_BLACK_CANDLE_LIMIT", "true").lower() == "true",
        BLACK_CANDLE_MAX_DROP=float(os.getenv("BLACK_CANDLE_MAX_DROP", "0.95")),
        ENABLE_RULE_OC_ABOVE_MA20=os.getenv("ENABLE_RULE_OC_ABOVE_MA20", "true").lower() == "true",
        ENABLE_RULE_LAST5_MA10_THRESHOLD=os.getenv("ENABLE_RULE_LAST5_MA10_THRESHOLD", "true").lower() == "true",
        MAX_DAYS_BELOW_MA10_IN_5=int(os.getenv("MAX_DAYS_BELOW_MA10_IN_5", "3")),
        ENABLE_RULE_MA5_GT_MA20=os.getenv("ENABLE_RULE_MA5_GT_MA20", "true").lower() == "true",

        MARKET_CAP_MIN=float(os.getenv("MARKET_CAP_MIN", "10000000000")),
        BATCH_SIZE=int(os.getenv("BATCH_SIZE", "120")),
        TOP_N=int(os.getenv("TOP_N", "20")),
        INCLUDE_TPEX=os.getenv("INCLUDE_TPEX", "false").lower() == "true",

        # ✅ 從環境變數讀，不要把 token/ID 寫死在程式裡
        TELEGRAM_BOT_TOKEN=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        TELEGRAM_CHAT_ID=os.getenv("TELEGRAM_CHAT_ID", ""),

        # 若你還保留 Email，可一併讀進來（沒有就留空）
        SMTP_HOST=os.getenv("SMTP_HOST", ""),
        SMTP_PORT=int(os.getenv("SMTP_PORT", "587")),
        SMTP_USE_TLS=os.getenv("SMTP_USE_TLS", "true").lower() == "true",
        SMTP_USER=os.getenv("SMTP_USER", ""),
        SMTP_PASS=os.getenv("SMTP_PASS", ""),
        SENDER_EMAIL=os.getenv("SENDER_EMAIL", ""),
        RECIPIENT_EMAIL=os.getenv("RECIPIENT_EMAIL", ""),
    )
    return params

# --------- Universe ---------
def build_universe(params) -> pd.DataFrame:
    """
    回傳欄位: code, name, yahoo
    - TWSE: 加 .TW
    - TPEX: 加 .TWO（當 INCLUDE_TPEX=true 時）
    """
    frames = []

    twse = fetch_twse_listed_equities()
    twse["yahoo"] = twse["code"].astype(str).str.zfill(4) + ".TW"
    frames.append(twse[["code", "name", "yahoo"]])

    if params.get("INCLUDE_TPEX", False):
        tpex = fetch_tpex_listed_equities()
        tpex["yahoo"] = tpex["code"].astype(str).str.zfill(4) + ".TWO"
        frames.append(tpex[["code", "name", "yahoo"]])

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("code").reset_index(drop=True)
    return df

# --------- Signal Engine ---------
def evaluate_signals_for_ticker(df: pd.DataFrame, params: Dict) -> Optional[Dict]:
    """
    技術面檢查（不含市值）：KD + 量能(今日/20MA量) + 價格/均線四規則
    回傳 dict 或 None
    """
    if df is None or df.empty:
        return None
    if df["Volume"].fillna(0).sum() == 0:
        return None

    open_ = pd.to_numeric(df["Open"], errors="coerce")
    high  = pd.to_numeric(df["High"], errors="coerce")
    low   = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    vol   = pd.to_numeric(df["Volume"], errors="coerce")

    # KD
    K, D = stochastic_kd(
        high, low, close,
        n=params.get("KD_N", 9),
        k_smooth=params.get("KD_K_SMOOTH", 3),
        d_period=params.get("KD_D_PERIOD", 3),
    )
    if K is None or D is None:
        return None

    # 至少要能計到 20MA 與量能參數
    need_len = max(
        20,
        params.get("VOLUME_LOOKBACK", 20) + max(30, params.get("KD_N", 9) + params.get("KD_K_SMOOTH", 3) + params.get("KD_D_PERIOD", 3))
    )
    if len(df) < need_len:
        return None

    # KD 條件
    if params.get("KD_REQUIRE_ZONE", False):
        cross = golden_cross_in_window(
            K, D,
            window=params.get("KD_CROSS_WINDOW", 3),
            zone_low=params.get("KD_ZONE_LOW", 40.0),
            zone_high=params.get("KD_ZONE_HIGH", 80.0),
            require_both_in_zone=True,
        )
        if cross is None:
            return None
    else:
        cross = golden_cross_recent_no_zone(K, D, window=params.get("KD_CROSS_WINDOW", 3))
        if cross is None or not (K.iloc[-1] > D.iloc[-1]):
            return None

    # 均線（Close 基礎）
    ma5  = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    if any(np.isnan([ma5.iloc[-1], ma10.iloc[-1], ma20.iloc[-1]])):
        return None

    # (1) 黑K跌幅限制
    if params.get("ENABLE_RULE_BLACK_CANDLE_LIMIT", True) and close.iloc[-1] < open_.iloc[-1]:
        if close.iloc[-1] < params.get("BLACK_CANDLE_MAX_DROP", 0.95) * open_.iloc[-1]:
            return None

    # (2) 開或收 至少一者 >= 20MA
    if params.get("ENABLE_RULE_OC_ABOVE_MA20", True):
        if not (open_.iloc[-1] >= ma20.iloc[-1] or close.iloc[-1] >= ma20.iloc[-1]):
            return None

    # (3) 近5日 Close < 10MA 的天數 ≤ 門檻
    if params.get("ENABLE_RULE_LAST5_MA10_THRESHOLD", True):
        last5_below10 = (close.iloc[-5:] < ma10.iloc[-5:]).sum()
        if last5_below10 > params.get("MAX_DAYS_BELOW_MA10_IN_5", 3):
            return None

    # (4) 5MA > 20MA
    if params.get("ENABLE_RULE_MA5_GT_MA20", True):
        if not (ma5.iloc[-1] > ma20.iloc[-1]):
            return None

    # 量能：今日 / 過去 N 日均量 >= 倍數（含等於）
    vol_ok, ratio = volume_spike(
        vol,
        lookback=params.get("VOLUME_LOOKBACK", 20),
        multiplier=params.get("VOLUME_MULTIPLIER", 1.5),
    )
    if not vol_ok:
        return None

    last_idx = df.index[-1]
    return {
        "date": pd.to_datetime(last_idx).date().isoformat(),
        "close": float(close.iloc[-1]),
        "K": float(K.iloc[-1]) if not np.isnan(K.iloc[-1]) else np.nan,
        "D": float(D.iloc[-1]) if not np.isnan(D.iloc[-1]) else np.nan,
        "vol_ratio": float(ratio),
        "cross_day": pd.to_datetime(df.index[cross]).date().isoformat(),
    }

# --------- Main ---------
def run_once():
    params = get_env_params()
    logger.info("Building universe from TWSE/TPEX ISIN page...")
    uni = build_universe(params)
    logger.info(f"Universe size: {len(uni)}")

    tickers = uni["yahoo"].tolist()
    code_map = dict(zip(uni["yahoo"], uni["code"]))
    name_map = dict(zip(uni["yahoo"], uni["name"]))

    logger.info("Downloading OHLCV from Yahoo in batches...")
    data_map = download_ohlcv_batches(
        tickers,
        period="6mo",
        interval="1d",
        batch_size=params.get("BATCH_SIZE", 120),
        retries=2,
        sleep_sec=1.0,
    )

    # 先做技術條件篩；等候選出來再抓市值，節省時間
    prelim_rows = []
    prelim_syms: List[str] = []
    for ysym, df in data_map.items():
        try:
            sig = evaluate_signals_for_ticker(df, params)
            if sig:
                prelim_rows.append({
                    "date": sig["date"],
                    "code": code_map.get(ysym, ""),
                    "name": name_map.get(ysym, ""),
                    "close": sig["close"],
                    "K": sig["K"],
                    "D": sig["D"],
                    "vol_ratio": sig["vol_ratio"],
                    "cross_day": sig["cross_day"],
                    "yahoo": ysym,
                })
                prelim_syms.append(ysym)
        except Exception as e:
            logger.warning(f"Signal evaluation failed for {ysym}: {e}")

    # 市值過濾（若有門檻）
    rows = []
    if prelim_rows:
        mc_min = params.get("MARKET_CAP_MIN", 0.0) or 0.0
        if mc_min > 0:
            logger.info(f"Fetching market caps from Yahoo for {len(prelim_syms)} candidates...")
            mc_map = get_market_caps(prelim_syms, retries=1, sleep=0.05)
        else:
            mc_map = {}

        for r in prelim_rows:
            mcap = mc_map.get(r["yahoo"])
            if mcap is None and mc_min > 0:
                # 沒拿到市值且有門檻，直接跳過
                continue
            if mc_min > 0 and mcap < mc_min:
                continue
            r["market_cap"] = mcap
            rows.append(r)

    # 輸出 CSV
    if rows:
        df_out = pd.DataFrame(rows).sort_values(["vol_ratio"], ascending=False)
    else:
        df_out = pd.DataFrame(columns=["date","code","name","close","K","D","vol_ratio","cross_day","market_cap"])

    today = datetime.now().strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"picks_{today}.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved results to {out_path} (count={len(df_out)})")

    # === 摘要訊息（只顯示：今日收盤、KD 值、放量倍數），前 TOP_N 名 ===
    top_n = int(params.get("TOP_N", 20))
    lines: List[str] = []
    if len(df_out) == 0:
        lines.append(f"{today} 今日無符合條件之個股。")
    else:
        topn = df_out.head(top_n)
        lines.append(f"{today} 選股結果（前{len(topn)}名）：")
        for _, r in topn.iterrows():
            k_str = f"{r['K']:.2f}" if pd.notna(r['K']) else "NA"
            d_str = f"{r['D']:.2f}" if pd.notna(r['D']) else "NA"
            vr_str = f"{r['vol_ratio']:.2f}x" if pd.notna(r['vol_ratio']) else "NA"
            lines.append(f"{r['code']} {r['name']} | 收 {r['close']:.2f} | K {k_str} / D {d_str} | 放量 {vr_str}")
    body = "\n".join(lines)

    # Telegram
    try:
        send_telegram_message(
            params.get("TELEGRAM_BOT_TOKEN"),
            params.get("TELEGRAM_CHAT_ID"),
            body,
        )
    except Exception as e:
        logger.error(f"Telegram send error: {e}\n{traceback.format_exc()}")

    # Email（可選）
    try:
        if params.get("RECIPIENT_EMAIL"):
            subject = f"[TWSE KD Screener] {today} 選股結果（共 {len(df_out)} 檔）"
            send_email_with_attachments(
                smtp_host=params.get("SMTP_HOST"),
                smtp_port=params.get("SMTP_PORT", 587),
                use_tls=params.get("SMTP_USE_TLS", True),
                username=params.get("SMTP_USER"),
                password=params.get("SMTP_PASS"),
                sender=params.get("SENDER_EMAIL") or params.get("SMTP_USER"),
                recipients=[params.get("RECIPIENT_EMAIL")],
                subject=subject,
                body=body,
                attachments=[str(out_path)],
            )
            logger.info(f"Email sent to {params.get('RECIPIENT_EMAIL')}")
        else:
            logger.info("RECIPIENT_EMAIL not set; skip email sending.")
    except Exception as e:
        logger.error(f"Email sending failed: {e}\n{traceback.format_exc()}")

if __name__ == "__main__":
    run_once()
