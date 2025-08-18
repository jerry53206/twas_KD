# -*- coding: utf-8 -*-
"""
twas_KD 主程式（無需 TA-Lib 版本）

功能概述
1) 以 2330.TW 判斷今日(Asia/Taipei)是否有最新收盤資料：若無 => 傳「今日為非交易日…」並結束。
2) 下載股池歷史資料，計算 KD(14,3,3) 與 MA20，依規則選股。
3) 以「上一輪成功執行的交易日」判定是否連續出現（支援隔週/連假/漏跑不中斷）。
4) 產出兩張表：連續≥2 與 非連續，輸出 CSV；Telegram 傳摘要並附檔。

環境變數
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID

檔案
- universe.csv：可選。一行一個代碼（不含 .TW/.TWO；若要上櫃，建議直接寫完整如 6488.TWO）
- state/streaks.json：每檔最後入選日與連續天數
- state/last_run.json：上一輪成功執行的交易日
- output/*.csv：輸出結果
"""

import os
import json
import sys
import logging
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional

import requests
import numpy as np
import pandas as pd
import yfinance as yf

# ---------- 基本設定 ----------
APP_NAME = "twas_KD"
ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "state"
OUTPUT_DIR = ROOT / "output"
STATE_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# 時區（統一用台北日期計算）
TZ_OFFSET_HOURS = 8  # Asia/Taipei (UTC+8)

# Telegram
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

# 日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(APP_NAME)

# ---------- 工具函式 ----------
def today_taipei() -> date:
    """以 UTC +8 取得台北當地日期"""
    return (datetime.utcnow() + timedelta(hours=TZ_OFFSET_HOURS)).date()

def y_to_tw_symbol(code: str) -> str:
    """
    將 4 位數台股代碼轉為 yfinance 代碼。
    若 CSV 已填 .TW/.TWO 就原樣使用；否則預設附上 .TW（若需上櫃請在 universe.csv 直接寫 6488.TWO）。
    """
    if code.endswith(".TW") or code.endswith(".TWO"):
        return code
    return f"{code}.TW"

def safe_read_universe() -> List[str]:
    """讀取 universe.csv；若不存在則用預設股池"""
    csv_path = ROOT / "universe.csv"
    if csv_path.exists():
        s = []
        for line in csv_path.read_text(encoding="utf-8").splitlines():
            line = line.strip().strip(",")
            if not line:
                continue
            s.append(line)
        if s:
            logger.info("使用 universe.csv：%d 檔", len(s))
            return s
    # 預設股池（可自行調整）
    default = [
        "2330", "2317", "2454", "2308", "2382", "2412", "2303", "2881", "2882", "2884",
        "2886", "2357", "1216", "2885", "3231", "3034", "2379", "6669", "5880", "2383",
        "3661", "3711", "4938", "1590", "1101", "1102", "2603", "2615", "9933", "8046",
    ]
    logger.info("使用預設股池：%d 檔", len(default))
    return default

def fetch_last_trade_date_of(symbol: str, period_days: int = 7) -> Optional[date]:
    """抓指定標的最近一筆收盤日期（轉為台北日期）"""
    try:
        df = yf.download(symbol, period=f"{period_days}d", interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty:
            return None
        last_ts = df.index[-1]
        # 視為 UTC，再 +8h 取日期（對判斷是否為今日足夠）
        last_dt = pd.to_datetime(last_ts).to_pydatetime()
        last_dt_tpe = last_dt + timedelta(hours=TZ_OFFSET_HOURS)
        return last_dt_tpe.date()
    except Exception as e:
        logger.warning("取 %s 最近收盤日期失敗：%s", symbol, e)
        return None

def fetch_history(symbols: List[str], lookback_days: int = 180) -> Dict[str, pd.DataFrame]:
    """以 yfinance 抓多檔日線資料"""
    out = {}
    for s in symbols:
        ys = y_to_tw_symbol(s)
        try:
            df = yf.download(ys, period=f"{lookback_days}d", interval="1d", auto_adjust=False, progress=False)
            if df is None or df.empty:
                logger.info("無資料：%s", ys)
                continue
            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                }
            )
            df.index = pd.to_datetime(df.index)
            out[s] = df
        except Exception as e:
            logger.warning("下載失敗 %s：%s", ys, e)
    return out

def calc_kd(df: pd.DataFrame, n: int = 14, k_smooth: int = 3, d_smooth: int = 3) -> pd.DataFrame:
    """
    計算 Stochastic KD (n, k_smooth, d_smooth)。
    內建處理盤整區間（最高=最低）之分母為 0：令 RSV=0.5，避免 NaN/Inf。
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    lowest_low = low.rolling(window=n, min_periods=n).min()
    highest_high = high.rolling(window=n, min_periods=n).max()
    denom = (highest_high - lowest_low)

    rsv = np.where(denom == 0, 0.5, (close - lowest_low) / denom)
    rsv = np.clip(rsv, 0.0, 1.0)

    rsv = pd.Series(rsv, index=close.index)
    K = rsv.ewm(alpha=1.0 / k_smooth, adjust=False, min_periods=k_smooth).mean() * 100.0
    D = K.ewm(alpha=1.0 / d_smooth, adjust=False, min_periods=d_smooth).mean()

    out = df.copy()
    out["K"] = K
    out["D"] = D
    return out

def moving_avg(series: pd.Series, n: int = 20) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

# ---------- 連續出現狀態 ----------
STREAKS_PATH = STATE_DIR / "streaks.json"
LAST_RUN_PATH = STATE_DIR / "last_run.json"

def load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_json(path: Path, obj: dict) -> None:
    try:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("寫入 %s 失敗：%s", path.name, e)

def load_streaks() -> Dict[str, Dict]:
    return load_json(STREAKS_PATH)

def save_streaks(streaks: Dict[str, Dict]) -> None:
    save_json(STREAKS_PATH, streaks)

def load_last_run_date() -> Optional[str]:
    obj = load_j_
