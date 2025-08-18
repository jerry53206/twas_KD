select# -*- coding: utf-8 -*-
"""
twas_KD 主程式（無需 TA-Lib 版本）

功能概述
1) 取得今日(Asia/Taipei)是否有最新收盤資料（以 2330.TW 判斷）。若無：傳訊息「今日為非交易日…」並結束。
2) 下載股池歷史資料，計算 KD(14,3,3) 與 MA20，選出入選清單（條件見 RULES）。
3) 以「上一輪成功執行的交易日」判定是否連續出現（支援隔週/連假/漏跑）。
4) 產出兩張表：連續≥2天 與 非連續，輸出 CSV；Telegram 傳摘要並附檔。

環境變數
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID

檔案
- universe.csv：可選。一行一個股票代碼（不含 .TW / .TWO）
- state/streaks.json：每檔最後入選日與連續天數
- state/last_run.json：上一輪成功執行的交易日
- output/*.csv：輸出結果
"""
from __future__ import annotations

import os
import io
import json
import sys
import time
import math
import shutil
import zipfile
import logging
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

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

# 時區（避免依賴外部套件，統一用台北日期計算）
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
    """以 UTC 時間加上 +8 取得台北當地「日」"""
    return (datetime.utcnow() + timedelta(hours=TZ_OFFSET_HOURS)).date()

def y_to_tw_symbol(code: str) -> str:
    """將 4 位數台股代碼轉為 yfinance 代碼（上市 .TW、上櫃 .TWO 的判斷這裡一律 .TW；若你有上櫃請自行在 universe.csv 加上 .TWO 完整代碼）"""
    # 若使用者自己在 CSV 給了 .TW/.TWO，就尊重之
    if code.endswith(".TW") or code.endswith(".TWO"):
        return code
    # 簡化：預設為上市 .TW；你可依需求自訂映射表
    return f"{code}.TW"

def safe_read_universe() -> List[str]:
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
    """抓指定標的最近一筆收盤日期"""
    try:
        df = yf.download(symbol, period=f"{period_days}d", interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty:
            return None
        last_ts = df.index[-1]
        # yfinance 回傳 tz-aware DatetimeIndex -> 轉為台北日期
        # 這裡直接視為 UTC，再 +8h 取日期（實務足夠）
        if hasattr(last_ts, "to_pydatetime"):
            last_dt = last_ts.to_pydatetime()
        else:
            last_dt = pd.to_datetime(last_ts).to_pydatetime()
        last_dt_tpe = last_dt + timedelta(hours=TZ_OFFSET_HOURS)
        return last_dt_tpe.date()
    except Exception as e:
        logger.warning("取 %s 最近收盤日期失敗：%s", symbol, e)
        return None

def fetch_history(symbols: List[str], lookback_days: int = 180) -> Dict[str, pd.DataFrame]:
    """以 yfinance 抓多檔日線"""
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
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    lowest_low = low.rolling(window=n, min_periods=n).min()
    highest_high = high.rolling(window=n, min_periods=n).max()
    denom = (highest_high - lowest_low)

    # 避免分母為 0（盤整區間最高=最低）：令 RSV=0.5
    rsv = pd.Series(np.where(denom == 0, 0.5, (close - lowest_low) / denom), index=close.index)
    rsv = pd.Series(np.clip(rsv, 0.0, 1.0), index=close.index)

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
    obj = load_json(LAST_RUN_PATH)
    return obj.get("last_trade_date")

def save_last_run_date(trade_date: date) -> None:
    save_json(LAST_RUN_PATH, {"last_trade_date": trade_date.isoformat()})

# ---------- 通知 ----------
def tg_send_message(text: str, parse_mode: Optional[str] = "Markdown") -> None:
    if not TG_TOKEN or not TG_CHAT_ID:
        logger.info("未設定 Telegram 環境變數，略過傳訊息。")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": text,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
        payload["disable_web_page_preview"] = True
    try:
        r = requests.post(url, data=payload, timeout=20)
        if r.status_code != 200:
            logger.warning("Telegram sendMessage 失敗：%s %s", r.status_code, r.text)
    except Exception as e:
        logger.warning("Telegram 傳訊息錯誤：%s", e)

def tg_send_document(file_path: Path, caption: Optional[str] = None) -> None:
    if not TG_TOKEN or not TG_CHAT_ID:
        logger.info("未設定 Telegram 環境變數，略過傳檔。")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendDocument"
    files = {"document": (file_path.name, open(file_path, "rb"))}
    data = {"chat_id": TG_CHAT_ID}
    if caption:
        data["caption"] = caption
    try:
        r = requests.post(url, data=data, files=files, timeout=60)
        if r.status_code != 200:
            logger.warning("Telegram sendDocument 失敗：%s %s", r.status_code, r.text)
    except Exception as e:
        logger.warning("Telegram 傳檔錯誤：%s", e)

# ---------- 規則（可依需求調整） ----------
@dataclass
class RuleParams:
    kd_n: int = 14
    kd_k: int = 3
    kd_d: int = 3
    ma_n: int = 20
    # 入選條件：KD 黃金交叉且 K、D < 50，並且收盤 > MA20
    # 這裡提供一個合理的預設，你可微調門檻
    max_kd_level: float = 50.0

def select_candidates(hist: Dict[str, pd.DataFrame], params: RuleParams) -> pd.DataFrame:
    rows = []
    for code, df in hist.items():
        if df.shape[0] < max(params.kd_n, params.ma_n) + 5:
            continue
        df2 = calc_kd(df, n=params.kd_n, k_smooth=params.kd_k, d_smooth=params.kd_d)
        df2["ma"] = moving_avg(df2["close"], params.ma_n)
        if df2["ma"].isna().all():
            continue

        tail = df2.dropna().iloc[-2:]
        if tail.shape[0] < 2:
            continue
        prev_row = tail.iloc[0]
        last_row = tail.iloc[1]

        # --- 關鍵修正：先取純量再做比較，避免產生 Series 布林 ---
        k_prev = float(prev_row["K"])
        d_prev = float(prev_row["D"])
        k_last = float(last_row["K"])
        d_last = float(last_row["D"])
        close_last = float(last_row["close"])
        ma_last = float(last_row["ma"])

        golden_cross = (k_prev <= d_prev) and (k_last > d_last)
        kd_ok = (k_last < params.max_kd_level) and (d_last < params.max_kd_level)
        ma_ok = close_last > ma_last

        if golden_cross and kd_ok and ma_ok:
            rows.append({
                "code": code,
                "date": last_row.name.date().isoformat(),
                "close": round(close_last, 2),
                "K": round(k_last, 2),
                "D": round(d_last, 2),
                "MA20": round(ma_last, 2),
            })

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values(by=["code"]).reset_index(drop=True)
    return df_out

# ---------- 主流程 ----------
def main():
    tpe_today = today_taipei()
    logger.info("台北日期：%s", tpe_today.isoformat())

    # 1) 判斷是否為交易日（以 2330.TW 最新收盤日期是否等於今日）
    last_tsmc_date = fetch_last_trade_date_of("2330.TW", period_days=10)
    if last_tsmc_date != tpe_today:
        msg = f"【{APP_NAME}】{tpe_today.isoformat()} 今日為非交易日，請開心過好每一天。"
        logger.info(msg)
        tg_send_message(msg)
        return

    # 2) 下載股池歷史資料
    universe = safe_read_universe()
    hist = fetch_history(universe, lookback_days=220)

    if not hist:
        warn = f"【{APP_NAME}】{tpe_today.isoformat()} 今日資料抓取失敗（股池無資料）。"
        logger.warning(warn)
        tg_send_message(warn)
        return

    # 3) 規則選股
    params = RuleParams()
    df_top = select_candidates(hist, params)

    if df_top.empty:
        note = f"【{APP_NAME}】{tpe_today.isoformat()} 今日無入選標的。"
        logger.info(note)
        tg_send_message(note)
        # 沒有入選清單就不更新 last_run.json（避免中斷基準）
        return

    # 4) 連續出現（以上一輪「成功產出」的交易日作為連續判定基準）
    prev_run_trade_date_str = load_last_run_date()
    streaks = load_streaks()  # { code: {"last_date": "YYYY-MM-DD", "streak": int} }
    updated = {}
    cont_days = []

    for _, r in df_top.iterrows():
        code = str(r["code"])
        prev = streaks.get(code)
        if prev and prev_run_trade_date_str and prev.get("last_date") == prev_run_trade_date_str:
            days = int(prev.get("streak", 1)) + 1
        else:
            days = 1
        cont_days.append(days)
        updated[code] = {"last_date": tpe_today.isoformat(), "streak": days}

    df_top["continuation_days"] = cont_days
    save_streaks(updated)  # 只保留今日入選的檔，避免無限膨脹

    # 5) 兩張表：連續 >=2 與 非連續
    df_cont = df_top[df_top["continuation_days"] >= 2].copy()
    df_single = df_top[df_top["continuation_days"] == 1].copy()

    # 6) 輸出 CSV
    cont_path = OUTPUT_DIR / f"continuous_{tpe_today.isoformat()}.csv"
    single_path = OUTPUT_DIR / f"non_continuous_{tpe_today.isoformat()}.csv"
    df_cont.to_csv(cont_path, index=False, encoding="utf-8-sig")
    df_single.to_csv(single_path, index=False, encoding="utf-8-sig")

    # 7) 更新 last_run.json（只在本輪有入選清單時才更新）
    save_last_run_date(tpe_today)

    # 8) 傳送 Telegram：摘要 + 附檔
    def fmt_table(df: pd.DataFrame, title: str) -> str:
        if df.empty:
            return f"*{title}*: 無"
        rows = [f"*{title}*（{len(df)} 檔）"]
        for _, rr in df.iterrows():
            rows.append(
                f"`{rr['code']}`  收盤 {rr['close']:.2f}｜K/D {rr['K']:.1f}/{rr['D']:.1f}｜MA20 {rr['MA20']:.2f}｜連續 {rr['continuation_days']} 天"
            )
        return "\n".join(rows)

    header = f"【{APP_NAME}】{tpe_today.isoformat()} 入選結果\n"
    body = "\n\n".join([
        fmt_table(df_cont, "連續兩天以上"),
        fmt_table(df_single, "無連續出現"),
    ])
    tg_send_message(header + body)

    # 附檔（CSV）
    tg_send_document(cont_path, caption=f"{APP_NAME} 連續兩天以上")
    tg_send_document(single_path, caption=f"{APP_NAME} 無連續出現")

    logger.info("完成。連續≥2: %d，非連續: %d", len(df_cont), len(df_single))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("程式發生例外：%s", e)
        # 發送錯誤通知（可選）
        try:
            tg_send_message(f"【{APP_NAME}】執行錯誤：{e}")
        except Exception:
            pass
        sys.exit(1)
