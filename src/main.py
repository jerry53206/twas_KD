#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import math
import logging
import traceback
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datetime import datetime, date
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # Py>=3.9 皆有，保險處理

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

# ----------------- 基本設定 -----------------
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
LOG_DIR = ROOT / "logs"
STATE_DIR = ROOT / "state"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("tdcc_kd_screener")


# ----------------- 參數 -----------------
def _get_bool(val: Optional[str], default: bool) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "t", "y", "yes"}

def get_env_params() -> dict:
    load_dotenv(ROOT / ".env", override=True)

    params = dict(
        TOP_N=int(os.getenv("TOP_N", "20")),

        KD_N=int(os.getenv("KD_N", "9")),
        KD_K_SMOOTH=int(os.getenv("KD_K_SMOOTH", "3")),
        KD_D_PERIOD=int(os.getenv("KD_D_PERIOD", "3")),
        KD_CROSS_WINDOW=int(os.getenv("KD_CROSS_WINDOW", "3")),
        KD_REQUIRE_ZONE=_get_bool(os.getenv("KD_REQUIRE_ZONE", "false"), False),
        KD_ZONE_LOW=float(os.getenv("KD_ZONE_LOW", "40")),
        KD_ZONE_HIGH=float(os.getenv("KD_ZONE_HIGH", "80")),

        VOLUME_LOOKBACK=int(os.getenv("VOLUME_LOOKBACK", "20")),
        VOLUME_MULTIPLIER=float(os.getenv("VOLUME_MULTIPLIER", "1.5")),
        LIQ_MIN_VOL_LAST10=int(os.getenv("LIQ_MIN_VOL_LAST10", "1000000")),

        ENABLE_RULE_BLACK_CANDLE_LIMIT=_get_bool(os.getenv("ENABLE_RULE_BLACK_CANDLE_LIMIT", "true"), True),
        BLACK_CANDLE_MAX_DROP=float(os.getenv("BLACK_CANDLE_MAX_DROP", "0.95")),
        ENABLE_RULE_OC_ABOVE_MA20=_get_bool(os.getenv("ENABLE_RULE_OC_ABOVE_MA20", "true"), True),
        ENABLE_RULE_LAST5_MA10_THRESHOLD=_get_bool(os.getenv("ENABLE_RULE_LAST5_MA10_THRESHOLD", "true"), True),
        MAX_DAYS_BELOW_MA10_IN_5=int(os.getenv("MAX_DAYS_BELOW_MA10_IN_5", "3")),
        ENABLE_RULE_MA5_GT_MA20=_get_bool(os.getenv("ENABLE_RULE_MA5_GT_MA20", "true"), True),

        MARKET_CAP_MIN=float(os.getenv("MARKET_CAP_MIN", "10000000000")),

        BATCH_SIZE=int(os.getenv("BATCH_SIZE", "120")),
        CONTINUATION_KEY=os.getenv("CONTINUATION_KEY", "yahoo"),
        INCLUDE_TPEX=_get_bool(os.getenv("INCLUDE_TPEX", "true"), True),

        TELEGRAM_BOT_TOKEN=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        TELEGRAM_CHAT_ID=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
    )

    logger.info(
        "Params loaded: KD=(%d,%d,%d), window=%d, zone=%s[%g~%g], "
        "VOL: N=%d x%.2f, LIQ10=%d, TOP_N=%d, MCAP_MIN=%g, TPEX=%s",
        params["KD_N"], params["KD_K_SMOOTH"], params["KD_D_PERIOD"],
        params["KD_CROSS_WINDOW"],
        "ON" if params["KD_REQUIRE_ZONE"] else "OFF",
        params["KD_ZONE_LOW"], params["KD_ZONE_HIGH"],
        params["VOLUME_LOOKBACK"], params["VOLUME_MULTIPLIER"],
        params["LIQ_MIN_VOL_LAST10"],
        params["TOP_N"], params["MARKET_CAP_MIN"],
        params["INCLUDE_TPEX"]
    )
    return params


# ----------------- 上市/上櫃清單 -----------------
EXCLUDE_KEYWORDS = (
    "受益|ETF|基金|購|售|權證|公司債|可轉債|交換債|受益憑證|特別股|存託憑證|不動產|REIT|期貨|選擇權|ETN"
)

def _fetch_isin_table(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=30)
    resp.encoding = resp.apparent_encoding or "utf-8"
    tables = pd.read_html(StringIO(resp.text))
    # 找到含「有價證券代號及名稱」的那張表
    target = None
    for t in tables:
        cols = [str(c) for c in t.columns]
        if any("有價證券代號及名稱" in str(c) for c in cols):
            target = t.copy()
            break
    if target is None:
        raise RuntimeError("ISIN 表格結構可能變動，找不到『有價證券代號及名稱』欄。")
    return target

def _parse_isin_equities(df: pd.DataFrame, exchange: str) -> pd.DataFrame:
    df = df.copy()
    # 一般第一欄是「有價證券代號及名稱」
    name_col = None
    for c in df.columns:
        if "有價證券代號及名稱" in str(c):
            name_col = c
            break
    if name_col is None:
        raise RuntimeError("ISIN 表格結構可能變動，找不到『有價證券代號及名稱』欄。")

    # 去掉表內「產業別」等小節
    df = df[~df[name_col].astype(str).str.contains("產業別|有價證券代號及名稱", na=False)]
    # 只保留「XXXX 名稱」格式
    parts = df[name_col].astype(str).str.split(r"\s+", n=1, regex=True)
    code = parts.str[0]
    name = parts.str[1].fillna("")
    # 過濾非四位數或全數字編碼
    code = code.str.extract(r"^(\d{4})$")[0]
    mask = code.notna() & ~name.str.contains(EXCLUDE_KEYWORDS, na=False)
    out = pd.DataFrame({
        "code": code[mask].str.zfill(4),
        "name": name[mask].str.strip()
    })
    out["exchange"] = exchange
    out = out.dropna(subset=["code"]).drop_duplicates(subset=["code"])
    return out.reset_index(drop=True)

def fetch_twse_tpex_equities(include_tpex: bool = True) -> pd.DataFrame:
    # TWSE
    urls_try = [
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",  # TWSE
    ]
    twse_ok = None
    for u in urls_try:
        try:
            raw = _fetch_isin_table(u)
            twse_ok = _parse_isin_equities(raw, "TWSE")
            break
        except Exception as e:
            logger.warning("TWSE 取數失敗: %s", e)
    if twse_ok is None:
        raise RuntimeError("TWSE ISIN 取得失敗。")

    if not include_tpex:
        return twse_ok

    # TPEX（上櫃）
    tpex_ok = None
    urls_try_tpex = [
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4",
        "https://isin.tpex.org.tw/isin/C_public.jsp?strMode=4",
    ]
    for u in urls_try_tpex:
        try:
            raw = _fetch_isin_table(u)
            tpex_ok = _parse_isin_equities(raw, "TPEX")
            break
        except Exception as e:
            logger.warning("TPEX 取數失敗嘗試 %s: %s", u, e)

    if tpex_ok is None:
        logger.warning("無法取得上櫃清單，僅使用上市。")
        return twse_ok

    return pd.concat([twse_ok, tpex_ok], ignore_index=True)


# ----------------- 資料下載 -----------------
def to_yahoo_symbol(code: str, exchange: str) -> str:
    # TWSE / TPEX 同用 .TW/.TWO
    suffix = ".TW" if exchange == "TWSE" else ".TWO"
    return f"{str(code).zfill(4)}{suffix}"

def download_ohlcv_batches(tickers: List[str],
                           period: str = "8mo",
                           interval: str = "1d",
                           batch_size: int = 120,
                           sleep_sec: float = 0.8) -> Dict[str, pd.DataFrame]:
    data_map: Dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            df = yf.download(
                tickers=batch, period=period, interval=interval,
                group_by="ticker", auto_adjust=False, threads=True, progress=False
            )
            if isinstance(df.columns, pd.MultiIndex):
                for t in batch:
                    if t in df.columns.levels[0]:
                        sub = df[t].dropna(how="all")
                        if not sub.empty:
                            sub = sub.rename(columns=str.title)  # Open/High/Low/Close/Adj Close/Volume
                            data_map[t] = sub
            else:
                # 單一股票情境
                sub = df.dropna(how="all")
                if not sub.empty:
                    sub = sub.rename(columns=str.title)
                    data_map[batch[0]] = sub
        except Exception as e:
            logger.warning("下載價格失敗 batch %d-%d: %s", i, i + len(batch), e)
        time.sleep(sleep_sec)
    return data_map


# ----------------- 技術指標 -----------------
def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()

def stochastic_kd(high: pd.Series, low: pd.Series, close: pd.Series,
                  n: int = 9, k_smooth: int = 3, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    ll = low.rolling(n, min_periods=n).min()
    hh = high.rolling(n, min_periods=n).max()
    rsv = (close - ll) / (hh - ll) * 100.0
    K = rsv.rolling(k_smooth, min_periods=k_smooth).mean()
    D = K.rolling(d_period, min_periods=d_period).mean()
    return K, D

def find_golden_cross_in_window(K: pd.Series, D: pd.Series,
                                window: int,
                                zone_low: float, zone_high: float,
                                require_zone: bool) -> Optional[int]:
    # 回傳 cross 的整數位置（iloc索引），若無則 None
    last = len(K) - 1
    start = max(1, last - window + 1)
    for i in range(start, last + 1):
        try:
            prev_k, prev_d = K.iloc[i - 1], D.iloc[i - 1]
            cur_k, cur_d  = K.iloc[i], D.iloc[i]
        except Exception:
            continue
        if np.isnan(prev_k) or np.isnan(prev_d) or np.isnan(cur_k) or np.isnan(cur_d):
            continue
        crossed = (prev_k <= prev_d) and (cur_k > cur_d)
        if not crossed:
            continue
        if require_zone:
            if not (zone_low <= cur_k <= zone_high and zone_low <= cur_d <= zone_high):
                continue
        return i
    return None


# ----------------- 市值（yfinance fast_info） -----------------
def get_market_caps_yf(tickers: List[str], sleep: float = 0.05) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for t in tickers:
        mc = None
        try:
            tk = yf.Ticker(t)
            # 優先 fast_info
            fi = getattr(tk, "fast_info", None)
            if fi and "market_cap" in fi and fi["market_cap"]:
                mc = float(fi["market_cap"])
            else:
                info = tk.info or {}
                mc = float(info.get("marketCap") or 0) or None
        except Exception:
            mc = None
        out[t] = mc
        time.sleep(sleep)
    return out


# ----------------- 持久化連續天數 -----------------
STREAKS_PATH = STATE_DIR / "streaks.json"

def load_streaks() -> Dict[str, Dict[str, int]]:
    if STREAKS_PATH.exists():
        try:
            return json.loads(STREAKS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_streaks(updated: Dict[str, Dict[str, int]]) -> None:
    STREAKS_PATH.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------- Telegram -----------------
def send_telegram_message(bot_token: str, chat_id: str, text: str) -> None:
    if not bot_token or not chat_id:
        logger.info("Telegram not configured; skip Telegram sending.")
        return
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=30)
        if resp.status_code != 200:
            logger.warning("Telegram send failed: %s %s", resp.status_code, resp.text)
    except Exception as e:
        logger.warning("Telegram send error: %s", e)


# ----------------- 單檔評估 -----------------
def evaluate_one(df: pd.DataFrame, params: dict) -> Optional[dict]:
    if df is None or df.empty:
        return None
    if df["Volume"].fillna(0).sum() == 0:
        return None

    # 至少要有 40 根才算較穩（MA20、KD等）
    if len(df) < max(40, params["KD_N"] + params["KD_K_SMOOTH"] + params["KD_D_PERIOD"] + 5):
        return None

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    open_ = df["Open"]
    vol   = df["Volume"]

    ma5  = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, 20)

    K, D = stochastic_kd(high, low, close,
                         n=params["KD_N"],
                         k_smooth=params["KD_K_SMOOTH"],
                         d_period=params["KD_D_PERIOD"])

    last = -1
    # —— B 結構 —— #
    if params["ENABLE_RULE_MA5_GT_MA20"]:
        if np.isnan(ma5.iloc[last]) or np.isnan(ma20.iloc[last]) or (ma5.iloc[last] <= ma20.iloc[last]):
            return None

    if params["ENABLE_RULE_OC_ABOVE_MA20"]:
        if np.isnan(ma20.iloc[last]) or (open_.iloc[last] < ma20.iloc[last] and close.iloc[last] < ma20.iloc[last]):
            return None

    if params["ENABLE_RULE_LAST5_MA10_THRESHOLD"]:
        last5 = close.iloc[-5:]
        below = (last5 < ma10.iloc[-5:]).sum()
        if below > params["MAX_DAYS_BELOW_MA10_IN_5"]:
            return None

    # —— C KD —— #
    cross_idx = find_golden_cross_in_window(
        K, D,
        window=params["KD_CROSS_WINDOW"],
        zone_low=params["KD_ZONE_LOW"], zone_high=params["KD_ZONE_HIGH"],
        require_zone=params["KD_REQUIRE_ZONE"]
    )
    if cross_idx is None:
        return None
    # 交叉後當下要 K > D
    if np.isnan(K.iloc[last]) or np.isnan(D.iloc[last]) or (K.iloc[last] <= D.iloc[last]):
        return None

    # —— D 量價 —— #
    # 當日成交量 / 過去20日均量（不含當日）
    if len(vol) < params["VOLUME_LOOKBACK"] + 1:
        return None
    v20 = vol.iloc[-(params["VOLUME_LOOKBACK"] + 1):-1].mean()
    if v20 <= 0:
        return None
    vol_ratio = float(vol.iloc[last] / v20)
    if vol_ratio < params["VOLUME_MULTIPLIER"]:
        return None

    # 黑K 限制：若收黑，收盤 >= 開盤 * 0.95
    if params["ENABLE_RULE_BLACK_CANDLE_LIMIT"]:
        if close.iloc[last] < open_.iloc[last]:
            if close.iloc[last] < open_.iloc[last] * params["BLACK_CANDLE_MAX_DROP"]:
                return None

    # —— E 流動性：近10日每日量 >= 門檻 —— #
    last10 = vol.iloc[-10:]
    if (last10 < params["LIQ_MIN_VOL_LAST10"]).any():
        return None

    # 輸出
    out = dict(
        date=pd.to_datetime(df.index[last]).date().isoformat(),
        close=float(close.iloc[last]),
        K=float(K.iloc[last]) if not np.isnan(K.iloc[last]) else None,
        D=float(D.iloc[last]) if not np.isnan(D.iloc[last]) else None,
        kd_spread=float(K.iloc[last] - D.iloc[last]) if (not np.isnan(K.iloc[last]) and not np.isnan(D.iloc[last])) else None,
        vol_ratio=float(vol_ratio),
        price_ma20_pct=float((close.iloc[last] - ma20.iloc[last]) / ma20.iloc[last]) if not np.isnan(ma20.iloc[last]) and ma20.iloc[last] != 0 else None,
    )
    return out


# ----------------- 主流程 -----------------
def run_once():
    params = get_env_params()

    # 1) 建立股票池
    logger.info("Building universe from TWSE/TPEX ISIN page...")
    uni = fetch_twse_tpex_equities(include_tpex=params["INCLUDE_TPEX"])
    uni["yahoo"] = uni.apply(lambda r: to_yahoo_symbol(r["code"], r["exchange"]), axis=1)
    logger.info("Universe size: %d", len(uni))

    # 2) 抓價量
    logger.info("Downloading OHLCV from Yahoo in batches...")
    tickers = uni["yahoo"].tolist()
    data_map = download_ohlcv_batches(tickers, period="8mo", interval="1d", batch_size=params["BATCH_SIZE"])

    # 3) 以全市場資料推算「最新/前一」交易日，並判斷是否非交易日
    latest_dates = []
    prev_dates = []
    for df in data_map.values():
        if df is None or df.empty:
            continue
        idx = pd.to_datetime(df.index)
        latest_dates.append(idx[-1].date())
        if len(idx) >= 2:
            prev_dates.append(idx[-2].date())

    latest_trade_date = max(latest_dates) if latest_dates else None
    tz = ZoneInfo("Asia/Taipei") if ZoneInfo else None
    today_tpe = datetime.now(tz).date() if tz else datetime.utcnow().date()

    if (latest_trade_date is None) or (latest_trade_date < today_tpe):
        # 非交易日：寫空CSV並發訊
        out_empty = OUTPUT_DIR / f"picks_{today_tpe.strftime('%Y%m%d')}.csv"
        pd.DataFrame(columns=[
            "date","code","name","exchange","yahoo","close","K","D","vol_ratio",
            "kd_spread","price_ma20_pct","score","continuation_days"
        ]).to_csv(out_empty, index=False, encoding="utf-8-sig")
        send_telegram_message(params["TELEGRAM_BOT_TOKEN"], params["TELEGRAM_CHAT_ID"],
                              "今日為非交易日，請開心過好每一天")
        logger.info("No today bars -> Non-trading day. Wrote empty CSV: %s", out_empty)
        return

    prev_trade_date = None
    if prev_dates:
        prev_candidates = [d for d in prev_dates if d < latest_trade_date]
        if prev_candidates:
            prev_trade_date = max(prev_candidates)
    if prev_trade_date is None:
        prev_trade_date = latest_trade_date

    # 4) 先做技術面與量能等條件，減少後續市值查詢量
    rows_pre: List[dict] = []
    for _, r in uni.iterrows():
        ysym = r["yahoo"]
        df = data_map.get(ysym)
        try:
            sig = evaluate_one(df, params)
            if sig:
                rows_pre.append({
                    "date": sig["date"],
                    "code": r["code"],
                    "name": r["name"],
                    "exchange": r["exchange"],
                    "yahoo": ysym,
                    "close": sig["close"],
                    "K": sig["K"],
                    "D": sig["D"],
                    "kd_spread": sig["kd_spread"],
                    "vol_ratio": sig["vol_ratio"],
                    "price_ma20_pct": sig["price_ma20_pct"],
                })
        except Exception as e:
            logger.warning("Signal evaluation failed for %s: %s", ysym, e)

    if not rows_pre:
        # 沒有任何通過技術面者，直接輸出空表
        today = latest_trade_date.strftime("%Y%m%d")
        out_path = OUTPUT_DIR / f"picks_{today}.csv"
        pd.DataFrame(columns=[
            "date","code","name","exchange","yahoo","close","K","D",
            "vol_ratio","kd_spread","price_ma20_pct","score","continuation_days"
        ]).to_csv(out_path, index=False, encoding="utf-8-sig")
        send_telegram_message(params["TELEGRAM_BOT_TOKEN"], params["TELEGRAM_CHAT_ID"],
                              f"今日無任何標的通過技術面條件。")
        logger.info("No technical candidates. Saved empty: %s", out_path)
        return

    # 5) 只對技術面通過者查市值，再套市值門檻
    cands_df = pd.DataFrame(rows_pre)
    logger.info("Fetching market caps from Yahoo for %d candidates...", len(cands_df))
    mcs = get_market_caps_yf(cands_df["yahoo"].tolist())
    cands_df["market_cap"] = cands_df["yahoo"].map(mcs)
    cands_df = cands_df[(~cands_df["market_cap"].isna()) & (cands_df["market_cap"] >= params["MARKET_CAP_MIN"])]

    if cands_df.empty:
        today = latest_trade_date.strftime("%Y%m%d")
        out_path = OUTPUT_DIR / f"picks_{today}.csv"
        cands_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        send_telegram_message(params["TELEGRAM_BOT_TOKEN"], params["TELEGRAM_CHAT_ID"],
                              f"技術面通過，但無符合市值門檻（≥{int(params['MARKET_CAP_MIN'])}）。")
        logger.info("No candidates after market-cap filter. Saved: %s", out_path)
        return

    # 6) 排序分數（a,b,c）與 score
    #   a = (2 - 0.02 * n_k) ；n_k 為 (K-D) 由大->小名次
    #   b = (2 - 0.02 * n_v) ；n_v 為 vol_ratio 由大->小名次
    #   c = price_ma20_pct  原始值
    cands_df = cands_df.copy()
    cands_df["rk_kd"] = cands_df["kd_spread"].rank(ascending=False, method="min").astype(int)
    cands_df["rk_vol"] = cands_df["vol_ratio"].rank(ascending=False, method="min").astype(int)
    cands_df["a"] = 2.0 - 0.02 * cands_df["rk_kd"]
    cands_df["b"] = 2.0 - 0.02 * cands_df["rk_vol"]
    cands_df["c"] = cands_df["price_ma20_pct"].fillna(0.0)
    cands_df["score"] = cands_df["a"] * cands_df["b"] * cands_df["c"]

    cands_df = cands_df.sort_values(["score", "vol_ratio", "kd_spread"], ascending=[False, False, False])

    # 7) 連續出現：依 prev_trade_date 判斷是否延續
    key_field = params.get("CONTINUATION_KEY", "yahoo")
    if key_field not in cands_df.columns:
        key_field = "yahoo"

    streaks = load_streaks()
    updated: Dict[str, Dict[str, int]] = {}
    cont_days = []
    for _, row in cands_df.iterrows():
        key = str(row[key_field])
        prev = streaks.get(key)
        if prev and prev.get("last_date") == str(prev_trade_date):
            days = int(prev.get("streak", 1)) + 1
        else:
            days = 1
        cont_days.append(days)
        updated[key] = {"last_date": str(latest_trade_date), "streak": days}
    cands_df["continuation_days"] = cont_days

    save_streaks(updated)

    # 8) 取前 TOP_N
    topN = max(1, int(params["TOP_N"]))
    df_top = cands_df.head(topN).reset_index(drop=True)

    # 9) 存檔 CSV
    today = latest_trade_date.strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"picks_{today}.csv"
    df_top.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Saved results to %s (count=%d)", out_path, len(df_top))

    # 10) Telegram 訊息：分兩表
    cont_df = df_top[df_top["continuation_days"] >= 2].copy()
    one_df = df_top[df_top["continuation_days"] < 2].copy()

    def fmt_row(r) -> str:
        # 只顯示：收盤、KD、放量倍數；(連N) 只在連續表顯示
        return f"{r['code']} {r['name']} | 收 {r['close']:.2f} | K {r['K']:.2f} / D {r['D']:.2f} | 倍數 {r['vol_ratio']:.2f}"

    lines = [f"TWSE/TPEX KD 選股結果（Top {len(df_top)}）"]
    lines.append(f"連續≥2天：{len(cont_df)} 檔； 單日：{len(one_df)} 檔")
    lines.append("")

    if not cont_df.empty:
        lines.append("▶ 連續出現（顯示連續天數）：")
        for _, r in cont_df.iterrows():
            lines.append(f"{fmt_row(r)} | (連{int(r['continuation_days'])})")
        lines.append("")

    lines.append("▶ 非連續標的：")
    if one_df.empty:
        lines.append("(無)")
    else:
        for _, r in one_df.iterrows():
            lines.append(fmt_row(r))

    send_telegram_message(params["TELEGRAM_BOT_TOKEN"], params["TELEGRAM_CHAT_ID"], "\n".join(lines))


if __name__ == "__main__":
    try:
        run_once()
    except Exception as e:
        logger.error("Fatal error: %s\n%s", e, traceback.format_exc())
        # 如果炸了也讓你在 TG 收到
        try:
            p = get_env_params()
            send_telegram_message(p["TELEGRAM_BOT_TOKEN"], p["TELEGRAM_CHAT_ID"], f"❌ Screener 發生錯誤：\n{e}")
        except Exception:
            pass
