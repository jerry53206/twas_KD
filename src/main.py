#!/usr/bin/env python3
import os
import sys
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# --- 你專案中的模組（沿用原結構） ---
from universe.twse_listed import fetch_twse_listed_equities  # 需回傳含 exchange 欄位（TWSE/TPEX）
from data_sources.yahoo import download_ohlcv_batches
from data_sources.yahoo_meta import get_market_caps
from indicators.ta import stochastic_kd  # 使用你原本的 KD 計算

# ---------------- 基本設定與 logging ----------------
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
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("twse_kd_screener")

# ----------------- 參數讀取 -----------------
def _to_bool(val: Optional[str], default: bool) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")

def get_env_params() -> Dict:
    load_dotenv(ROOT / ".env", override=True)
    params = dict(
        # ——— KD 與交叉條件 ———
        KD_N=int(os.getenv("KD_N", "9")),
        KD_K_SMOOTH=int(os.getenv("KD_K_SMOOTH", "3")),
        KD_D_PERIOD=int(os.getenv("KD_D_PERIOD", "3")),
        KD_CROSS_WINDOW=int(os.getenv("KD_CROSS_WINDOW", "3")),   # 近幾日內出現黃金交叉
        KD_REQUIRE_ZONE=_to_bool(os.getenv("KD_REQUIRE_ZONE", "false"), False),
        KD_ZONE_LOW=float(os.getenv("KD_ZONE_LOW", "40")),
        KD_ZONE_HIGH=float(os.getenv("KD_ZONE_HIGH", "80")),

        # ——— 成交量條件 ———
        VOLUME_LOOKBACK=int(os.getenv("VOLUME_LOOKBACK", "20")),
        VOLUME_MULTIPLIER=float(os.getenv("VOLUME_MULTIPLIER", "1.5")),  # 今日量 / 20日均量
        MIN_VOL_10D=int(os.getenv("MIN_VOL_10D", "1000000")),            # 近10日逐日下限（股）

        # ——— MA 規則與黑Ｋ限制 ———
        ENABLE_RULE_MA5_GT_MA20=_to_bool(os.getenv("ENABLE_RULE_MA5_GT_MA20", "true"), True),
        ENABLE_RULE_OC_ABOVE_MA20=_to_bool(os.getenv("ENABLE_RULE_OC_ABOVE_MA20", "true"), True),
        ENABLE_RULE_LAST5_MA10_THRESHOLD=_to_bool(os.getenv("ENABLE_RULE_LAST5_MA10_THRESHOLD", "true"), True),
        MAX_DAYS_BELOW_MA10_IN_5=int(os.getenv("MAX_DAYS_BELOW_MA10_IN_5", "3")),
        ENABLE_RULE_BLACK_CANDLE_LIMIT=_to_bool(os.getenv("ENABLE_RULE_BLACK_CANDLE_LIMIT", "true"), True),
        BLACK_CANDLE_MAX_DROP=float(os.getenv("BLACK_CANDLE_MAX_DROP", "0.95")),

        # ——— 市值與批次 ———
        MARKET_CAP_MIN=float(os.getenv("MARKET_CAP_MIN", "10000000000")),
        BATCH_SIZE=int(os.getenv("BATCH_SIZE", "120")),

        # ——— 排名輸出 ———
        TOP_N=int(os.getenv("TOP_N", "20")),

        # ——— 通知 ———
        TELEGRAM_BOT_TOKEN=os.getenv("TELEGRAM_BOT_TOKEN"),
        TELEGRAM_CHAT_ID=os.getenv("TELEGRAM_CHAT_ID"),

        # Email 若未設定則自動略過
        SMTP_HOST=os.getenv("SMTP_HOST"),
        SMTP_PORT=int(os.getenv("SMTP_PORT", "587")),
        SMTP_USE_TLS=_to_bool(os.getenv("SMTP_USE_TLS", "true"), True),
        SMTP_USER=os.getenv("SMTP_USER"),
        SMTP_PASS=os.getenv("SMTP_PASS"),
        SENDER_EMAIL=os.getenv("SENDER_EMAIL"),
        RECIPIENT_EMAIL=os.getenv("RECIPIENT_EMAIL"),
    )

    zone_flag = "ON" if params["KD_REQUIRE_ZONE"] else "OFF"
    logger.info(
        "Params loaded: KD=(%d,%d,%d), window=%d, zone=%s[%.0f~%.0f], VOL: N=%d x%.2f, "
        "MIN_VOL_10D=%d, TOP_N=%d, MCAP_MIN=%d",
        params["KD_N"], params["KD_K_SMOOTH"], params["KD_D_PERIOD"],
        params["KD_CROSS_WINDOW"], zone_flag, params["KD_ZONE_LOW"], params["KD_ZONE_HIGH"],
        params["VOLUME_LOOKBACK"], params["VOLUME_MULTIPLIER"],
        params["MIN_VOL_10D"], params["TOP_N"], int(params["MARKET_CAP_MIN"])
    )
    return params

# --------------- 工具函式 ---------------
def make_yahoo_symbol(code: str, exchange: str) -> str:
    """依交易所決定 Yahoo 後綴：TWSE→.TW；TPEX→.TWO"""
    suffix = ".TW" if str(exchange).upper() == "TWSE" else ".TWO"
    return str(code).zfill(4) + suffix

def build_universe(params: Dict) -> pd.DataFrame:
    """
    以 TWSE ISIN 頁取得「上市/上櫃」普通股清單。
    需包含欄位：code, name, exchange ∈ {TWSE, TPEX}
    """
    logger.info("Building universe from TWSE/TPEX ISIN page...")
    df = fetch_twse_listed_equities()  # 你在 universe.twse_listed 已處理合併
    df["yahoo"] = df.apply(lambda r: make_yahoo_symbol(r["code"], r["exchange"]), axis=1)
    return df[["code", "name", "exchange", "yahoo"]]

def golden_cross_in_last_n(
    K: pd.Series, D: pd.Series, window: int, require_zone: bool,
    zone_low: float, zone_high: float
) -> Optional[int]:
    """
    回傳最近 window 根內是否有 K 由下往上穿越 D 的索引（最後一次），若無則 None。
    若 require_zone=True，則交叉發生當日的 K、D 需同時位於 [zone_low, zone_high]。
    """
    n = len(K)
    if n < 2:
        return None
    start = max(1, n - window)
    found: Optional[int] = None
    for i in range(start, n):
        k_prev, d_prev = K.iloc[i - 1], D.iloc[i - 1]
        k_curr, d_curr = K.iloc[i], D.iloc[i]
        if np.isnan([k_prev, d_prev, k_curr, d_curr]).any():
            continue
        crossed = (k_prev <= d_prev) and (k_curr > d_curr)
        if not crossed:
            continue
        if require_zone:
            in_zone = (zone_low <= k_curr <= zone_high) and (zone_low <= d_curr <= zone_high)
            if not in_zone:
                continue
        found = i
    return found

def volume_today_over_prevN(volume: pd.Series, lookback: int, multiplier: float) -> Tuple[bool, float, float, float]:
    """
    今日量 / 過去 N 日平均量 >= multiplier
    回傳 (bool, ratio, v_today, v_avgN)
    """
    if len(volume) < lookback + 1:
        return False, np.nan, np.nan, np.nan
    v_today = float(volume.iloc[-1])
    v_avg = float(volume.iloc[-(lookback + 1):-1].mean())
    if v_avg <= 0 or np.isnan(v_today) or np.isnan(v_avg):
        return False, np.nan, v_today, v_avg
    ratio = v_today / v_avg
    return (ratio >= multiplier), ratio, v_today, v_avg

def last_n_all_volume_at_least(volume: pd.Series, n: int, min_shares: int) -> bool:
    """最近 n 根每根量都 >= min_shares"""
    if len(volume) < n:
        return False
    tail = volume.iloc[-n:]
    if tail.isna().any():
        return False
    return bool((tail >= min_shares).all())

def moving_averages(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    return close.rolling(5).mean(), close.rolling(10).mean(), close.rolling(20).mean()

# --------------- 單檔訊號與規則檢查 ---------------
def evaluate_signals_for_ticker(
    df: pd.DataFrame,
    params: Dict,
) -> Optional[Dict]:
    """
    對單一 Yahoo 歷史資料 DataFrame（欄位包含 Open, High, Low, Close, Volume）檢查：
    - 近 N=KD_CROSS_WINDOW 日內黃金交叉，且目前 K>D；（可選擇是否限制區間）
    - 今日量 / 20 日均量 >= VOLUME_MULTIPLIER
    - Liquidity：近 10 日逐日量皆 >= MIN_VOL_10D
    - 趨勢：MA5>MA20；(Open>=MA20 or Close>=MA20)；最近 5 日收盤低於 MA10 的天數 <= 阈值
    - 黑Ｋ限制：若 Close<Open，則 Close >= Open * BLACK_CANDLE_MAX_DROP
    通過者回傳指標與後續排名所需資料。
    """
    if df is None or df.empty:
        return None
    if set(["Open", "High", "Low", "Close", "Volume"]) - set(df.columns):
        return None

    # 歷史長度檢查（KD 與 MA、量能）
    min_hist = max(
        60,
        params["VOLUME_LOOKBACK"] + 10,
        params["KD_N"] + params["KD_K_SMOOTH"] + params["KD_D_PERIOD"] + 10
    )
    if len(df) < min_hist:
        return None

    # 移除全零量/全 NaN
    if df["Volume"].fillna(0).sum() <= 0:
        return None

    # ---- 計算 KD ----
    K, D = stochastic_kd(
        df["High"], df["Low"], df["Close"],
        n=params["KD_N"],
        k_smooth=params["KD_K_SMOOTH"],
        d_period=params["KD_D_PERIOD"]
    )
    if K is None or D is None or len(K) != len(df) or len(D) != len(df):
        return None

    k_last, d_last = float(K.iloc[-1]), float(D.iloc[-1])

    # 近 window 日內是否發生黃金交叉（且可選擇是否需在區間）
    cross_idx = golden_cross_in_last_n(
        K, D,
        window=params["KD_CROSS_WINDOW"],
        require_zone=params["KD_REQUIRE_ZONE"],
        zone_low=params["KD_ZONE_LOW"],
        zone_high=params["KD_ZONE_HIGH"],
    )
    if cross_idx is None:
        return None

    # 當下 K 必須 > D
    if not (k_last > d_last):
        return None

    # ---- MA 計算與規則 ----
    ma5, ma10, ma20 = moving_averages(df["Close"])
    ma5_last = float(ma5.iloc[-1])
    ma10_last = float(ma10.iloc[-1])
    ma20_last = float(ma20.iloc[-1])
    close_last = float(df["Close"].iloc[-1])
    open_last = float(df["Open"].iloc[-1])

    # MA5 > MA20
    if params["ENABLE_RULE_MA5_GT_MA20"] and not (ma5_last > ma20_last):
        return None

    # 開盤或收盤其一不可低於 MA20（等同於 open>=MA20 or close>=MA20）
    if params["ENABLE_RULE_OC_ABOVE_MA20"] and not ((open_last >= ma20_last) or (close_last >= ma20_last)):
        return None

    # 最近 5 日，收盤 < MA10 的天數 <= 阈值
    if params["ENABLE_RULE_LAST5_MA10_THRESHOLD"]:
        last5 = df["Close"].iloc[-5:]
        last5_ma10 = ma10.iloc[-5:]
        days_below = int((last5 < last5_ma10).sum())
        if days_below > params["MAX_DAYS_BELOW_MA10_IN_5"]:
            return None

    # 黑Ｋ限制：若收黑（close<open），則 close >= open * BLACK_CANDLE_MAX_DROP（預設 0.95）
    if params["ENABLE_RULE_BLACK_CANDLE_LIMIT"] and (close_last < open_last):
        if close_last < open_last * params["BLACK_CANDLE_MAX_DROP"]:
            return None

    # ---- 量能條件 ----
    ok_vol, vol_ratio, v_today, v_avg = volume_today_over_prevN(
        df["Volume"], params["VOLUME_LOOKBACK"], params["VOLUME_MULTIPLIER"]
    )
    if not ok_vol:
        return None

    # 近 10 日逐日量下限
    if not last_n_all_volume_at_least(df["Volume"], 10, params["MIN_VOL_10D"]):
        return None

    # —— 排名需要的指標 ——
    kd_spread = k_last - d_last
    trend_strength = (close_last - ma20_last) / ma20_last if ma20_last > 0 else np.nan
    if np.isnan(trend_strength):
        return None

    res = {
        "date": pd.to_datetime(df.index[-1]).date().isoformat(),
        "close": close_last,
        "open": open_last,
        "K": k_last,
        "D": d_last,
        "kd_spread": kd_spread,
        "vol_ratio": float(vol_ratio),
        "v_today": float(v_today),
        "v_avg20": float(v_avg),
        "ma20": ma20_last,
        "trend_strength": float(trend_strength),
        "cross_day": pd.to_datetime(df.index[cross_idx]).date().isoformat(),
    }
    return res

# --------------- Telegram ---------------
def send_telegram_message(token: str, chat_id: str, text: str) -> None:
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, data={"chat_id": chat_id, "text": text})
        if resp.status_code != 200:
            logger.warning(f"Telegram send failed ({resp.status_code}): {resp.text}")
        else:
            logger.info("Telegram message sent.")
    except Exception as e:
        logger.warning(f"Telegram exception: {e}")

# --------------- 主流程 ---------------
def run_once():
    params = get_env_params()

    # 1) 建立投資池（上市＋上櫃），帶出 Yahoo 代碼
    uni = build_universe(params)
    logger.info(f"Universe size: {len(uni)}")

    tickers = uni["yahoo"].tolist()
    code_map = dict(zip(uni["yahoo"], uni["code"]))
    name_map = dict(zip(uni["yahoo"], uni["name"]))

    # 2) 下載 OHLCV
    logger.info("Downloading OHLCV from Yahoo in batches...")
    data_map = download_ohlcv_batches(
        tickers,
        period="6mo",
        interval="1d",
        batch_size=params["BATCH_SIZE"],
        retries=2,
        sleep_sec=1.0
    )

    # 3) 先依技術面與量能條件過濾（不含市值）→ 取得候選名單
    prelim_rows: List[Dict] = []
    prelim_syms: List[str] = []
    for ysym, df in data_map.items():
        try:
            sig = evaluate_signals_for_ticker(df, params)
            if sig:
                sig_row = {
                    "yahoo": ysym,
                    "code": code_map.get(ysym, ""),
                    "name": name_map.get(ysym, ""),
                    **sig,
                }
                prelim_rows.append(sig_row)
                prelim_syms.append(ysym)
        except Exception as e:
            logger.warning(f"Signal evaluation failed for {ysym}: {e}")

    if not prelim_rows:
        logger.info("No candidates after technical filters.")
        # 仍輸出空 CSV 方便紀錄
        today = datetime.now().strftime("%Y%m%d")
        out_path = OUTPUT_DIR / f"picks_{today}.csv"
        pd.DataFrame([]).to_csv(out_path, index=False, encoding="utf-8-sig")
        # Telegram 若配置，送空訊息
        if params.get("TELEGRAM_BOT_TOKEN") and params.get("TELEGRAM_CHAT_ID"):
            send_telegram_message(
                params["TELEGRAM_BOT_TOKEN"],
                params["TELEGRAM_CHAT_ID"],
                f"[TWSE/TPEX KD Screener] {today}\n今日無符合條件之個股。"
            )
        return

    # 4) 市值（僅對 prelim 名單查詢，以節省時間）
    logger.info(f"Fetching market caps from Yahoo for {len(prelim_rows)} candidates...")
    mc_map = get_market_caps(prelim_syms, retries=1, sleep=0.05)

    # 5) 套用市值門檻
    final_rows: List[Dict] = []
    for r in prelim_rows:
        mcap = mc_map.get(r["yahoo"])
        if (mcap is None) or (mcap < params["MARKET_CAP_MIN"]):
            continue
        r["market_cap"] = float(mcap)
        final_rows.append(r)

    if not final_rows:
        logger.info("No candidates after market cap filter.")
        today = datetime.now().strftime("%Y%m%d")
        out_path = OUTPUT_DIR / f"picks_{today}.csv"
        pd.DataFrame(prelim_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        if params.get("TELEGRAM_BOT_TOKEN") and params.get("TELEGRAM_CHAT_ID"):
            send_telegram_message(
                params["TELEGRAM_BOT_TOKEN"],
                params["TELEGRAM_CHAT_ID"],
                f"[TWSE/TPEX KD Screener] {today}\n通過技術面，但無達市值門檻（≥100億）的個股。"
            )
        return

    # 6) 排名（a,b,c → Score）
    df_out = pd.DataFrame(final_rows)

    # 排名名次（1 為最佳）
    df_out["n_k"] = df_out["kd_spread"].rank(method="first", ascending=False).astype(int)
    df_out["n_v"] = df_out["vol_ratio"].rank(method="first", ascending=False).astype(int)

    # a, b, c 與 Score
    df_out["a"] = 2.0 - 0.02 * df_out["n_k"]
    df_out["b"] = 2.0 - 0.02 * df_out["n_v"]
    df_out["c"] = df_out["trend_strength"]
    df_out["score"] = df_out["a"] * df_out["b"] * df_out["c"]

    # 依 score 由高到低排序
    df_out.sort_values(["score", "vol_ratio", "kd_spread"], ascending=[False, False, False], inplace=True)

    # 輸出 CSV
    today = datetime.now().strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"picks_{today}.csv"
    keep_cols = [
        "date", "code", "name", "close", "K", "D", "vol_ratio",
        "cross_day", "market_cap", "kd_spread", "n_k", "n_v", "a", "b", "c", "score"
    ]
    for col in keep_cols:
        if col not in df_out.columns:
            df_out[col] = np.nan
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved results to {out_path} (count={len(df_out)})")

    # 7) 通知（Telegram：只顯示 收盤/KD/放量倍數；取前 TOP_N）
    top_n = int(params["TOP_N"])
    if params.get("TELEGRAM_BOT_TOKEN") and params.get("TELEGRAM_CHAT_ID"):
        head = df_out.head(top_n)
        lines = [f"[TWSE/TPEX KD Screener] {today} 前{min(top_n, len(head))}名"]
        for _, r in head.iterrows():
            try:
                line = (
                    f"{str(r['code']).zfill(4)} {r['name']} | "
                    f"收盤 {float(r['close']):.2f} | "
                    f"KD {float(r['K']):.2f}/{float(r['D']):.2f} | "
                    f"量能倍數 {float(r['vol_ratio']):.2f}"
                )
                lines.append(line)
            except Exception:
                # 容錯避免因 NaN 導致整體失敗
                lines.append(f"{str(r.get('code','')).zfill(4)} {r.get('name','')} | 資料格式異常")
        send_telegram_message(
            params["TELEGRAM_BOT_TOKEN"],
            params["TELEGRAM_CHAT_ID"],
            "\n".join(lines)
        )
    else:
        logger.info("Telegram not configured; skip Telegram sending.")

    # 8) Email（若已設定則帶附件寄出；否則略過）
    try:
        if params.get("RECIPIENT_EMAIL") and params.get("SMTP_HOST") and params.get("SMTP_USER"):
            from notify.emailer import send_email_with_attachments
            subject = f"[TWSE/TPEX KD Screener] {today} 選股結果（共 {len(df_out)} 檔）"
            # 只在信文中摘要前 10 檔，欄位同 Telegram
            body_lines = [f"前10名（依 score）："]
            for _, r in df_out.head(10).iterrows():
                try:
                    body_lines.append(
                        f"{str(r['code']).zfill(4)} {r['name']} | "
                        f"收盤 {float(r['close']):.2f} | "
                        f"KD {float(r['K']):.2f}/{float(r['D']):.2f} | "
                        f"量能倍數 {float(r['vol_ratio']):.2f}"
                    )
                except Exception:
                    body_lines.append(f"{str(r.get('code','')).zfill(4)} {r.get('name','')} | 資料格式異常")
            body = "\n".join(body_lines)

            send_email_with_attachments(
                smtp_host=params["SMTP_HOST"],
                smtp_port=params["SMTP_PORT"],
                use_tls=params["SMTP_USE_TLS"],
                username=params["SMTP_USER"],
                password=params["SMTP_PASS"],
                sender=params["SENDER_EMAIL"] or params["SMTP_USER"],
                recipients=[params["RECIPIENT_EMAIL"]],
                subject=subject,
                body=body,
                attachments=[str(out_path)]
            )
            logger.info(f"Email sent to {params['RECIPIENT_EMAIL']}")
        else:
            logger.info("RECIPIENT_EMAIL/SMTP 未設定；略過寄信。")
    except Exception as e:
        logger.error(f"Email sending failed: {e}\n{traceback.format_exc()}")

# ----------------- 進入點 -----------------
if __name__ == "__main__":
    run_once()
