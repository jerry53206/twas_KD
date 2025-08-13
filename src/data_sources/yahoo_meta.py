# src/data_sources/yahoo_meta.py
import time
import logging
from typing import List, Dict, Optional
import yfinance as yf

logger = logging.getLogger("twse_kd_screener")

def get_market_caps(tickers: List[str], retries: int = 1, sleep: float = 0.05) -> Dict[str, float]:
    """
    回傳 {symbol: market_cap_float}；單位以 Yahoo 回傳為準。
    台股 .TW 通常為新台幣。
    """
    out: Dict[str, float] = {}
    for sym in tickers:
        mc: Optional[float] = None
        attempt = 0
        while attempt <= retries and mc is None:
            try:
                tk = yf.Ticker(sym)
                # 1) fast_info
                try:
                    fi = getattr(tk, "fast_info", None)
                    if fi:
                        mc = fi.get("market_cap", None)
                except Exception:
                    mc = None
                # 2) 備援：info
                if mc is None:
                    try:
                        info = tk.get_info() or {}
                        mc = info.get("marketCap", None)
                    except Exception:
                        mc = None
            except Exception as e:
                logger.warning(f"{sym} get_market_caps error: {e}")
            finally:
                attempt += 1
                if mc is None and attempt <= retries:
                    time.sleep(sleep)
        if isinstance(mc, (int, float)) and mc > 0:
            out[sym] = float(mc)
        time.sleep(sleep)
    return out
