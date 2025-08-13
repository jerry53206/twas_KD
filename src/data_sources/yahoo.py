from typing import List, Dict
import time
import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger("twse_kd_screener")

def download_ohlcv_batches(tickers: List[str],
                           period: str = "6mo",
                           interval: str = "1d",
                           batch_size: int = 120,
                           retries: int = 2,
                           sleep_sec: float = 1.0) -> Dict[str, pd.DataFrame]:
    """
    以 batch 方式透過 yfinance.download 抓日線 OHLCV；回傳 dict: {symbol: df}
    """
    result: Dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        attempt = 0
        while attempt <= retries:
            try:
                df = yf.download(
                    tickers=" ".join(batch),
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=False,
                    threads=True,
                    progress=False,
                    timeout=60
                )
                # yfinance 在單一/多標的回傳格式不同，統一處理
                if isinstance(df.columns, pd.MultiIndex):
                    # 多標的：第一層是 ticker
                    for sym in batch:
                        if sym in df.columns.levels[0]:
                            sdf = df[sym].dropna(how="all")
                            if not sdf.empty:
                                # 確保欄位順序
                                cols = ["Open","High","Low","Close","Adj Close","Volume"]
                                cols = [c for c in cols if c in sdf.columns]
                                result[sym] = sdf[cols].copy()
                        else:
                            logger.warning(f"{sym}: no data in multi-index frame")
                else:
                    # 單標的：直接一張表
                    sdf = df.dropna(how="all")
                    if not sdf.empty:
                        result[batch[0]] = sdf.copy()
                break
            except Exception as e:
                attempt += 1
                if attempt > retries:
                    logger.error(f"Batch {i}-{i+len(batch)} failed: {e}")
                else:
                    logger.warning(f"Batch error, retry {attempt}/{retries} ... {e}")
                    time.sleep(sleep_sec)
    return result
