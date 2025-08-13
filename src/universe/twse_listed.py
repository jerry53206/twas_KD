# src/universe/twse_listed.py
import re
import time
from io import StringIO
from typing import Literal

import requests
import pandas as pd

TWSE_ISIN_URL = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"  # 上市
TPEX_ISIN_URL = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"  # 上櫃

# 排除：ETF/ETN/基金/受益憑證/債/權證/牛熊/特別股/存託憑證/DR…（不分大小寫）
_EXCLUDE_RE = re.compile(
    r"(?:ETF|ETN|基金|受益|受益憑證|債|公司債|可轉債|轉換|購|售|權證|認購|認售|牛|熊|特別股|優先股|存託憑證|DR)",
    re.IGNORECASE,
)

def _http_get(url: str, retries: int = 3, sleep: float = 0.6) -> str:
    last_err = None
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=20)
            if r.ok:
                return r.text
            last_err = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:
            last_err = e
        time.sleep(sleep)
    raise RuntimeError(f"GET failed: {url} ({last_err})")

def _read_isin_table(html: str) -> pd.DataFrame:
    # 用 lxml 解析；若表頭不在 columns，就把第 1 列提升為表頭
    tables = pd.read_html(StringIO(html), flavor="lxml")
    if not tables:
        raise RuntimeError("ISIN 頁面未解析出表格。")
    df = tables[0]
    if not any(isinstance(c, str) and ("有價" in c or "ISIN" in c or "證券" in c) for c in df.columns):
        header = df.iloc[0].astype(str).tolist()
        df = df.iloc[1:].copy()
        df.columns = header
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    return df

def _pick_column(colnames, keywords):
    """在多種候選欄名中做模糊比對（去空白、包含關鍵字即命中）"""
    def norm(s: str) -> str:
        return re.sub(r"\s+", "", str(s))
    for c in colnames:
        nc = norm(c)
        for k in keywords:
            if k in nc:
                return c
    return None

def _parse_isin_equities(df: pd.DataFrame, exchange: Literal["TWSE", "TPEX"]) -> pd.DataFrame:
    cols = list(df.columns)

    # 可能的欄名：合併欄 or 拆開欄
    code_name_col = _pick_column(cols, ["有價證券代號及名稱"])
    code_col = _pick_column(cols, ["有價證券代號", "證券代號", "股票代號", "代號"])
    name_col = _pick_column(cols, ["有價證券名稱", "證券名稱", "股票名稱", "名稱"])

    if code_name_col is not None:
        s = df[code_name_col].astype(str)
        m = s.str.extract(r"^\s*(\d{4})\s*(.+)$")
        m.columns = ["code", "name"]
        out = m.dropna()
    elif code_col is not None and name_col is not None:
        out = df[[code_col, name_col]].copy()
        out.columns = ["code", "name"]
        out["code"] = out["code"].astype(str).str.extract(r"(\d{4})")[0]
        out = out.dropna(subset=["code", "name"])
    else:
        # 萬一表頭再變形，用前兩欄做最後嘗試
        tmp = df.iloc[:, :2].copy()
        tmp.columns = ["code", "name"]
        tmp["code"] = tmp["code"].astype(str).str.extract(r"(\d{4})")[0]
        out = tmp.dropna(subset=["code", "name"])

    # 僅保留 4 碼股票代號；名稱過濾不需要的商品
    out = out[out["code"].str.fullmatch(r"\d{4}")]
    out = out[~out["name"].astype(str).str.contains(_EXCLUDE_RE)]
    out["name"] = out["name"].astype(str).str.strip()
    out = out.drop_duplicates(subset=["code"]).reset_index(drop=True)
    out["exchange"] = exchange
    if out.empty:
        raise RuntimeError("ISIN 表格結構可能變動，解析後為空。")
    return out[["code", "name", "exchange"]]

def fetch_twse_listed_equities() -> pd.DataFrame:
    """回傳（上市＋上櫃）普通股清單：columns = code, name, exchange"""
    tw_html = _http_get(TWSE_ISIN_URL)
    tw_df = _read_isin_table(tw_html)
    twse = _parse_isin_equities(tw_df, "TWSE")

    tp_html = _http_get(TPEX_ISIN_URL)
    tp_df = _read_isin_table(tp_html)
    tpex = _parse_isin_equities(tp_df, "TPEX")

    return pd.concat([twse, tpex], ignore_index=True)
