# src/universe/twse_listed.py
import os
import re
from io import StringIO
from typing import Optional

import pandas as pd
import requests

TWSE_ISIN_URL = "https://isin.twse.com.tw/isin/C_public.jsp"

# 排除的種類／名稱關鍵字（盡量只保留普通股）
EXCLUDE_INDUSTRY = {
    "ETF", "ETN", "受益證券", "封閉式基金", "指數投資證券", "債券", "證券投資信託"
}
EXCLUDE_NAME_REGEX = re.compile(
    r"(特別股|受益|權證|牛熊|認購|認售|WARRANT|公司債|可轉債|存託憑證)", re.IGNORECASE
)


def _fetch_isin_table(str_mode: int) -> pd.DataFrame:
    """
    讀取 TWSE ISIN 公開頁面：上市=2，上櫃=4
    """
    r = requests.get(TWSE_ISIN_URL, params={"strMode": str_mode}, timeout=30)
    # 該頁面為中文編碼（常見 cp950/big5），手動指定比較穩定
    r.encoding = "cp950"
    tables = pd.read_html(StringIO(r.text))
    # 選第一個含有 ISIN 欄位的表
    for t in tables:
        if any("ISIN" in str(c) for c in t.columns):
            return t
    # 退而求其次：回傳第一個表
    return tables[0]


def _parse_isin_equities(raw: pd.DataFrame, exchange: str) -> pd.DataFrame:
    """
    從 ISIN 表格中萃取普通股清單，回傳欄位：code, name, exchange
    """
    def find_col(keyword: str) -> Optional[str]:
        for c in raw.columns:
            sc = str(c)
            if keyword in sc:
                return c
        return None

    col_cn = find_col("代號") or find_col("名稱")
    col_isin = find_col("ISIN")
    col_ind = find_col("產業")

    if col_cn is None:
        raise RuntimeError("ISIN 表格結構可能變動，找不到『有價證券代號及名稱』欄。")

    df = raw.copy()

    # 去掉分類列（例如「股票」小節標題等），通常這些列的 ISIN 會是 NaN
    if col_isin in df.columns:
        df = df[df[col_isin].notna()]

    # 拆出代號與名稱：格式多為「2330  台積電」
    cn = df[col_cn].astype(str).str.strip()
    m = cn.str.extract(r"^(?P<code>\d{4})\s+(?P<name>.+)$")

    df = df.loc[m.index].copy()
    df["code"] = m["code"]
    df["name"] = m["name"].str.strip()

    # 只收 4 碼數字代號（普通股），排除權證/特別股等（通常非純數字或非 4 碼）
    df = df[df["code"].str.fullmatch(r"\d{4}")]

    # 排除非普通股（ETF 等），以及名稱關鍵字
    if col_ind in df.columns:
        df = df[~df[col_ind].astype(str).isin(EXCLUDE_INDUSTRY)]
    df = df[~df["name"].str.contains(EXCLUDE_NAME_REGEX)]

    df["exchange"] = exchange
    return df[["code", "name", "exchange"]].drop_duplicates().reset_index(drop=True)


def fetch_twse_only() -> pd.DataFrame:
    raw = _fetch_isin_table(2)  # 上市
    return _parse_isin_equities(raw, "TWSE")


def fetch_tpex_only() -> pd.DataFrame:
    raw = _fetch_isin_table(4)  # 上櫃
    return _parse_isin_equities(raw, "TPEX")


def fetch_twse_listed_equities() -> pd.DataFrame:
    """
    供 main.py 相容：讀取上市；若環境變數 INCLUDE_TPEX=true，則連同上櫃一起回傳。
    回傳至少包含欄位：code, name, exchange
    """
    include_tpex = os.getenv("INCLUDE_TPEX", "false").lower() == "true"

    df_twse = fetch_twse_only()
    if include_tpex:
        try:
            df_tpex = fetch_tpex_only()
            df = pd.concat([df_twse, df_tpex], ignore_index=True)
        except Exception:
            # 若 TPEX 讀取失敗，就先提供 TWSE
            df = df_twse
    else:
        df = df_twse

    return df
