# src/universe/tpex_listed.py
import pandas as pd
import requests
from io import StringIO

TPEX_ISIN_URL = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"
UA = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/123.0.0.0 Safari/537.36"
}

# 簡易排除關鍵字（避免 ETF/權證/債券/受益等）
EXCLUDE_KEYWORDS = [
    "ETF", "受益", "權證", "認購", "認售", "公司債", "轉換", "交換",
    "存託憑證", "特別股", "優先股", "甲特", "乙特", "丙特", "債", "票據"
]

def fetch_tpex_listed_equities() -> pd.DataFrame:
    r = requests.get(TPEX_ISIN_URL, headers=UA, timeout=20)
    # ISIN 網頁多以 BIG5 呈現
    r.encoding = "big5"
    tables = pd.read_html(StringIO(r.text))
    df = None
    for t in tables:
        if "有價證券代號及名稱" in t.columns:
            df = t
            break
    if df is None or df.empty:
        raise RuntimeError("未能從 TPEX ISIN 解析出上櫃普通股名單。")

    name_col = "有價證券代號及名稱"
    # 解析「代號 名稱」
    tmp = df[name_col].astype(str).str.extract(r"^(\d+)\s+(.+)$")
    tmp.columns = ["code", "name"]
    df = pd.concat([tmp, df], axis=1)

    # 只留 4 位數數字代碼（排除特別股帶字尾）
    df = df[df["code"].str.match(r"^\d{4}$", na=False)]

    # 市場別含「上櫃」
    mkt_cols = [c for c in df.columns if "市場別" in str(c)]
    if mkt_cols:
        df = df[df[mkt_cols[0]].astype(str).str.contains("上櫃", na=False)]

    # 排除不需要的產品別
    pat = "|".join(EXCLUDE_KEYWORDS)
    df = df[~df["name"].astype(str).str.contains(pat, na=False)]

    return df[["code", "name"]].drop_duplicates("code").reset_index(drop=True)