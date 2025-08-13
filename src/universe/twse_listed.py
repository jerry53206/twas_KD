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

import os
import pandas as pd

# 若你已有新版的總表函式（例如 fetch_listed_universe），這裡用它做相容包裝
try:
    # 假設你現在的主函式叫這個（若名稱不同，就改成你的實名）
    from .twse_listed import fetch_listed_universe  # 若本檔內已有定義，這行可刪
except Exception:
    pass

def fetch_twse_listed_equities():
    """
    供 main.py 相容用。
    回傳欄位至少含：code, name, yahoo
    INCLUDE_TPEX=true 時，含上櫃，yahoo 後綴自動 .TW/.TWO
    """
    include_tpex = os.getenv("INCLUDE_TPEX", "false").lower() == "true"
    # 若你的檔案裡已經有一個整合的函式，直接呼叫它（把名稱改成你的既有函式名）
    if 'fetch_listed_universe' in globals():
        df = fetch_listed_universe(include_tpex=include_tpex)
    else:
        # 如果你分成兩個函式，請把下列兩行替換成你的實名
        df_twse = fetch_twse_only()      # ← 改成你的 TWSE 取數函式
        if include_tpex:
            df_tpex = fetch_tpex_only()  # ← 改成你的 TPEX 取數函式
            df = pd.concat([df_twse, df_tpex], ignore_index=True)
        else:
            df = df_twse

    # 保證有 yahoo 欄位
    if "yahoo" not in df.columns:
        if "exchange" in df.columns:
            suf = df["exchange"].map({"TWSE": ".TW", "TPEX": ".TWO"}).fillna(".TW")
            df["yahoo"] = df["code"].astype(str).str.zfill(4) + suf
        else:
            df["yahoo"] = df["code"].astype(str).str.zfill(4) + ".TW"
    return df[["code", "name", "yahoo"]]
