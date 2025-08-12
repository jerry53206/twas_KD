# src/universe/twse_listed.py
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path

# 排除的關鍵字（名稱或有價證券別含這些就略過）
EXCLUDE_KEYWORDS = r"(?:ETF|受益|指數投資證券|特別股|存託憑證|權證|票券|債|不動產資產信託)"
OPENAPI_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
ISIN_URL = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"}

# 專案根目錄 = src/../..，快取路徑：<root>/cache/listed_universe.csv
CACHE_PATH = (Path(__file__).resolve().parents[2] / "cache" / "listed_universe.csv")


def _normalize(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("（", "(").replace("）", ")").replace("\u3000", " ").strip()
    return re.sub(r"\s+", " ", s)


def _fetch_from_openapi(timeout: int = 20) -> pd.DataFrame:
    """
    以 TWSE OpenData 取得上市公司名單。
    常見欄位：公司代號、公司名稱、公司簡稱、產業別、上市日期、市場別、有價證券別...
    """
    r = requests.get(OPENAPI_URL, headers=UA, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    df = pd.DataFrame(js)

    code_col = next((c for c in ["公司代號", "證券代號", "Code", "code"] if c in df.columns), None)
    name_col = next((c for c in ["公司名稱", "公司簡稱", "Name", "name"] if c in df.columns), None)
    market_col = next((c for c in ["市場別", "Market"] if c in df.columns), None)
    type_col = next((c for c in ["有價證券別", "SecurityType"] if c in df.columns), None)

    if code_col is None or name_col is None:
        raise RuntimeError("OpenAPI 欄位異常：缺少代號或名稱欄位")

    df[code_col] = df[code_col].astype(str).str.strip()
    df[name_col] = df[name_col].astype(str).map(_normalize)

    # 只留 4 碼數字代碼
    df = df[df[code_col].str.fullmatch(r"\d{4}")]

    # 只留「上市」
    if market_col and market_col in df.columns:
        df = df[df[market_col].astype(str).str.contains("上市", na=False)]

    # 排除 ETF/受益/權證等
    if type_col and type_col in df.columns:
        df = df[~df[type_col].astype(str).str.contains(EXCLUDE_KEYWORDS)]
    df = df[~df[name_col].astype(str).str.contains(EXCLUDE_KEYWORDS)]

    out = df[[code_col, name_col]].drop_duplicates()
    out.columns = ["code", "name"]
    return out.reset_index(drop=True)


def _fetch_from_isin(timeout: int = 20) -> pd.DataFrame:
    """
    後備方案：解析 ISIN 網頁（Big5）。僅靠第一欄「代號  名稱」抽取，再用關鍵字過濾。
    """
    r = requests.get(ISIN_URL, headers=UA, timeout=timeout)
    r.encoding = "big5"
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if not table:
        raise RuntimeError("ISIN 頁面找不到表格")

    recs = []
    for tr in table.find_all("tr"):
        tds = tr.find_all(["td", "th"])
        if not tds:
            continue
        first = _normalize(tds[0].get_text())
        if "有價證券代號及名稱" in first:
            continue
        m = re.match(r"^(?P<code>\d{4,6})[ \u3000]+(?P<name>.+)$", first)
        if not m:
            continue
        code = m.group("code")
        name = _normalize(m.group("name"))
        if len(code) == 4 and not re.search(EXCLUDE_KEYWORDS, name):
            recs.append({"code": code, "name": name})

    if not recs:
        raise RuntimeError("ISIN 解析不到任何普通股代碼")

    return pd.DataFrame(recs).drop_duplicates().reset_index(drop=True)


def fetch_twse_listed_equities(timeout: int = 20) -> pd.DataFrame:
    """
    主函式：OpenData → ISIN → 快取 三段式。
    """
    # 1) 先試 OpenData
    try:
        df = _fetch_from_openapi(timeout=timeout)
        if not df.empty:
            # 成功即寫入快取
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(CACHE_PATH, index=False, encoding="utf-8-sig")
            return df
    except Exception:
        pass

    # 2) 再試 ISIN
    try:
        df = _fetch_from_isin(timeout=timeout)
        if not df.empty:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(CACHE_PATH, index=False, encoding="utf-8-sig")
            return df
    except Exception:
        pass

    # 3) 最後用快取
    if CACHE_PATH.exists():
        df = pd.read_csv(CACHE_PATH, dtype=str)
        if not df.empty:
            return df[["code", "name"]].drop_duplicates().reset_index(drop=True)

    raise RuntimeError("無法取得上市普通股名單（OpenData/ISIN/快取皆不可用）")