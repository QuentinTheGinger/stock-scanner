# fundamentals_fetcher.py
import time
import requests
import pandas as pd
import re
from typing import Dict, Any

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

# -------------------------------
# Hilfsfunktion: Reinigungs-Helper
# -------------------------------
def _parse_number(val):
    if isinstance(val, (int, float)):
        return val
    if not val or val == '-' or val is None:
        return None
    s = str(val).strip().replace(",", "").upper()
    multipliers = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000, 'T': 1_000_000_000_000}
    m = re.match(r"^([-+]?[0-9]*\.?[0-9]+)([KMBT]?)$", s)
    if not m:
        # try to strip percent sign or dollar sign
        s2 = s.replace("$", "").replace("%", "")
        try:
            return float(s2)
        except Exception:
            return None
    num, suf = m.groups()
    return float(num) * multipliers.get(suf, 1)

# -------------------------------
# 1️⃣ Yahoo JSON endpoint (robuster als HTML)
# -------------------------------
def fetch_yahoo_json(symbol: str) -> Dict[str, Any]:
    """
    Benutzt Yahoo's quoteSummary JSON endpoint (modules: defaultKeyStatistics, price, financialData).
    Liefert ein dict mit wichtigen Kennzahlen (sofern vorhanden).
    """
    base = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{}"
    params = {"modules": "price,defaultKeyStatistics,financialData"}
    url = base.format(symbol)
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=12)
        if r.status_code != 200:
            # manche Symbole benötigen -P oder sind regional anders; gib Fehler zurück
            return {}
        j = r.json()
        # defensive navigation
        res = {}
        try:
            q = j.get("quoteSummary", {}).get("result", [None])[0]
            if not q:
                return {}
            # price block
            price = q.get("price", {})
            if "marketCap" in price and price["marketCap"] and "raw" in price["marketCap"]:
                res["market_cap"] = float(price["marketCap"]["raw"])
            # key stats
            ks = q.get("defaultKeyStatistics", {})
            if "trailingPE" in ks and ks["trailingPE"] and "raw" in ks["trailingPE"]:
                res["trailing_pe"] = float(ks["trailingPE"]["raw"])
            if "priceToBook" in ks and ks["priceToBook"] and "raw" in ks["priceToBook"]:
                res["price_to_book"] = float(ks["priceToBook"]["raw"])
            # financial data
            fd = q.get("financialData", {})
            if "profitMargins" in fd and fd["profitMargins"] and "raw" in fd["profitMargins"]:
                res["profit_margin"] = float(fd["profitMargins"]["raw"])
            if "totalDebt" in fd and fd["totalDebt"] and "raw" in fd["totalDebt"]:
                res["total_debt"] = float(fd["totalDebt"]["raw"])
            if "totalRevenue" in fd and fd["totalRevenue"] and "raw" in fd["totalRevenue"]:
                res["total_revenue"] = float(fd["totalRevenue"]["raw"])
            if "returnOnEquity" in fd and fd["returnOnEquity"] and "raw" in fd["returnOnEquity"]:
                res["return_on_equity"] = float(fd["returnOnEquity"]["raw"])
        except Exception:
            pass
        res["symbol"] = symbol
        return res
    except Exception:
        return {}

# -------------------------------
# 2️⃣ Macrotrends (Fallback)
# -------------------------------
from bs4 import BeautifulSoup

def fetch_macrotrends_fundamentals(symbol: str) -> Dict[str, Any]:
    url = f"https://www.macrotrends.net/stocks/charts/{symbol.lower()}/{symbol.lower()}/financial-ratios"
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code != 200:
            return {}
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(separator=" ").lower()
        out = {"symbol": symbol}
        # Suche nach einfachen Pattern (defensive)
        m_eps = re.search(r"earnings per share .*?([0-9]+\.[0-9]+)", text)
        if m_eps:
            out["eps"] = _parse_number(m_eps.group(1))
        m_revenue = re.search(r"revenue .*?\$?([0-9]+(?:\.[0-9]+)?[kmbt]?)", text)
        if m_revenue:
            out["revenue"] = _parse_number(m_revenue.group(1))
        m_roe = re.search(r"return on equity .*?([0-9]+\.[0-9]+)%", text)
        if m_roe:
            out["roe"] = _parse_number(m_roe.group(1))
        return out
    except Exception:
        return {}

# -------------------------------
# 3️⃣ Kombinations-Funktion & Rate limiting
# -------------------------------
def fetch_combined_fundamentals(symbol: str) -> Dict[str, Any]:
    """
    Versucht Yahoo JSON; falls leer -> Macrotrends; gibt kombinierte dict zurück.
    """
    out = fetch_yahoo_json(symbol)
    # kurz sleep zur Vorsicht
    time.sleep(0.4)
    if not out or len(out) < 2:
        # fallback
        mf = fetch_macrotrends_fundamentals(symbol)
        for k,v in (mf or {}).items():
            if k not in out or out.get(k) is None:
                out[k] = v
    out["symbol"] = symbol
    return out

# -------------------------------
# 4️⃣ Mehrere Symbole abfragen (DataFrame) – mit Cache
# -------------------------------
def fetch_fundamentals_for_symbols(symbols):
    import os, time
    import pandas as pd

    cache_dir = "data/cache/fundamentals"
    os.makedirs(cache_dir, exist_ok=True)

    results = []

    for i, sym in enumerate(symbols):
        cache_path = os.path.join(cache_dir, f"{sym}.parquet")

        # 🔹 Wenn Cache existiert und jünger als 30 Tage → verwenden
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            age_hours = (time.time() - mtime) / 3600
            if age_hours < 24 * 30:
                try:
                    df = pd.read_parquet(cache_path)
                    if not df.empty:
                        results.append(df.iloc[0].to_dict())
                        print(f"⚡ Fundamentaldaten-Cache genutzt für {sym}")
                        continue
                except Exception:
                    pass  # falls Cache beschädigt → neu laden

        # 🔸 Wenn kein Cache oder älter → neu abrufen
        try:
            print(f"💰 Lade Fundamentaldaten für {sym} …")
            data = fetch_combined_fundamentals(sym)
            if data:
                df = pd.DataFrame([data])
                df.to_parquet(cache_path, index=False)
                results.append(data)
                print(f"💾 Fundamentaldaten gecached für {sym}")
        except Exception as e:
            print(f"⚠️ Fehler beim Laden {sym}: {e}")

        # adaptive sleep (wie vorher)
        if (i % 20) == 19:
            time.sleep(1.5)
        else:
            time.sleep(0.3)

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)
