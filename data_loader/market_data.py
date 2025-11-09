import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Fallback: yfinance
import yfinance as yf

# Lade .env Datei
load_dotenv()

# Debug: Keys ausgeben
print("DEBUG ALPHA =", os.getenv("ALPHA_VANTAGE_API_KEY"))
print("DEBUG TWELVE =", os.getenv("TWELVEDATA_API_KEY"))

def get_alpha_vantage(symbol: str, interval: str = "60min") -> pd.DataFrame:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return pd.DataFrame()

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key,
        "outputsize": "compact"
    }

    r = requests.get(url, params=params)
    if r.status_code != 200:
        return pd.DataFrame()

    data = r.json()
    key = f"Time Series ({interval})"
    if key not in data:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(data[key], orient="index", dtype=float)
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns=lambda x: x[3:])  # "1. open" → "open"
    df = df.sort_index()
    return df.reset_index().rename(columns={"index": "date"})


def get_twelvedata(symbol: str, interval: str = "1h") -> pd.DataFrame:
    api_key = os.getenv("TWELVEDATA_API_KEY")
    if not api_key:
        return pd.DataFrame()

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key,
        "outputsize": 100
    }

    r = requests.get(url, params=params)
    if r.status_code != 200:
        return pd.DataFrame()

    data = r.json()
    if "values" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.rename(columns={
        "datetime": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume"
    })
    df = df.sort_values("date")
    return df


def get_yfinance(symbol: str, interval: str = "1h") -> pd.DataFrame:
    try:
        df = yf.download(symbol, period="5d", interval=interval, progress=False)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df = df.rename(columns={
            "Datetime": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        return df
    except Exception as e:
        print("⚠️ Fehler bei yfinance:", e)
        return pd.DataFrame()


def get_market_data(symbol: str) -> pd.DataFrame:
    # 1️⃣ Alpha Vantage
    df = get_alpha_vantage(symbol)
    if not df.empty:
        print(f"✅ Alpha Vantage Daten für {symbol} geladen.")
        return df

    # 2️⃣ TwelveData
    df = get_twelvedata(symbol)
    if not df.empty:
        print(f"✅ TwelveData Daten für {symbol} geladen.")
        return df

    # 3️⃣ yfinance
    df = get_yfinance(symbol)
    if not df.empty:
        print(f"✅ yfinance Daten für {symbol} geladen.")
        return df

    print(f"❌ Keine Kursdaten für {symbol} verfügbar.")
    return pd.DataFrame()


if __name__ == "__main__":
    symbol = "AAPL"
    df = get_market_data(symbol)
    print(df.head())
