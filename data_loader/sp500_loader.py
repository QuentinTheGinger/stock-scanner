import os
import requests
import pandas as pd

# Hauptfunktion
def get_sp500_tickers():
    tickers = []

    # 1️⃣ Versuch: Wikipedia
    try:
        print("🌍 Lade S&P500 von Wikipedia…")
        resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", timeout=10)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        df = tables[0]
        tickers = df["Symbol"].tolist()
        print(f"✅ Wikipedia erfolgreich: {len(tickers)} Ticker geladen.")
        return tickers
    except Exception as e:
        print(f"⚠️ Wikipedia Fehler: {e}")

    # 2️⃣ Versuch: DataHub (CSV)
    try:
        print("📡 Lade S&P500 von DataHub…")
        url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        df = pd.read_csv(url)
        tickers = df["Symbol"].tolist()
        print(f"✅ DataHub erfolgreich: {len(tickers)} Ticker geladen.")
        return tickers
    except Exception as e:
        print(f"⚠️ DataHub Fehler: {e}")

    # 3️⃣ Fallback: statische Liste
    print("❌ Keine Quelle erfolgreich, nutze Fallback-Liste.")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
    print(f"✅ Fallback: {len(tickers)} Ticker geladen.")
    return tickers


if __name__ == "__main__":
    tickers = get_sp500_tickers()
    print("📊 Geladene Ticker:", tickers[:20], "...")
