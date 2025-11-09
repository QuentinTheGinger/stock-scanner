# news_fetcher.py
import requests
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import quote_plus
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser
import time

analyzer = SentimentIntensityAnalyzer()


def fetch_rss_feed(url, max_items=20):
    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:max_items]:
            items.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", ""),
                "published": entry.get("published", ""),
                "link": entry.get("link", "")
            })
        return items
    except Exception as e:
        print(f"⚠️ RSS-Fehler bei {url}: {e}")
        return []


def fetch_google_news(query):
    encoded = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
    return fetch_rss_feed(url)


def fetch_bing_news(query):
    encoded = quote_plus(query)
    url = f"https://www.bing.com/news/search?q={encoded}&format=rss"
    return fetch_rss_feed(url)


def fetch_yahoo_finance_news(symbol):
    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
        return fetch_rss_feed(url)
    except Exception as e:
        print(f"⚠️ Yahoo-RSS-Fehler für {symbol}: {e}")
        return []


def fetch_news_for_symbol(symbol, company_name, window_days=7):
    """Holt News von Google, Bing und Yahoo für ein Symbol."""
    print(f"🗞️  Hole News für {symbol} ({company_name})...")
    queries = [f"{company_name} stock", f"{symbol} stock"]
    all_items = []
    for q in queries:
        all_items += fetch_google_news(q)
        all_items += fetch_bing_news(q)
        all_items += fetch_yahoo_finance_news(symbol)
        time.sleep(0.5)

    if not all_items:
        print(f"⚠️ Keine News gefunden für {symbol}")
        return pd.DataFrame(columns=["symbol", "publishedAt", "title", "description", "url"])

    df = pd.DataFrame(all_items)
    df["symbol"] = symbol
    df["publishedAt"] = pd.to_datetime(df["published"], errors="coerce")
    df["description"] = df["summary"].fillna("")
    df["url"] = df["link"]
    df = df.dropna(subset=["title"])
    return df[["symbol", "publishedAt", "title", "description", "url"]]


def sentiment_score_text(text):
    s = analyzer.polarity_scores(text or "")
    return s["compound"], s["pos"], s["neg"], s["neu"]


def compute_news_features(df_news, ref_dates):
    rows = []
    for symbol in df_news["symbol"].unique():
        sub = df_news[df_news["symbol"] == symbol].copy()
        sub["text"] = (sub["title"].fillna("") + ". " + sub["description"].fillna("")).str[:2000]
        subsent = sub["text"].apply(sentiment_score_text)
        subsent = pd.DataFrame(subsent.tolist(), columns=["compound", "pos", "neg", "neu"], index=sub.index)
        sub = pd.concat([sub, subsent], axis=1)

        for d in ref_dates:
            window_start = pd.to_datetime(d) - timedelta(hours=24)

            # 🕒 Zeitzonen vereinheitlichen (Fehlerfix)
            sub["publishedAt"] = pd.to_datetime(sub["publishedAt"], errors="coerce").dt.tz_localize(None)
            window_end = pd.to_datetime(d)

            # Auswahl der Artikel innerhalb der letzten 24 Stunden
            sel = sub[sub["publishedAt"].between(window_start, window_end)]
            cnt = len(sel)

            if cnt == 0:
                rows.append({
                    "symbol": symbol,
                    "date": pd.to_datetime(d).date(),
                    "news_count_24h": 0,
                    "news_sentiment_24h": 0.0,
                    "news_pos_ratio_24h": 0.0,
                    "news_neg_event_24h": 0
                })
            else:
                # gewichteter Durchschnitt / einfache Mittelwerte
                mean_comp = sel["compound"].mean()
                pos_ratio = (sel["compound"] > 0.05).sum() / cnt
                neg_flag = int((sel["compound"] < -0.6).any())  # sehr negative Headline vorhanden

                rows.append({
                    "symbol": symbol,
                    "date": pd.to_datetime(d).date(),
                    "news_count_24h": cnt,
                    "news_sentiment_24h": mean_comp,
                    "news_pos_ratio_24h": pos_ratio,
                    "news_neg_event_24h": neg_flag
                })

    return pd.DataFrame(rows)
