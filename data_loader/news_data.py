# data_loader/news_data.py
import os, time
import requests
from dotenv import load_dotenv
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_API_KEY")
FIRECRAWL_KEY = os.getenv("FIRECRAWL_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

def get_news_titles(symbol, page_size=10):
    out = []
    if NEWSAPI_KEY:
        try:
            url = ("https://newsapi.org/v2/everything?"
                   f"q={symbol}&pageSize={page_size}&language=en&sortBy=publishedAt&apiKey={NEWSAPI_KEY}")
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            j = r.json()
            for a in j.get("articles", []):
                t = (a.get("title") or "") + " " + (a.get("description") or "")
                if t.strip():
                    out.append(t.strip())
        except Exception:
            pass
    # Firecrawl (optional)
    if not out and FIRECRAWL_KEY:
        try:
            url = f"https://api.firecrawl.com/search?q={symbol}&limit={page_size}&key={FIRECRAWL_KEY}"
            r = requests.get(url, timeout=10)
            j = r.json()
            for it in j.get("results", []):
                out.append(it.get("title","") + " " + it.get("snippet",""))
        except Exception:
            pass
    return out

def sentiment_openai_batch(texts):
    """Return mean sentiment in [-1, +1] using OpenAI chat completion."""
    if not OPENAI_KEY or not texts:
        return 0.0
    try:
        import openai
        openai.api_key = OPENAI_KEY
        results = []
        for t in texts[:8]:
            prompt = (
                "Rate the sentiment of the following headline/short text on a scale -1 (very bearish) to +1 (very bullish). "
                f"Text: {t}\nAnswer with a single number between -1 and 1."
            )
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                max_tokens=8,
                temperature=0.0,
            )
            txt = resp.choices[0].message.content.strip()
            try:
                val = float(txt.split()[0])
            except Exception:
                try:
                    val = float(txt)
                except Exception:
                    val = 0.0
            results.append(max(min(val, 1.0), -1.0))
            time.sleep(0.3)
        if not results:
            return 0.0
        return float(sum(results)/len(results))
    except Exception:
        return 0.0

def get_news_sentiment(symbol):
    titles = get_news_titles(symbol)
    if not titles:
        return 0.0
    return sentiment_openai_batch(titles)
