#!/usr/bin/env python3
# analysispredict_next5_FIXED.py
# Vollständig bereinigte und optimierte Version von analysispredict_next5.py
# (nur minimale, notwendige Fixes: df_feat/fundamentals-Block, Einrückungen)

import os
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 📰 News & Sentiment-Features
from news_fetcher import fetch_news_for_symbol, compute_news_features
from fundamentals_fetcher import fetch_fundamentals_for_symbols

import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Optional: TF/Keras
TF_OK = False
try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    TF_OK = True
except Exception:
    TF_OK = False

# Optional: PyTorch (keine weitere Nutzung in dieser Version)
try:
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass

# Daten-Loader (vom Nutzer bereitgestelltes Modul)
try:
    from data_loader.sp500_loader import get_sp500_tickers
except Exception:
    # Fallback: einfache Liste (wenn Modul nicht vorhanden)
    def get_sp500_tickers():
        return ["AAPL","MSFT","AMZN","GOOGL","META"]

import yfinance as yf

# SKLearn / Modelle / Hilfsfunktionen
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.base import clone

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False

if TF_OK:
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout

import optuna
import json

# -----------------------------
# CONFIG (leicht anpassbar)
# -----------------------------
CONFIG = {
    # Datengrundlage / Limits
    "max_tickers_analysis": 150,
    "max_tickers_backtest": 50,
    "start_date": "2024-01-01",   # z.B. "2022-01-01" oder None
    "years_back": 2,              # wie viele Jahre historisch laden (Backtesting-Performance)

    # Modelle / Training
    "use_optuna": False,
    "train_rnn": False,
    "seq_len": 20,
    "epochs": 8,
    "batch_size": 64,

    # Backtest
    "backtest_freq": "7D",       # Schrittweite (z.B. "7D","14D","30D") oder "1D" für jeden Tag
    "backtest_top_n": 20,
    "run_backtest": True,

    # Klassifikation thresholds
    "prob_threshold": 0.5,
    "strong_p": 0.75,

    # Sonstiges
    "verbose": True,
    "save_predictions": True,
}

# Parameter-Pfad
PARAM_DIR = "parameters_models"
PARAM_FILE = os.path.join(PARAM_DIR, "best_params.json")
os.makedirs(PARAM_DIR, exist_ok=True)

def load_best_params():
    if os.path.exists(PARAM_FILE):
        try:
            with open(PARAM_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_best_params(new_params):
    try:
        with open(PARAM_FILE, "w") as f:
            json.dump(new_params, f, indent=4)
        if CONFIG["verbose"]:
            print(f"💾 Beste Parameter gespeichert: {PARAM_FILE}")
    except Exception as e:
        print("⚠️ Param save failed:", e)

# -----------------------------
# Features & Utilities
# -----------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Erwartung: Spalten 'date','open','high','low','close','volume','symbol' (close lowercase)
    if "close" not in df.columns:
        # try uppercase fallback
        for c in df.columns:
            if c.lower() == "close":
                df = df.rename(columns={c: "close"})
                break
    df = df.sort_values("date").reset_index(drop=True)
    df["return_1d"] = df["close"].pct_change()
    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)

    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["std_5"] = df["close"].rolling(5).std()
    df["std_20"] = df["close"].rolling(20).std()

    # === NEU: Kurzfristige technische Features (3-Tage) ===
    df["momentum_3d"] = df["close"] / df["close"].shift(3) - 1
    df["daily_return"] = df["close"].pct_change()
    df["std_3d"] = df["daily_return"].rolling(window=3).std()
    df["ma_3d"] = df["close"].rolling(window=3).mean()
    df.drop(columns=["daily_return"], inplace=True)

    df["rsi_14"] = rsi(df["close"], 14)
    df["ema_12"] = ema(df["close"], 12)
    df["ema_26"] = ema(df["close"], 26)
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = ema(df["macd"], 9)

    df["fwd_5d_return"] = df["close"].shift(-5) / df["close"] - 1.0
    df["target"] = (df["fwd_5d_return"] > 0).astype(int)

    df["vol_20d"] = df["return_1d"].rolling(20).std() * math.sqrt(252)

    df = df.dropna().reset_index(drop=True)
    return df

def download_history(tickers, years_back=3):
    """Lädt historische Kurse mit yfinance; gibt dict[ticker]=df"""
    import datetime
    start = datetime.datetime.now() - datetime.timedelta(days=365 * years_back)
    end = datetime.datetime.now()
    try:
        raw = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=True, threads=True, progress=False)
    except Exception as e:
        print("❌ Fehler beim Download:", e)
        return {}

    data = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            try:
                df = raw[t].reset_index().rename(columns=str.lower)
                if "date" not in df.columns:
                    df = df.rename(columns={df.columns[0]: "date"})
                df["symbol"] = t
                data[t] = df
            except Exception as e:
                if CONFIG["verbose"]:
                    print(f"⚠️ Fehler bei {t}: {e}")
    else:
        # Single ticker case
        df = raw.reset_index().rename(columns=str.lower)
        if "date" not in df.columns and df.shape[1] >= 1:
            df = df.rename(columns={df.columns[0]: "date"})
        sym = tickers[0] if isinstance(tickers, (list,tuple)) else tickers
        df["symbol"] = sym
        data[sym] = df
    return data

# -----------------------------
# Optuna Tuning (RF / XGB)
# -----------------------------
def tune_rf(X, y, n_trials=30):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "random_state": SEED,
            "n_jobs": -1
        }
        model = RandomForestClassifier(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring="roc_auc")
        return scores.mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    if CONFIG["verbose"]:
        print("🎯 Best RF params:", study.best_params)
    return study.best_params

def tune_xgb(X, y, n_trials=30):
    if not XGB_OK:
        return {}
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
            "random_state": SEED,
            "n_jobs": -1,
            "verbosity": 0,
            "use_label_encoder": False
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring="roc_auc")
        return scores.mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    if CONFIG["verbose"]:
        print("🎯 Best XGB params:", study.best_params)
    return study.best_params

# -----------------------------
# Model-Fit Funktionen
# -----------------------------
def fit_rf(X_tr, y_tr, **kwargs):
    clf = RandomForestClassifier(
        n_estimators=kwargs.get("n_estimators", 400),
        max_depth=kwargs.get("max_depth", 8),
        min_samples_leaf=kwargs.get("min_samples_leaf", 3),
        random_state=SEED,
        n_jobs=-1
    )
    clf.fit(X_tr, y_tr)
    return clf

def fit_xgb(X_tr, y_tr, **kwargs):
    if not XGB_OK:
        return None
    clf = XGBClassifier(
        n_estimators=kwargs.get("n_estimators", 600),
        max_depth=kwargs.get("max_depth", 6),
        learning_rate=kwargs.get("learning_rate", 0.05),
        subsample=kwargs.get("subsample", 0.9),
        colsample_bytree=kwargs.get("colsample_bytree", 0.8),
        reg_lambda=kwargs.get("reg_lambda", 1.0),
        random_state=SEED,
        n_jobs=-1,
        use_label_encoder=False,
        verbosity=0
    )
    clf.fit(X_tr, y_tr, verbose=False)
    return clf

def make_rnn_dataset(df_feat, features):
    X_seq, y_seq = [], []
    for sym, g in df_feat.groupby("symbol"):
        g = g.sort_values("date")
        vals = g[features].values
        targets = (g["fwd_5d_return"].values > 0).astype(float)
        if len(vals) <= CONFIG["seq_len"]:
            continue
        for i in range(CONFIG["seq_len"], len(vals)):
            X_seq.append(vals[i-CONFIG["seq_len"]:i])
            y_seq.append(targets[i])
    if len(X_seq) == 0:
        return None, None
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

def fit_lstm(X_tr, y_tr, X_val, y_val, input_dim):
    if not TF_OK or X_tr is None:
        return None
    model = Sequential([
        LSTM(64, input_shape=(CONFIG["seq_len"], input_dim), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=CONFIG["epochs"], batch_size=CONFIG["batch_size"], verbose=0)
    return model

def fit_gru(X_tr, y_tr, X_val, y_val, input_dim):
    if not TF_OK or X_tr is None:
        return None
    model = Sequential([
        GRU(64, input_shape=(CONFIG["seq_len"], input_dim), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=CONFIG["epochs"], batch_size=CONFIG["batch_size"], verbose=0)
    return model

# -----------------------------
# Backtest-Funktion (robust)
# -----------------------------
def backtest_classif(df_feat, estimator, features, ref_return_col, backtest_top_n=10, seq_len=20, backtest_freq="30D", min_train_size=200):
    """Rolling backtest: erstellt für jeden backtest-step ein neues Modell (clone(estimator)).
       estimator: sklearn estimator instance (unfitted)
    """
    results = []
    hit_rates = []
    mean_returns = []

    dates = sorted(df_feat["date"].unique())
    if len(dates) == 0:
        return pd.DataFrame(), np.nan, np.nan

    # Schrittweite in Tagen
    try:
        step = pd.to_timedelta(backtest_freq)
    except Exception:
        step = pd.Timedelta("30D")

    cur_date = dates[seq_len] if seq_len < len(dates) else dates[0]

    while cur_date <= dates[-6]:  # sicherstellen, dass future 5 Tage existiert
        # Trainingsfenster: alle Daten <= cur_date
        window = df_feat[df_feat["date"] <= cur_date]
        if len(window) < min_train_size:
            # advance
            cur_date = cur_date + step
            continue

        X_tr = window[features]
        y_tr = window["target"]

        # klone estimator und trainiere
        try:
            model = clone(estimator)
            model.fit(X_tr, y_tr)
        except Exception as e:
            if CONFIG["verbose"]:
                print("⚠️ Model fit failed in backtest:", e)
            cur_date = cur_date + step
            continue

        # heutige Kandidaten
        todays = df_feat[df_feat["date"] == cur_date].copy()
        if todays.empty:
            cur_date = cur_date + step
            continue
        X_today = todays[features]
        try:
            proba = model.predict_proba(X_today)[:,1]
        except Exception:
            proba = np.zeros(len(X_today), dtype=float)

        todays["p_up"] = proba
        todays["pred_up"] = (todays["p_up"] >= CONFIG["prob_threshold"]).astype(int)

        # künftige 5d Returns (innerhalb der nächsten 5 Kalendertage verfügbar in df_feat)
        future_mask = (df_feat["date"] > cur_date) & (df_feat["date"] <= cur_date + pd.Timedelta(5, "D"))
        future = df_feat[future_mask][["symbol", ref_return_col]].rename(columns={ref_return_col: f"{ref_return_col}_future"})
        merged = todays.merge(future, on="symbol", how="left")

        # defensive check – falls Spalte fehlt oder leer
        future_col = f"{ref_return_col}_future"
        if future_col not in merged.columns:
            cur_date = cur_date + step
            continue

        merged = merged.dropna(subset=[future_col])
        if merged.empty:
            cur_date = cur_date + step
            continue

        top_up = merged.nlargest(backtest_top_n, "p_up")
        top_down = merged.nsmallest(backtest_top_n, "p_up")
        if top_up.empty or top_down.empty:
            cur_date = cur_date + step
            continue

        hit_up = (top_up[future_col] > 0).sum() / len(top_up)
        hit_down = (top_down[future_col] < 0).sum() / len(top_down)
        avg_up = top_up[future_col].mean()
        avg_down = top_down[future_col].mean()

        hit_rate = (hit_up + hit_down) / 2.0
        avg_ret = (avg_up - avg_down) / 2.0

        results.append({
            "date": cur_date,
            "hit_up": hit_up,
            "hit_down": hit_down,
            "hit_rate": hit_rate,
            "avg_up": avg_up,
            "avg_down": avg_down,
            "avg_return": avg_ret
        })
        hit_rates.append(hit_rate)
        mean_returns.append(avg_ret)

        cur_date = cur_date + step

    return pd.DataFrame(results), (np.mean(hit_rates) if hit_rates else np.nan), (np.mean(mean_returns) if mean_returns else np.nan)

# -----------------------------
# MAIN Pipeline
# -----------------------------
def main():
    if CONFIG["verbose"]:
        print("📊 Lade S&P500 Ticker…")
    tickers = get_sp500_tickers()
    if CONFIG["max_tickers_analysis"] and CONFIG["max_tickers_analysis"] < len(tickers):
        tickers = tickers[: CONFIG["max_tickers_analysis"]]
    if CONFIG["verbose"]:
        print(f"✅ {len(tickers)} Ticker gewählt.")

    if CONFIG["verbose"]:
        print("⬇️ Lade Kurs-Historien… (dies kann je nach Anzahl Ticker etwas dauern)")
    raw = download_history(tickers, years_back=CONFIG["years_back"])
    if not raw:
        print("❌ Keine Kursdaten geladen. Beende.")
        return

    feats_list = []
    needed = [
        "return_1d","return_5d","return_10d",
        "ma_5","ma_10","ma_20",
        "std_5","std_20",
        "rsi_14","ema_12","ema_26",
        "macd","macd_signal",
        "vol_20d","fwd_5d_return","target"
    ]

    for t, df in raw.items():
        try:
            f = make_features(df)
            if all(col in f.columns for col in needed) and not f.empty:
                feats_list.append(f)
            else:
                if CONFIG["verbose"]:
                    print(f"⚠️ {t}: zu wenige Daten für Features – übersprungen.")
        except Exception as e:
            if CONFIG["verbose"]:
                print(f"⚠️ {t}: Feature-Fehler {e} – übersprungen.")

    if not feats_list:
        print("❌ Keine brauchbaren Daten für Features – Abbruch.")
        return

    # Erzeuge df_feat (HIER wird df_feat definiert – vorher fehlte evtl. diese Definition in einigen Varianten)
    df_feat = pd.concat(feats_list, ignore_index=True)
    df_feat = df_feat.sort_values(["symbol","date"]).reset_index(drop=True)

    # -------------------------------
    # 🔢 Fundamentaldaten Integration (robust, single block)
    # -------------------------------
    try:
        print("💰 Lade Fundamentaldaten (Yahoo + Macrotrends)…")
        symbols_for_funda = df_feat["symbol"].unique().tolist()[:CONFIG.get("max_tickers_analysis", 50)]
        df_funda = fetch_fundamentals_for_symbols(symbols_for_funda)

        if not df_funda.empty:
            # defensive: vereinheitliche Spaltennamen (kleinschreibung)
            df_funda.columns = [c.lower() for c in df_funda.columns]

            # bring keys ins dataframe (falls fehlen)
            for k in ["market_cap","trailing_pe","forward_pe","price_to_book","profit_margin",
                      "return_on_equity","total_debt","total_revenue","eps","revenue","roe"]:
                if k not in df_funda.columns:
                    df_funda[k] = None

            # Merge
            df_feat = df_feat.merge(df_funda, on="symbol", how="left")

            # Saubere numerische Spalten: konvertiere/parse numbers
            num_cols = ["market_cap","trailing_pe","forward_pe","price_to_book",
                        "profit_margin","return_on_equity","total_debt","total_revenue","eps","revenue","roe"]
            for c in num_cols:
                if c in df_feat.columns:
                    df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce")

            # Erstelle ratio/features aus Fundamentaldaten (nützlich fürs Modell)
            if "total_debt" in df_feat.columns and "total_revenue" in df_feat.columns:
                df_feat["debt_to_revenue"] = (df_feat["total_debt"] / df_feat["total_revenue"]).replace([np.inf, -np.inf], np.nan)
            if "market_cap" in df_feat.columns:
                df_feat["marketcap_log"] = np.log1p(df_feat["market_cap"].fillna(0.0))

            # Fill na mit sinnvollen Defaults (0 oder median wo angebracht)
            for c in ["trailing_pe","forward_pe","price_to_book","profit_margin","return_on_equity","roe","eps"]:
                if c in df_feat.columns:
                    med = df_feat[c].median(skipna=True)
                    df_feat[c] = df_feat[c].fillna(med if pd.notnull(med) else 0.0)

            for c in ["market_cap","total_debt","total_revenue","revenue","debt_to_revenue","marketcap_log"]:
                if c in df_feat.columns:
                    df_feat[c] = df_feat[c].fillna(0.0)

            print(f"✅ Fundamentaldaten erfolgreich integriert: {df_funda['symbol'].nunique()} Symbole")
        else:
            print("⚠️ Keine Fundamentaldaten geladen.")
    except Exception as e:
        print(f"❌ Fehler bei Fundamentaldaten: {e}")

    if CONFIG["verbose"]:
        print("ℹ️ df_feat shape:", df_feat.shape)
        print("ℹ️ Beispiele (erste 3 rows):")
        print(df_feat.head(3).to_string(index=False))

        # === 🔎 NEWS & SENTIMENT INTEGRATION ===
        print("📰 Lade News & Sentiment-Daten (NewsAPI + ggf. Firecrawl)...")

        # Datumsliste für Aggregation (z. B. tägliche Features)
        dates = pd.to_datetime(df_feat["date"].unique())

        # Symbol-Liste (hier kannst du später Unternehmensnamen ergänzen)
        symbols = df_feat["symbol"].unique()
        symbols_and_names = [(s, s) for s in symbols]  # falls keine Namen verfügbar

        all_news = []
        for sym, name in symbols_and_names[:CONFIG.get("max_tickers_analysis", 10)]:  # erst wenige zum Testen
            dfn = fetch_news_for_symbol(sym, name, window_days=7)
            if not dfn.empty:
                all_news.append(dfn)

        if all_news:
            df_news = pd.concat(all_news, ignore_index=True)
            print(f"✅ News geladen: {len(df_news)} Artikel.")
        else:
            df_news = pd.DataFrame(columns=["symbol", "publishedAt", "title", "description", "url"])
            print("⚠️ Keine News gefunden – möglicherweise API-Limit erreicht oder leere Anfrage.")

        # Feature-Aggregation (z. B. 24h Fenster)
        if not df_news.empty:
            news_feats = compute_news_features(df_news, dates)
            if "date" in news_feats.columns:
                news_feats["date"] = pd.to_datetime(news_feats["date"])
                df_feat["date"] = pd.to_datetime(df_feat["date"])
                df_feat = df_feat.merge(news_feats, how="left", on=["symbol", "date"])
                for c in ["news_count_24h", "news_sentiment_24h", "news_pos_ratio_24h", "news_neg_event_24h"]:
                    if c in df_feat.columns:
                        df_feat[c] = df_feat[c].fillna(0)
                print("✅ News-Features hinzugefügt:", [c for c in df_feat.columns if "news_" in c])
            else:
                print("⚠️ Keine 'date'-Spalte in News-Features gefunden.")
        else:
            print("⚠️ Keine News geladen – überspringe News-Merge.")

        # Fehlende Werte auffüllen (nochmal zur Sicherheit)
        for c in ["news_count_24h", "news_sentiment_24h", "news_pos_ratio_24h", "news_neg_event_24h"]:
            if c in df_feat.columns:
                df_feat[c] = df_feat[c].fillna(0)

        print("✅ News-Features hinzugefügt:", [c for c in df_feat.columns if "news_" in c])

    # -------------------------------
    # After data-prep: Feature selection + training setup
    # -------------------------------
    feature_cols = [
        "return_1d","return_5d","return_10d",
        "ma_3d","momentum_3d","std_3d",
        "ma_5","ma_10","ma_20",
        "std_5","std_20",
        "rsi_14","ema_12","ema_26",
        "macd","macd_signal",
        "vol_20d"
    ]
    target_col = "target"
    ref_return_col = "fwd_5d_return"

    existing_features = [f for f in feature_cols if f in df_feat.columns]
    if not existing_features:
        print("❌ Keine Feature-Spalten vorhanden – Abbruch.")
        raise SystemExit
    if target_col not in df_feat.columns:
        print("❌ Zielspalte fehlt – Abbruch.")
        raise SystemExit

    df_feat = df_feat.dropna(subset=existing_features + [target_col, ref_return_col])
    X = df_feat[existing_features]
    y = df_feat[target_col]

    # einfacher Zeitbasierter Split (80/20 chronologisch)
    split_idx = int(len(df_feat) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Optuna (optional)
    best_params_all = load_best_params()
    rf_params = best_params_all.get("rf", {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 3})
    xgb_params = best_params_all.get("xgb", {})

    if CONFIG["use_optuna"]:
        if CONFIG["verbose"]:
            print("🔍 Optuna läuft…")
        try:
            rf_params = tune_rf(X_train, y_train, n_trials=10)
            best_params_all["rf"] = rf_params
        except Exception as e:
            print("⚠️ Optuna RF failed:", e)
        if XGB_OK:
            try:
                xgb_params = tune_xgb(X_train, y_train, n_trials=10)
                best_params_all["xgb"] = xgb_params
            except Exception as e:
                print("⚠️ Optuna XGB failed:", e)
        save_best_params(best_params_all)

    rf = fit_rf(X_train, y_train, **rf_params)
    rf_val_proba = rf.predict_proba(X_test)[:,1]
    rf_val_pred = (rf_val_proba >= CONFIG["prob_threshold"]).astype(int)

    print("   RF  Acc={:.3f}  F1={:.3f}  AUC={:.3f}".format(
        accuracy_score(y_test, rf_val_pred),
        f1_score(y_test, rf_val_pred, zero_division=0),
        roc_auc_score(y_test, rf_val_proba)
    ))

    xgb = None
    xgb_val_proba = None
    if XGB_OK:
        try:
            xgb = fit_xgb(X_train, y_train, **xgb_params)
            if xgb is not None:
                xgb_val_proba = xgb.predict_proba(X_test)[:,1]
                xgb_val_pred = (xgb_val_proba >= CONFIG["prob_threshold"]).astype(int)
                print("   XGB Acc={:.3f}  F1={:.3f}  AUC={:.3f}".format(
                    accuracy_score(y_test, xgb_val_pred),
                    f1_score(y_test, xgb_val_pred, zero_division=0),
                    roc_auc_score(y_test, xgb_val_proba)
                ))
        except Exception as e:
            if CONFIG["verbose"]:
                print("⚠️ XGB train/predict failed:", e)

    # RNNs (optional)
    X_seq, y_seq = make_rnn_dataset(df_feat, existing_features)
    lstm, gru = None, None
    if CONFIG["train_rnn"] and TF_OK and X_seq is not None:
        n = len(X_seq)
        split = int(n * 0.8)
        X_seq_tr, y_seq_tr = X_seq[:split], y_seq[:split]
        X_seq_val, y_seq_val = X_seq[split:], y_seq[split:]
        lstm = fit_lstm(X_seq_tr, y_seq_tr, X_seq_val, y_seq_val, input_dim=len(existing_features))
        gru = fit_gru(X_seq_tr, y_seq_tr, X_seq_val, y_seq_val, input_dim=len(existing_features))
    elif CONFIG["train_rnn"] and not TF_OK:
        if CONFIG["verbose"]:
            print("ℹ️ TensorFlow nicht verfügbar - RNNs übersprungen.")

    # Ensemble auf Test set
    preds_proba = [rf_val_proba]
    weights = [0.6]
    if xgb_val_proba is not None:
        preds_proba.append(xgb_val_proba)
        weights.append(0.3)
    # (RNNs on test set would require mapping; skipped from ensemble unless explicitly available)
    if len(preds_proba) > 0:
        w = np.array(weights) / np.sum(weights)
        stacked_proba = np.average(np.vstack(preds_proba), axis=0, weights=w)
        stacked_pred = (stacked_proba >= CONFIG["prob_threshold"]).astype(int)
        print("📏 Ensemble  Acc={:.3f}  F1={:.3f}  AUC={:.3f}".format(
            accuracy_score(y_test, stacked_pred),
            f1_score(y_test, stacked_pred, zero_division=0),
            roc_auc_score(y_test, stacked_proba)
        ))

        # -------------------------------------
        # 🔧 Confidence-Filter (für 7D Stabilität)
        # -------------------------------------
        if CONFIG["backtest_freq"] == "7D":
            conf_mask = (stacked_proba > 0.55) | (stacked_proba < 0.45)
            X_conf = X_test[conf_mask]
            y_conf = y_test[conf_mask]
            if len(X_conf) > 50:
                if CONFIG["verbose"]:
                    print(f"🧩 Confidence-Filter aktiv: {len(X_conf)} von {len(X_test)} Test-Samples behalten.")
                rf_conf = fit_rf(pd.concat([X_train, X_conf]), pd.concat([y_train, y_conf]), **rf_params)
                rf = rf_conf
            else:
                if CONFIG["verbose"]:
                    print("⚠️ Confidence-Filter übersprungen (zu wenige sichere Beispiele).")

    else:
        stacked_proba = None

    # -----------------------------
    # Backtest (rolling) -- classification based
    # -----------------------------
    if CONFIG["run_backtest"]:
        if CONFIG["verbose"]:
            print("\\n🔎 Starte Backtest (Classification, rolling)...")
        rf_for_bt = RandomForestClassifier(**rf_params)
        bt_df, bt_hit_rate, bt_mean_ret = backtest_classif(df_feat, rf_for_bt, existing_features, ref_return_col, backtest_top_n=CONFIG["backtest_top_n"], seq_len=CONFIG["seq_len"], backtest_freq=CONFIG["backtest_freq"])
        if CONFIG["verbose"]:
            print(f"📈 Backtest (RF): Mean hit rate Top{CONFIG['backtest_top_n']} = {bt_hit_rate:.3f}  Mean 5d return = {bt_mean_ret:.4f}")
        if XGB_OK:
            try:
                xgb_for_bt = XGBClassifier(**xgb_params)
                bt_df_xgb, bt_hit_xgb, bt_mean_xgb = backtest_classif(df_feat, xgb_for_bt, existing_features, ref_return_col, backtest_top_n=CONFIG["backtest_top_n"], seq_len=CONFIG["seq_len"], backtest_freq=CONFIG["backtest_freq"])
                if CONFIG["verbose"]:
                    print(f"📈 Backtest (XGB): Mean hit rate Top{CONFIG['backtest_top_n']} = {bt_hit_xgb:.3f}  Mean 5d return = {bt_mean_xgb:.4f}")
            except Exception as e:
                if CONFIG["verbose"]:
                    print("⚠️ XGB backtest failed:", e)

    # Random benchmark (simple)
    np.random.seed(SEED)
    if CONFIG["run_backtest"] and 'bt_df' in locals() and not bt_df.empty:
        rand_returns = []
        for _ in range(len(bt_df)):
            rand_sample = df_feat.sample(CONFIG["backtest_top_n"])
            rand_returns.append(rand_sample[ref_return_col].mean())
        rand_returns = pd.Series(rand_returns)
        if CONFIG["verbose"]:
            print(f"🎲 Zufall Backtest: Mean 5d return = {rand_returns.mean():.4f}")

    # -----------------------------
    # Aktuelle Vorhersage (nächste 5 Tage)
    # -----------------------------
    latest_rows = df_feat.sort_values("date").groupby("symbol").tail(1).copy()
    X_now = latest_rows[existing_features].values

    final_proba = np.zeros(len(latest_rows), dtype=float)
    count_blenders = 0.0

    # RF
    try:
        final_proba += 0.7 * rf.predict_proba(X_now)[:,1]
        count_blenders += 0.7
    except Exception as e:
        if CONFIG["verbose"]:
            print("⚠️ RF predict failed:", e)

    # XGB
    if xgb is not None:
        try:
            final_proba += 0.25 * xgb.predict_proba(X_now)[:,1]
            count_blenders += 0.25
        except Exception as e:
            if CONFIG["verbose"]:
                print("⚠️ XGB predict failed:", e)

    if count_blenders > 0:
        final_proba = final_proba / count_blenders
    else:
        try:
            final_proba = rf.predict_proba(X_now)[:,1]
        except Exception:
            final_proba = np.zeros(len(latest_rows), dtype=float)

    latest_rows["p_up"] = final_proba
    latest_rows["pred_up"] = (latest_rows["p_up"] >= CONFIG["prob_threshold"]).astype(int)
    latest_rows["strength"] = latest_rows["p_up"].apply(lambda p: "strong_up" if p >= CONFIG["strong_p"] else ("strong_down" if p <= (1-CONFIG["strong_p"]) else ("up" if p > CONFIG["prob_threshold"] else ("down" if p < CONFIG["prob_threshold"] else "neutral"))))

    vol = latest_rows.get("vol_20d", pd.Series(np.nan, index=latest_rows.index)).fillna(latest_rows["vol_20d"].median() if "vol_20d" in latest_rows.columns else 0.0)
    latest_rows["risk_score"] = (vol.rank(pct=True) * 100).round(1)

    # Merge available actual future returns (falls vorhanden)
    latest_future = df_feat.sort_values("date").groupby("symbol").tail(1)[["symbol","fwd_5d_return"]]
    latest_rows = latest_rows.merge(latest_future, on="symbol", how="left")

    # sichere finale Ausgabe ohne fehlende Spalten
    cols = ["symbol","date","p_up","pred_up"]
    for optional in ["strength", "risk_score"]:
        if optional in latest_rows.columns:
            cols.append(optional)

    out = latest_rows[cols].sort_values("p_up", ascending=False).reset_index(drop=True)

    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output","predictions_next5d_classification.csv")
    if CONFIG["save_predictions"]:
        out.to_csv(out_path, index=False)

    print("\\n🏁 Top Prognosen (nächste 5 Börsentage) -- Klassifikation:")
    def fmt_prob(x): return f"{x*100:5.1f}%"
    print(out.head(CONFIG["backtest_top_n"]).to_string(index=False, formatters={
        "p_up": fmt_prob,
        "actual_5d_return": lambda x: f"{x*100:,.2f}%" if pd.notnull(x) else "NA",
        "risk_score": lambda x: f"{x:,.1f}"
    }))
    if CONFIG["save_predictions"]:
        print(f"\\n💾 Gespeichert: {out_path}")
    else:
        print("\\nℹ️ Speichern deaktiviert (CONFIG['save_predictions']=False)")

# if __name__ == "__main__":
#     main()
