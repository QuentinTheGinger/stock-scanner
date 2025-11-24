#!/usr/bin/env python3
# analysispredict_next5_FIXED.py
# Neu implementierte, Render-Starter-kompatible Version (ohne TensorFlow)
# - Memory-schonend: verarbeitet Ticker sequenziell
# - Backtest optional (light/backtest)
# - News & Fundamentals integriert (limitierbar)
# - Returns: dict mit summary (für worker logging)

import os
import sys
import math
import gc
import json
import time
import warnings
from typing import List, Dict, Any, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# try imports (optional heavy libs)
try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

# small random seed stability
SEED = 42
np.random.seed(SEED)

# user-provided fetchers (keep as before)
# news_fetcher.fetch_news_for_symbol(sym, name, window_days)
# news_fetcher.compute_news_features(df_news, dates)
# fundamentals_fetcher.fetch_fundamentals_for_symbols(symbols)
try:
    from news_fetcher import fetch_news_for_symbol, compute_news_features
except Exception:
    # placeholder no-op implementations to avoid import errors
    def fetch_news_for_symbol(sym, name, window_days=7):
        return pd.DataFrame(columns=["symbol", "publishedAt", "title", "description", "url"])
    def compute_news_features(df_news, dates):
        return pd.DataFrame(columns=["symbol", "date"])

try:
    from fundamentals_fetcher import fetch_fundamentals_for_symbols
except Exception:
    def fetch_fundamentals_for_symbols(symbols: List[str]) -> pd.DataFrame:
        return pd.DataFrame(columns=["symbol"])

# optional: data loader
try:
    from data_loader.sp500_loader import get_sp500_tickers
except Exception:
    def get_sp500_tickers():
        # fallback small list
        return ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]

# yfinance - used per-ticker to avoid huge multiindex downloads
try:
    import yfinance as yf
except Exception:
    yf = None

# -----------------------------
# CONFIG
# -----------------------------
CONFIG = {
    # Data selection
    "max_tickers_analysis": 150,     # how many tickers to analyze (can be all)
    "max_news_tickers": 30,          # limit for news fetching (to save rate/ram)
    "years_back": 1,                 # history horizon (1 = 1 year)
    "start_date": None,              # optional fixed start date string "YYYY-MM-DD"

    # Models / training
    "use_optuna": False,
    "train_rnn": False,              # keep False: no TF
    "seq_len": 20,
    "epochs": 8,
    "batch_size": 64,

    # Backtest
    "run_backtest": False,           # keep default OFF for deploy-friendly
    "backtest_freq": "7D",
    "backtest_top_n": 20,
    "light_backtest": True,          # if True uses light method saving memory

    # thresholds
    "prob_threshold": 0.5,
    "strong_p": 0.75,

    # performance / memory
    "rf_n_estimators": 200,
    "xgb_n_estimators": 300,

    # misc
    "verbose": True,
    "save_predictions": True,
    "output_dir": "output",
    "parameters_dir": "parameters_models",
}

# ensure folders
os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(CONFIG["parameters_dir"], exist_ok=True)

PARAM_FILE = os.path.join(CONFIG["parameters_dir"], "best_params.json")

def save_json_safe(path: str, data: Any):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, default=str, indent=2)
    except Exception as e:
        if CONFIG["verbose"]:
            print("⚠️ failed saving json:", e)

def load_json_safe(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# -----------------------------
# Utility feature functions
# -----------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _now_iso():
    return datetime.utcnow().isoformat() + "Z"

# -----------------------------
# History download (per-ticker, memory-friendly)
# -----------------------------
def download_history_for_ticker(ticker: str, years_back: int = 1, start_date: str = None) -> pd.DataFrame:
    """Downloads historical OHLCV for a single ticker using yfinance (if available). Returns DataFrame or None."""
    if yf is None:
        return None

    end = datetime.now()
    if start_date:
        try:
            start = datetime.fromisoformat(start_date)
        except Exception:
            start = end - timedelta(days=365 * years_back)
    else:
        start = end - timedelta(days=365 * years_back)

    try:
        # Render-kompatibler Daten-Loader (Ticker.history statt yf.download)
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period="1y", interval="1d", auto_adjust=True)

        # falls Yahoo keine Daten zurückgibt → abbrechen
        if df is None or df.empty:
            raise ValueError(f"No price data returned for ticker {ticker}")

        # Reset Index + Spalten normalisieren
        df = df.reset_index().rename(columns=str.lower)

        # falls das Datum nicht korrekt heißt
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})

        df["symbol"] = ticker

        # ensure required columns exist
        for c in ["open", "high", "low", "close", "volume", "date", "symbol"]:
            if c not in df.columns:
                df[c] = pd.NA

        return df[["date", "open", "high", "low", "close", "volume", "symbol"]]

    except Exception as e:
        if CONFIG["verbose"]:
            print(f"⚠️ yfinance download failed for {ticker}: {e}")
        return None

# -----------------------------
# Feature creation (per-ticker)
# -----------------------------
def make_features_for_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ensure close column
    if "close" not in df.columns:
        for c in df.columns:
            if c.lower() == "close":
                df = df.rename(columns={c: "close"})
                break
    df = df.sort_values("date").reset_index(drop=True)
    # numeric conversions
    for c in ["close", "open", "high", "low", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # quick sanity
    if df["close"].isnull().all():
        return pd.DataFrame()

    df["return_1d"] = df["close"].pct_change()
    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)

    df["ma_3d"] = df["close"].rolling(3).mean()
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()

    df["std_3d"] = df["close"].pct_change().rolling(3).std()
    df["std_5"] = df["close"].pct_change().rolling(5).std()
    df["std_20"] = df["close"].pct_change().rolling(20).std()

    df["momentum_3d"] = df["close"] / df["close"].shift(3) - 1
    df["rsi_14"] = rsi(df["close"], 14)
    df["ema_12"] = ema(df["close"], 12)
    df["ema_26"] = ema(df["close"], 26)
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = ema(df["macd"], 9)

    df["fwd_5d_return"] = df["close"].shift(-5) / df["close"] - 1.0
    df["target"] = (df["fwd_5d_return"] > 0).astype(int)

    df["vol_20d"] = df["return_1d"].rolling(20).std() * math.sqrt(252)

    # drop rows with NaNs (due to rolling windows)
    df = df.dropna().reset_index(drop=True)
    return df

# -----------------------------
# Aggregate pipeline: stream over tickers
# -----------------------------
def build_feature_dataframe(tickers: List[str]) -> pd.DataFrame:
    """Downloads and builds features for tickers sequentially (memory-friendly)."""
    feats = []
    count = 0
    for t in tickers:
        count += 1
        if CONFIG["verbose"]:
            print(f"[data] processing {t} ({count}/{len(tickers)})", flush=True)
        df = download_history_for_ticker(t, years_back=CONFIG["years_back"], start_date=CONFIG["start_date"])
        if df is None or df.empty:
            continue
        try:
            f = make_features_for_df(df)
        except Exception as e:
            if CONFIG["verbose"]:
                print(f"⚠️ feature build failed for {t}: {e}", flush=True)
            continue
        if f.empty:
            continue
        feats.append(f)
        # free memory after each ticker
        del df
        gc.collect()
    if not feats:
        return pd.DataFrame()
    df_feat = pd.concat(feats, ignore_index=True)
    df_feat = df_feat.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df_feat

# -----------------------------
# Fundamentals & News integration
# -----------------------------
def merge_fundamentals(df_feat: pd.DataFrame, max_symbols: int = None) -> pd.DataFrame:
    if df_feat.empty:
        return df_feat
    symbols = list(df_feat["symbol"].unique())
    if max_symbols:
        symbols = symbols[:max_symbols]
    try:
        df_funda = fetch_fundamentals_for_symbols(symbols)
        if df_funda is None or df_funda.empty:
            if CONFIG["verbose"]:
                print("⚠️ No fundamentals returned.")
            return df_feat
        # normalize columns
        df_funda.columns = [c.lower() for c in df_funda.columns]
        # ensure symbol column exists
        if "symbol" not in df_funda.columns:
            return df_feat
        df_feat = df_feat.merge(df_funda, on="symbol", how="left")
        # numeric conversions for common columns if present
        for c in ["market_cap", "trailing_pe", "forward_pe", "price_to_book", "profit_margin", "return_on_equity", "total_debt", "total_revenue", "eps", "revenue", "roe"]:
            if c in df_feat.columns:
                df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce")
        # simple feature engineering
        if "total_debt" in df_feat.columns and "total_revenue" in df_feat.columns:
            df_feat["debt_to_revenue"] = (df_feat["total_debt"] / df_feat["total_revenue"]).replace([np.inf, -np.inf], np.nan)
        if "market_cap" in df_feat.columns:
            df_feat["marketcap_log"] = np.log1p(df_feat["market_cap"].fillna(0.0))
        # fill na with sensible defaults
        df_feat = df_feat.fillna(method="ffill").fillna(0)
        if CONFIG["verbose"]:
            print("[funda] merged fundamentals", flush=True)
    except Exception as e:
        if CONFIG["verbose"]:
            print("⚠️ fundamentals merge failed:", e, flush=True)
    return df_feat

def merge_news_features(df_feat: pd.DataFrame, max_symbols: int = None) -> pd.DataFrame:
    if df_feat.empty:
        return df_feat
    symbols = list(df_feat["symbol"].unique())
    if max_symbols:
        symbols = symbols[:max_symbols]
    all_news = []
    # limit news tickers to avoid rate limits / memory
    for sym in symbols:
        try:
            news_df = fetch_news_for_symbol(sym, sym, window_days=7)
            if news_df is None or news_df.empty:
                continue
            all_news.append(news_df)
        except Exception:
            continue
    if not all_news:
        return df_feat
    df_news = pd.concat(all_news, ignore_index=True)
    try:
        dates = pd.to_datetime(df_feat["date"].unique())
        news_feats = compute_news_features(df_news, dates)
        if "date" in news_feats.columns:
            news_feats["date"] = pd.to_datetime(news_feats["date"])
            df_feat["date"] = pd.to_datetime(df_feat["date"])
            df_feat = df_feat.merge(news_feats, how="left", on=["symbol", "date"])
            for c in news_feats.columns:
                if c not in df_feat.columns:
                    df_feat[c] = 0
            df_feat = df_feat.fillna(0)
            if CONFIG["verbose"]:
                print("[news] merged news features", flush=True)
    except Exception as e:
        if CONFIG["verbose"]:
            print("⚠️ news merge failed:", e, flush=True)
    return df_feat

# -----------------------------
# Model helpers (RF + optional XGB)
# -----------------------------
def fit_rf(X_tr, y_tr, params=None):
    p = params or {}
    clf = RandomForestClassifier(
        n_estimators=int(p.get("n_estimators", CONFIG["rf_n_estimators"])),
        max_depth=int(p.get("max_depth", 8)),
        min_samples_leaf=int(p.get("min_samples_leaf", 3)),
        random_state=SEED,
        n_jobs=1
    )
    clf.fit(X_tr, y_tr)
    return clf

def fit_xgb(X_tr, y_tr, params=None):
    if not XGB_OK:
        return None
    p = params or {}
    clf = XGBClassifier(
        n_estimators=int(p.get("n_estimators", CONFIG["xgb_n_estimators"])),
        max_depth=int(p.get("max_depth", 6)),
        learning_rate=float(p.get("learning_rate", 0.05)),
        subsample=float(p.get("subsample", 0.9)),
        colsample_bytree=float(p.get("colsample_bytree", 0.8)),
        reg_lambda=float(p.get("reg_lambda", 1.0)),
        random_state=SEED,
        n_jobs=1,
        verbosity=0,
        use_label_encoder=False
    )
    clf.fit(X_tr, y_tr, verbose=False)
    return clf

# -----------------------------
# Light backtest (low-memory)
# -----------------------------
def light_backtest(df_feat: pd.DataFrame, estimator, features: List[str], ref_return_col: str, top_n: int = 10) -> Tuple[pd.DataFrame, float, float]:
    """Light rolling backtest: sample only a subset of dates to reduce memory & time."""
    results = []
    dates = sorted(df_feat["date"].unique())
    if len(dates) < 10:
        return pd.DataFrame(), float("nan"), float("nan")
    # sample up to 30 timepoints evenly
    npoints = min(30, len(dates)//2)
    idxs = np.linspace(0, len(dates)-6, npoints, dtype=int)
    for idx in idxs:
        cur_date = dates[int(idx)]
        window = df_feat[df_feat["date"] <= cur_date]
        if len(window) < 100:
            continue
        X_tr = window[features]
        y_tr = window["target"]
        try:
            model = clone(estimator)
            model.fit(X_tr, y_tr)
        except Exception:
            continue
        todays = df_feat[df_feat["date"] == cur_date]
        if todays.empty:
            continue
        try:
            proba = model.predict_proba(todays[features])[:,1]
        except Exception:
            proba = np.zeros(len(todays))
        todays = todays.copy()
        todays["p_up"] = proba
        if len(todays) < top_n:
            continue
        top_up = todays.nlargest(top_n, "p_up")
        hit = (top_up[ref_return_col] > 0).sum() / len(top_up)
        mean_ret = top_up[ref_return_col].mean()
        results.append({"date": cur_date, "hit_rate": hit, "mean_return": mean_ret})
    if not results:
        return pd.DataFrame(), float("nan"), float("nan")
    dfres = pd.DataFrame(results)
    return dfres, dfres["hit_rate"].mean(), dfres["mean_return"].mean()

# -----------------------------
# Full pipeline main
# -----------------------------
def main() -> Dict[str, Any]:
    start_ts = time.time()
    summary = {
        "timestamp": _now_iso(),
        "tickers_analyzed": 0,
        "backtest_mean_5d_return": None,
        "accuracy": None,
        "mean_5_day_return": None,
        "error": None
    }
    try:
        if CONFIG["verbose"]:
            print("▶ Starting analysis pipeline", flush=True)

        # 1) tickers
        tickers = get_sp500_tickers()
        if CONFIG["max_tickers_analysis"] and CONFIG["max_tickers_analysis"] < len(tickers):
            tickers = tickers[: CONFIG["max_tickers_analysis"]]
        if CONFIG["verbose"]:
            print(f"[init] using {len(tickers)} tickers", flush=True)

        # 2) build features (sequential to save memory)
        df_feat = build_feature_dataframe(tickers)
        if df_feat.empty:
            raise RuntimeError("No feature data available after download")

        # 3) merge fundamentals (limited)
        df_feat = merge_fundamentals(df_feat, max_symbols=None)  # you requested all fundamentals stay
        # 4) news (limit to max_news_tickers to save rate/memory)
        df_feat = merge_news_features(df_feat, max_symbols=CONFIG["max_news_tickers"])

        if CONFIG["verbose"]:
            print("[data] final df_feat shape:", df_feat.shape, flush=True)
            if CONFIG["verbose"]:
                print(df_feat.head(2).to_string(index=False), flush=True)

        # 5) features selection
        feature_cols = [
            "return_1d","return_5d","return_10d",
            "ma_3d","momentum_3d","std_3d",
            "ma_5","ma_10","ma_20",
            "std_5","std_20",
            "rsi_14","ema_12","ema_26",
            "macd","macd_signal",
            "vol_20d",
            # fundamentals/news columns may be present dynamically
        ]
        # keep only available
        existing_features = [f for f in feature_cols if f in df_feat.columns]
        # try to add some numeric fundamentals if present
        for c in ["marketcap_log", "debt_to_revenue", "trailing_pe", "profit_margin"]:
            if c in df_feat.columns:
                existing_features.append(c)

        if not existing_features:
            raise RuntimeError("No features available for modeling")

        ref_return_col = "fwd_5d_return"
        if ref_return_col not in df_feat.columns:
            raise RuntimeError("Reference return column missing")

        # drop rows with NaNs in required cols
        df_feat = df_feat.dropna(subset=existing_features + [ref_return_col, "target"])
        if df_feat.empty:
            raise RuntimeError("No rows left after dropping NaNs")

        # simple time-based split
        df_feat = df_feat.sort_values("date").reset_index(drop=True)
        split_idx = int(len(df_feat) * 0.8)
        train = df_feat.iloc[:split_idx]
        test = df_feat.iloc[split_idx:]

        X_train = train[existing_features]
        y_train = train["target"]
        X_test = test[existing_features]
        y_test = test["target"]

        # 6) train models
        # load saved params (if any)
        best_params = load_json_safe(PARAM_FILE)
        rf_params = best_params.get("rf", {})
        xgb_params = best_params.get("xgb", {})

        rf = fit_rf(X_train, y_train, params=rf_params)
        rf_val_proba = rf.predict_proba(X_test)[:,1]
        rf_val_pred = (rf_val_proba >= CONFIG["prob_threshold"]).astype(int)

        acc = accuracy_score(y_test, rf_val_pred)
        f1 = f1_score(y_test, rf_val_pred, zero_division=0)
        auc = roc_auc_score(y_test, rf_val_proba)
        summary["accuracy"] = acc

        if CONFIG["verbose"]:
            print(f"[model] RF Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f}", flush=True)

        xgb = None
        xgb_val_proba = None
        if XGB_OK:
            try:
                xgb = fit_xgb(X_train, y_train, params=xgb_params)
                xgb_val_proba = xgb.predict_proba(X_test)[:,1]
                if CONFIG["verbose"]:
                    print("[model] XGB trained", flush=True)
            except Exception as e:
                if CONFIG["verbose"]:
                    print("⚠️ XGB train failed:", e, flush=True)

        # stacking / ensemble
        preds_proba = [rf_val_proba]
        weights = [0.7]
        if xgb_val_proba is not None:
            preds_proba.append(xgb_val_proba)
            weights.append(0.3)
        w = np.array(weights) / np.sum(weights)
        stacked_proba = np.average(np.vstack(preds_proba), axis=0, weights=w)
        stacked_pred = (stacked_proba >= CONFIG["prob_threshold"]).astype(int)

        if CONFIG["verbose"]:
            print("[model] Ensemble results:", flush=True)
            try:
                print(f" Acc={accuracy_score(y_test, stacked_pred):.3f} F1={f1_score(y_test, stacked_pred, zero_division=0):.3f} AUC={roc_auc_score(y_test, stacked_proba):.3f}", flush=True)
            except Exception:
                pass

        # 7) backtest (optional)
        mean_5d_return = None
        if CONFIG["run_backtest"]:
            if CONFIG["light_backtest"]:
                bt_df, bt_hit, bt_mean = light_backtest(df_feat, RandomForestClassifier(**rf.get_params()), existing_features, ref_return_col, top_n=CONFIG["backtest_top_n"])
            else:
                # full backtest would be heavy; removed in starter version
                bt_df, bt_hit, bt_mean = pd.DataFrame(), float("nan"), float("nan")
            summary["backtest_mean_5d_return"] = bt_mean
            mean_5d_return = bt_mean
            if CONFIG["verbose"]:
                print("[backtest] mean_5d_return:", bt_mean, flush=True)

        # 8) now produce final predictions for latest date per symbol
        latest_rows = df_feat.sort_values("date").groupby("symbol").tail(1).copy()
        X_now = latest_rows[existing_features].values

        final_proba = np.zeros(len(latest_rows), dtype=float)
        count_weights = 0.0
        try:
            final_proba += 0.7 * rf.predict_proba(X_now)[:,1]
            count_weights += 0.7
        except Exception:
            pass
        if xgb is not None:
            try:
                final_proba += 0.3 * xgb.predict_proba(X_now)[:,1]
                count_weights += 0.3
            except Exception:
                pass
        if count_weights > 0:
            final_proba = final_proba / count_weights
        else:
            try:
                final_proba = rf.predict_proba(X_now)[:,1]
            except Exception:
                final_proba = np.zeros(len(latest_rows))

        latest_rows["p_up"] = final_proba
        latest_rows["pred_up"] = (latest_rows["p_up"] >= CONFIG["prob_threshold"]).astype(int)
        latest_rows["strength"] = latest_rows["p_up"].apply(lambda p: "strong_up" if p >= CONFIG["strong_p"] else ("strong_down" if p <= (1-CONFIG["strong_p"]) else ("up" if p > CONFIG["prob_threshold"] else ("down" if p < CONFIG["prob_threshold"] else "neutral"))))
        latest_rows["risk_score"] = (latest_rows.get("vol_20d", pd.Series(np.nan, index=latest_rows.index)).rank(pct=True) * 100).round(1)

        cols = ["symbol", "date", "p_up", "pred_up", "strength", "risk_score"]
        out = latest_rows[cols].sort_values("p_up", ascending=False).reset_index(drop=True)

        # save output
        out_path = os.path.join(CONFIG["output_dir"], "predictions_next5d_classification.csv")
        if CONFIG["save_predictions"]:
            try:
                out.to_csv(out_path, index=False)
                if CONFIG["verbose"]:
                    print("[output] saved predictions to", out_path, flush=True)
            except Exception as e:
                if CONFIG["verbose"]:
                    print("⚠️ failed to save predictions:", e, flush=True)

        # summary
        summary["tickers_analyzed"] = len(latest_rows)
        summary["mean_5_day_return"] = mean_5d_return
        summary["accuracy"] = summary.get("accuracy", acc)

        # return structured result for worker
        runtime = time.time() - start_ts
        summary["runtime_seconds"] = runtime
        if CONFIG["verbose"]:
            print("✅ Pipeline finished in {:.1f}s".format(runtime), flush=True)

        return {"status": "ok", "summary": summary, "predictions_path": out_path}
    except Exception as e:
        tb = None
        try:
            import traceback as _tb
            tb = _tb.format_exc()
        except Exception:
            tb = str(e)
        if CONFIG["verbose"]:
            print("❌ Pipeline failed:", e, flush=True)
            print(tb, flush=True)
        summary["error"] = str(e)
        summary["traceback"] = tb
        return {"status": "error", "summary": summary}

# end of file
