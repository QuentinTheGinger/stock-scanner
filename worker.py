# worker.py
import os
import sys
import json
import traceback
import threading
import datetime
from typing import Optional, Any, Dict, List

from fastapi import FastAPI, BackgroundTasks, Header, HTTPException
from fastapi.responses import JSONResponse

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
import gc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_FILE = os.path.join(DATA_DIR, "analysis_results.json")
STATUS_FILE = os.path.join(DATA_DIR, "analysis_status.json")
PREDICTIONS_JSON = os.path.join(OUTPUT_DIR, "predictions_latest.json")

# init files if missing
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)
if not os.path.exists(STATUS_FILE):
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump({"running": False, "start_time": None, "triggered_by": None, "note": None}, f)

_lock = threading.Lock()

def read_json_file(path: str, default: Any):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def write_json_file(path: str, data: Any):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, default=str, indent=2)
    except Exception as e:
        print(f"[worker] failed to write {path}: {e}", flush=True)

def append_result(record: Dict):
    arr = read_json_file(RESULTS_FILE, [])
    arr.insert(0, record)
    arr = arr[:200]
    write_json_file(RESULTS_FILE, arr)

def _now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def _update_status(running: bool, triggered_by: Optional[str] = None, note: Optional[str] = None):
    write_json_file(STATUS_FILE, {"running": running, "start_time": _now_iso() if running else None, "triggered_by": triggered_by, "note": note})

def _save_predictions_from_result(result: Any):
    """
    If the analysis returns a path to predictions (CSV) or a DataFrame-like structure,
    convert to JSON that frontend can read easily.
    Expected result dict keys:
      - 'predictions_path' (str) OR
      - 'predictions' (list of dicts)
    This writes PREDICTIONS_JSON.
    """
    try:
        if isinstance(result, dict) and result.get("predictions_path"):
            p = result.get("predictions_path")
            # try to read CSV if exists
            if os.path.exists(p):
                import csv
                rows = []
                with open(p, "r", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    for r in reader:
                        # convert numeric strings if possible
                        rows.append({k: _try_parse_number(v) for k, v in r.items()})
                write_json_file(PREDICTIONS_JSON, rows)
                return True
        if isinstance(result, dict) and result.get("predictions") and isinstance(result["predictions"], list):
            write_json_file(PREDICTIONS_JSON, result["predictions"])
            return True
    except Exception as e:
        print("[worker] _save_predictions_from_result failed:", e, flush=True)
    return False

def _try_parse_number(v: Any):
    try:
        if v is None:
            return None
        s = str(v)
        if s == "":
            return None
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return v

def run_once(triggered_by: str = "scheduled"):
    """
    Run the heavy analysis pipeline (lazy import).
    Writes results to RESULTS_FILE and writes a predictions JSON for the frontend.
    """
    with _lock:
        _update_status(True, triggered_by, note=None)

    start_local = datetime.datetime.now().isoformat()
    print(f"[worker] Starting run ({triggered_by}) at {start_local}", flush=True)
    result_record: Dict[str, Any] = {"timestamp": _now_iso(), "triggered_by": triggered_by}

    try:
        # Ensure we can import the analysis function
        sys.path.append(BASE_DIR)
        try:
            from analysispredict_next5_FIXED import main as run_pipeline
        except Exception as e:
            raise RuntimeError(f"Failed importing analysis pipeline: {e}")

        # Run pipeline (this may take a while)
        pipeline_result = run_pipeline()  # expected to be dict {"status":"ok","summary":..., "predictions_path":...}
        result_record["result"] = pipeline_result

        # copy common summary fields for quick glance
        if isinstance(pipeline_result, dict):
            summary = pipeline_result.get("summary", {})
            result_record["summary"] = summary
            # pick accuracy / mean_5_day_return if present
            for k in ("accuracy", "mean_5_day_return", "backtest_mean_5d_return"):
                if k in summary:
                    result_record[k] = summary.get(k)

        # try to save predictions JSON for frontend convenience
        try:
            saved = _save_predictions_from_result(pipeline_result)
            if saved:
                result_record["predictions_saved"] = True
            else:
                result_record["predictions_saved"] = False
        except Exception as e:
            result_record["predictions_saved"] = False
            print("[worker] error saving predictions:", e, flush=True)

        append_result(result_record)
        print(f"[worker] Run finished ({triggered_by})", flush=True)
    except Exception as e:
        tb = traceback.format_exc()
        print("[worker] Run failed:", str(e), flush=True)
        print(tb, flush=True)
        result_record["error"] = str(e)
        result_record["traceback"] = tb
        append_result(result_record)
    finally:
        # free memory and update status
        try:
            gc.collect()
        except Exception:
            pass
        with _lock:
            _update_status(False, None, note=None)

# ---------------------------
# Scheduler setup
# ---------------------------
tz = pytz.timezone("Europe/Berlin")
sched = BackgroundScheduler(timezone=tz)

# schedule desired times (CET) — adjust hours as you like
sched.add_job(lambda: threading.Thread(target=run_once, args=("08:00_CET",), daemon=True).start(),
              CronTrigger(hour=8, minute=0, timezone=tz))
sched.add_job(lambda: threading.Thread(target=run_once, args=("15:30_CET",), daemon=True).start(),
              CronTrigger(hour=15, minute=30, timezone=tz))
sched.add_job(lambda: threading.Thread(target=run_once, args=("17:30_CET",), daemon=True).start(),
              CronTrigger(hour=17, minute=30, timezone=tz))

# heartbeat for logs
def heartbeat():
    print(f"[worker] heartbeat {_now_iso()}", flush=True)

sched.add_job(heartbeat, 'interval', minutes=30)
sched.start()

# ---------------------------
# FastAPI server
# ---------------------------
app = FastAPI(title="Analysis Worker")

# Optional token (set WORKER_TOKEN env var). If set, POST /trigger and protected endpoints require header: x-worker-token
WORKER_TOKEN = os.environ.get("WORKER_TOKEN")

def _check_token(header_token: Optional[str]):
    if WORKER_TOKEN:
        if not header_token or header_token != WORKER_TOKEN:
            raise HTTPException(status_code=401, detail="invalid token")

@app.get("/healthz")
def healthz():
    return {"ok": True, "time": _now_iso()}

@app.post("/trigger")
def trigger(background_tasks: BackgroundTasks, mode: str = "manual", x_worker_token: Optional[str] = Header(None, convert_underscores=False)):
    # token check
    _check_token(x_worker_token)
    # quick status check
    try:
        status = read_json_file(STATUS_FILE, {"running": False})
        if status.get("running"):
            return JSONResponse({"status": "already_running", "start_time": status.get("start_time")}, status_code=409)
    except Exception:
        pass

    # schedule the run in a background thread to return immediately
    t = threading.Thread(target=run_once, args=(mode,), daemon=True)
    t.start()
    return {"status": "scheduled", "triggered_by": mode, "timestamp": _now_iso()}

@app.get("/status")
def api_status():
    return read_json_file(STATUS_FILE, {"running": False, "start_time": None, "triggered_by": None, "note": None})

@app.get("/results")
def api_results():
    """Return the last N run records (summary)."""
    arr = read_json_file(RESULTS_FILE, [])
    return {"results": arr[:20]}

@app.get("/predictions")
def api_predictions():
    """
    Return latest predictions in JSON form for frontend.
    If PREDICTIONS_JSON exists, return it.
    Otherwise, try to return predictions from the last run (if present in results).
    """
    preds = read_json_file(PREDICTIONS_JSON, None)
    if preds:
        return {"predictions": preds, "source": "predictions_json"}
    # fallback: look at latest results entry
    arr = read_json_file(RESULTS_FILE, [])
    if arr:
        latest = arr[0]
        # try to extract predictions field
        if isinstance(latest.get("result"), dict) and latest["result"].get("predictions"):
            return {"predictions": latest["result"]["predictions"], "source": "inline_result"}
        # else return summary
        return {"predictions": None, "summary": latest.get("summary", {})}
    return {"predictions": None, "summary": {}}

# allow running with `python -m worker` for local debug (Render will use `uvicorn worker:app`)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("worker:app", host="0.0.0.0", port=port)
