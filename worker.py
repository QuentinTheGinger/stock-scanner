# worker.py
import os
import sys
import json
import traceback
import threading
import datetime
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(DATA_DIR, "analysis_results.json")
STATUS_FILE = os.path.join(DATA_DIR, "analysis_status.json")

# init files if missing
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)
if not os.path.exists(STATUS_FILE):
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump({"running": False, "start_time": None, "triggered_by": None, "note": None}, f)

_lock = threading.Lock()

def read_results():
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def append_result(result):
    arr = read_results()
    arr.insert(0, result)
    arr = arr[:100]
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(arr, f, default=str)

def write_status(d):
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, default=str)

def _now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def run_once(triggered_by="scheduled"):
    """Execute pipeline in a thread-safe way and store results. Lazy-imports the heavy pipeline."""
    with _lock:
        start_ts = _now_iso()
        write_status({"running": True, "start_time": start_ts, "triggered_by": triggered_by, "note": None})

    try:
        # Lazy import pipeline to reduce startup memory/time
        sys.path.append(BASE_DIR)
        try:
            from analysispredict_next5_FIXED import main as run_pipeline
        except Exception as e:
            raise RuntimeError(f"Failed to import analysis pipeline: {e}")

        start_local = datetime.datetime.now().isoformat()
        print(f"[worker] Starting run ({triggered_by}) at {start_local}", flush=True)

        result = run_pipeline()

        rec = {
            "timestamp": _now_iso(),
            "triggered_by": triggered_by,
            "result": result if isinstance(result, (dict, list, str, int, float)) else {"note": "run finished"},
        }
        if isinstance(result, dict):
            # copy common fields if present
            rec["accuracy"] = result.get("accuracy")
            rec["mean_5_day_return"] = result.get("mean_5_day_return")
            rec["backtest"] = result.get("backtest", None)

        append_result(rec)
        print(f"[worker] Run finished ({triggered_by})", flush=True)
    except Exception as e:
        tb = traceback.format_exc()
        err_rec = {
            "timestamp": _now_iso(),
            "triggered_by": triggered_by,
            "error": str(e),
            "traceback": tb,
        }
        append_result(err_rec)
        print("[worker] Run failed:", str(e), flush=True)
        print(tb, flush=True)
    finally:
        with _lock:
            write_status({"running": False, "start_time": None, "triggered_by": None, "note": None})

# ---------------------------
# Scheduler setup
# ---------------------------
tz = pytz.timezone("Europe/Berlin")
sched = BackgroundScheduler(timezone=tz)

# schedule desired times (CET)
sched.add_job(lambda: threading.Thread(target=run_once, args=("08:00_CET",), daemon=True).start(),
              CronTrigger(hour=8, minute=0, timezone=tz))
sched.add_job(lambda: threading.Thread(target=run_once, args=("15:30_CET",), daemon=True).start(),
              CronTrigger(hour=15, minute=30, timezone=tz))
sched.add_job(lambda: threading.Thread(target=run_once, args=("17:30_CET",), daemon=True).start(),
              CronTrigger(hour=17, minute=30, timezone=tz))

# heartbeat for logs
def heartbeat():
    print(f"[worker] heartbeat {datetime.datetime.utcnow().isoformat()}", flush=True)

sched.add_job(heartbeat, 'interval', minutes=30)
sched.start()

# ---------------------------
# FastAPI server
# ---------------------------
app = FastAPI(title="Analysis Worker")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/trigger")
def trigger(background_tasks: BackgroundTasks, mode: str = "manual"):
    # quick status check
    try:
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            status = json.load(f)
        if status.get("running"):
            return JSONResponse({"status": "already_running", "start_time": status.get("start_time")}, status_code=409)
    except Exception:
        pass

    # start run in background thread
    t = threading.Thread(target=run_once, args=(mode,), daemon=True)
    t.start()
    return {"status": "scheduled", "triggered_by": mode}

@app.get("/status")
def api_status():
    try:
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"running": False, "start_time": None, "triggered_by": None, "note": "status_read_error"}

@app.get("/results")
def api_results():
    return {"results": read_results()[:20]}

# run server when executed directly (python -m worker)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("worker:app", host="0.0.0.0", port=port)
