# worker.py
import os, sys, json, traceback, threading, datetime
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

# init files
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)
if not os.path.exists(STATUS_FILE):
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump({"running": False, "start_time": None, "triggered_by": None, "note": None}, f)

_lock = threading.Lock()

def read_results():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def append_result(result):
    arr = read_results()
    arr.insert(0, result)
    arr = arr[:100]
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(arr, f, default=str)

def write_status(d):
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, default=str)

def run_once(triggered_by="scheduled"):
    """Lazy import to reduce startup time & RAM"""
    with _lock:
        st = datetime.datetime.utcnow().isoformat() + "Z"
        write_status({"running": True, "start_time": st, "triggered_by": triggered_by, "note": None})
    try:
        # Lazy import pipeline
        sys.path.append(BASE_DIR)
        from analysispredict_next5_FIXED import main as run_pipeline

        start_local = datetime.datetime.now().isoformat()
        print(f"[worker] Starting run ({triggered_by}) at {start_local}", flush=True)
        result = run_pipeline()

        rec = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "triggered_by": triggered_by,
            "result": result if isinstance(result, (dict, list, str, int, float)) else {"note": "run finished"},
        }
        if isinstance(result, dict):
            rec["accuracy"] = result.get("accuracy")
            rec["mean_5_day_return"] = result.get("mean_5_day_return")
            rec["backtest"] = result.get("backtest", None)
        append_result(rec)
        print(f"[worker] Run finished ({triggered_by})", flush=True)
    except Exception as e:
        tb = traceback.format_exc()
        append_result({
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "triggered_by": triggered_by,
            "error": str(e),
            "traceback": tb
        })
        print("[worker] Run failed:", e, flush=True)
    finally:
        with _lock:
            write_status({"running": False, "start_time": None, "triggered_by": None, "note": None})

# ---------------------------
# Scheduler (optional)
# ---------------------------
tz = pytz.timezone("Europe/Berlin")
sched = BackgroundScheduler(timezone=tz)
sched.add_job(lambda: threading.Thread(target=run_once, args=("08:00_CET",), daemon=True).start(),
              CronTrigger(hour=8, minute=0, timezone=tz))
sched.add_job(lambda: threading.Thread(target=run_once, args=("15:30_CET",), daemon=True).start(),
              CronTrigger(hour=15, minute=30, timezone=tz))
sched.add_job(lambda: threading.Thread(target=run_once, args=("17:30_CET",), daemon=True).start(),
              CronTrigger(hour=17, minute=30, timezone=tz))
sched.add_job(lambda: print(f"[worker] heartbeat {datetime.datetime.utcnow().isoformat()}"), 'interval', minutes=30)
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
    with _lock:
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                status = json.load(f)
            if status.get("running"):
                return JSONResponse({"status": "already_running", "start_time": status.get("start_time")}, status_code=409)
        except Exception:
            pass
    threading.Thread(target=run_once, args=(mode,), daemon=True).start()
    return {"status": "scheduled", "triggered_by": mode}

@app.get("/status")
def api_status():
    with open(STATUS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/results")
def api_results():
    return {"results": read_results()[:20]}

# ---------------------------
# Main block für Render
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("worker:app", host="0.0.0.0", port=port)
