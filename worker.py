# worker.py
import os
import sys
import json
import traceback
import datetime
import time
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
import threading

# damit dein Hauptskript gefunden wird
sys.path.append(os.path.dirname(__file__))

# dein existierendes Script
from analysispredict_next5_FIXED import main as run_pipeline

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
    with _lock:
        st = datetime.datetime.utcnow().isoformat() + "Z"
        write_status({"running": True, "start_time": st, "triggered_by": triggered_by, "note": None})
    try:
        start_local = datetime.datetime.now().isoformat()
        print(f"[worker] Starting run ({triggered_by}) at {start_local}")
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
        print(f"[worker] Run finished ({triggered_by})")
    except Exception as e:
        tb = traceback.format_exc()
        append_result({
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "triggered_by": triggered_by,
            "error": str(e),
            "traceback": tb
        })
        print("[worker] Run failed:", e)
    finally:
        with _lock:
            write_status({"running": False, "start_time": None, "triggered_by": None, "note": None})

# Scheduler config: times in CET (Europe/Berlin)
tz = pytz.timezone("Europe/Berlin")
sched = BlockingScheduler(timezone=tz)

# deine gewünschten Termine (ungefähr fertig sein sollen)
# 08:00 CET (vor EU-Opening)
sched.add_job(lambda: run_once("08:00_CET"), CronTrigger(hour=8, minute=0, timezone=tz))

# 15:30 CET (vor US-Opening)
sched.add_job(lambda: run_once("15:30_CET"), CronTrigger(hour=15, minute=30, timezone=tz))

# 17:30 CET (2h nach US-start; US open ≈ 15:30 CET)
sched.add_job(lambda: run_once("17:30_CET"), CronTrigger(hour=17, minute=30, timezone=tz))

# also optional: einmal am Tag ein Cleanup / Heartbeat
def heartbeat():
    print("[worker] heartbeat", datetime.datetime.now().isoformat())
sched.add_job(heartbeat, 'interval', minutes=30)

if __name__ == "__main__":
    print("[worker] Scheduler starting. Press Ctrl+C to exit.")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("[worker] Scheduler stopped.")
