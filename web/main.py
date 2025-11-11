# web/main.py
from fastapi import FastAPI, BackgroundTasks, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import threading, time, json, os, sys, traceback, datetime
from typing import Optional

# damit dein Hauptskript gefunden wird
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from analysispredict_next5_FIXED import main as run_pipeline  # dein existierendes Script

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(DATA_DIR, "analysis_results.json")
STATUS_FILE = os.path.join(DATA_DIR, "analysis_status.json")

# initial files if missing
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)
if not os.path.exists(STATUS_FILE):
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump({"running": False, "start_time": None, "pid": None, "note": None}, f)

app = FastAPI(title="Aktienprognose API + Dashboard")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

_lock = threading.Lock()

def read_status():
    with open(STATUS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def write_status(d):
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, default=str)

def read_results():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def append_result(result):
    arr = read_results()
    arr.insert(0, result)  # newest first
    # keep only last 100 (safety)
    arr = arr[:100]
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(arr, f, default=str)

def _run_analysis_wrapper(triggered_by="manual"):
    """Wrapper to run your pipeline and store outputs / backtest summary."""
    with _lock:
        st = datetime.datetime.utcnow().isoformat() + "Z"
        status = {"running": True, "start_time": st, "triggered_by": triggered_by, "note": None}
        write_status(status)
    try:
        # run_pipeline() should return either dict/list or None.
        result = run_pipeline()
        # Build result record
        rec = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "triggered_by": triggered_by,
            "result": result if isinstance(result, (dict, list, str, int, float)) else {"note": "run finished"},
        }
        # try to extract accuracy / mean5 from result if available
        # (best-effort: if result is dict and has keys)
        if isinstance(result, dict):
            rec["accuracy"] = result.get("accuracy")
            rec["mean_5_day_return"] = result.get("mean_5_day_return")
            rec["backtest"] = result.get("backtest", None)
        append_result(rec)
    except Exception as e:
        tb = traceback.format_exc()
        append_result({
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "triggered_by": triggered_by,
            "error": str(e),
            "traceback": tb
        })
    finally:
        with _lock:
            write_status({"running": False, "start_time": None, "triggered_by": None, "note": None})

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    """Dashboard page (Jinja2)"""
    results = read_results()[:6]  # last 6
    status = read_status()
    # prepare table rows: ensure backtest key exists or "none"
    for r in results:
        if "backtest" not in r:
            r["backtest"] = r.get("result", {}).get("backtest") if isinstance(r.get("result"), dict) else None
    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": results,
        "status": status,
        "last6_backtests": [ (r.get("backtest") if r.get("backtest") is not None else "none") for r in results ],
    })

@app.post("/start")
def start_analysis(trigger: Optional[str] = Form("manual")):
    """Start analysis via POST (button). Prevent concurrent runs."""
    status = read_status()
    if status.get("running"):
        return JSONResponse({"status": "already_running", "start_time": status.get("start_time")}, status_code=409)

    # start background thread
    t = threading.Thread(target=_run_analysis_wrapper, args=(trigger,), daemon=True)
    t.start()
    return JSONResponse({"status": "started", "triggered_by": trigger})

@app.get("/status")
def api_status():
    """JSON status for polling in frontend"""
    return read_status()

@app.get("/results")
def api_results():
    """JSON results for last analyses"""
    return {"results": read_results()[:20]}  # expose last 20

# quick health endpoint
@app.get("/healthz")
def healthz():
    return {"ok": True}
