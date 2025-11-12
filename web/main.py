# web/main.py
from fastapi import FastAPI, BackgroundTasks, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import threading, json, os, sys, traceback, datetime
from typing import Optional

# Pfad für Hauptskript
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from analysispredict_next5_FIXED import main as run_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(DATA_DIR, "analysis_results.json")
STATUS_FILE = os.path.join(DATA_DIR, "analysis_status.json")

# Sicherstellen, dass JSON-Dateien existieren
for path, default_content in [
    (RESULTS_FILE, []),
    (STATUS_FILE, {"running": False, "start_time": None, "pid": None, "note": None}),
]:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_content, f)

app = FastAPI(title="Aktienprognose Dashboard")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

_lock = threading.Lock()

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, default=str, indent=2)

def _run_analysis(triggered_by="manual"):
    with _lock:
        status = {"running": True, "start_time": datetime.datetime.utcnow().isoformat() + "Z"}
        write_json(STATUS_FILE, status)
    try:
        result = run_pipeline()
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "triggered_by": triggered_by,
            "result": result if isinstance(result, (dict, list, str, int, float)) else {"note": "run finished"},
        }
        results = read_json(RESULTS_FILE)
        results.insert(0, record)
        results = results[:100]
        write_json(RESULTS_FILE, results)
    except Exception as e:
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "triggered_by": triggered_by,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        results = read_json(RESULTS_FILE)
        results.insert(0, record)
        write_json(RESULTS_FILE, results)
    finally:
        with _lock:
            write_json(STATUS_FILE, {"running": False, "start_time": None})

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    results = read_json(RESULTS_FILE)[:6]
    status = read_json(STATUS_FILE)
    return templates.TemplateResponse("index.html", {"request": request, "results": results, "status": status})

@app.post("/start")
def start_analysis(trigger: Optional[str] = Form("manual")):
    status = read_json(STATUS_FILE)
    if status.get("running"):
        return JSONResponse({"status": "already_running"}, status_code=409)
    threading.Thread(target=_run_analysis, args=(trigger,), daemon=True).start()
    return JSONResponse({"status": "started", "triggered_by": trigger})

@app.get("/status")
def api_status():
    return read_json(STATUS_FILE)

@app.get("/results")
def api_results():
    return {"results": read_json(RESULTS_FILE)[:20]}

@app.get("/healthz")
def healthz():
    return {"ok": True}

# 🟠 Wichtig: nur hier Server starten
if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("web.main:app", host="0.0.0.0", port=port)
