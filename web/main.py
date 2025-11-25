# web/main.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os, json, requests, logging
from typing import Optional, Any, Dict

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("web.main")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(DATA_DIR, "analysis_results.json")
STATUS_FILE = os.path.join(DATA_DIR, "analysis_status.json")

# Ensure local fallback files exist (frontend can still read them if worker unreachable)
for path, default_content in [
    (RESULTS_FILE, []),
    (STATUS_FILE, {"running": False, "start_time": None, "pid": None, "note": None}),
]:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_content, f)

app = FastAPI(title="Aktienprognose Dashboard")
# static & templates unchanged
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Worker settings from environment
WORKER_URL = os.environ.get("WORKER_URL")  # e.g. https://stock-scanner-worker-bk5g.onrender.com
# If you put the worker token into the frontend environment (so frontend can call worker),
# set FRONTEND_WORKER_TOKEN on the frontend service to the same value as the worker's WORKER_TOKEN.
FRONTEND_WORKER_TOKEN = os.environ.get("WORKER_TOKEN") or os.environ.get("FRONTEND_WORKER_TOKEN")
# timeouts
HTTP_TIMEOUT = float(os.environ.get("WORKER_HTTP_TIMEOUT", "10"))

def read_json_local(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _worker_get(path: str, timeout: float = HTTP_TIMEOUT) -> Dict[str, Any]:
    """GET from worker, return {'ok': False, 'error':...} on failure"""
    if not WORKER_URL:
        return {"ok": False, "error": "WORKER_URL not configured"}
    url = WORKER_URL.rstrip("/") + path
    headers = {}
    if FRONTEND_WORKER_TOKEN:
        headers["x-worker-token"] = FRONTEND_WORKER_TOKEN
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return {"ok": True, "data": r.json()}
    except Exception as e:
        log.warning("worker GET failed %s: %s", url, str(e))
        return {"ok": False, "error": str(e)}

def _worker_post(path: str, params: Dict = None, timeout: float = HTTP_TIMEOUT) -> Dict[str, Any]:
    if not WORKER_URL:
        return {"ok": False, "error": "WORKER_URL not configured"}
    url = WORKER_URL.rstrip("/") + path
    headers = {}
    if FRONTEND_WORKER_TOKEN:
        headers["x-worker-token"] = FRONTEND_WORKER_TOKEN
    try:
        r = requests.post(url, headers=headers, params=params or {}, timeout=timeout)
        # allow 200/202 etc.
        try:
            data = r.json()
        except Exception:
            data = {"text": r.text}
        return {"ok": True, "status_code": r.status_code, "data": data}
    except Exception as e:
        log.warning("worker POST failed %s: %s", url, str(e))
        return {"ok": False, "error": str(e)}

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """
    Render dashboard. Try to fetch status and recent results from worker.
    Fall back to local files if worker is unreachable.
    """
    # try worker status
    status_res = _worker_get("/status")
    if status_res["ok"]:
        status = status_res["data"]
    else:
        status = read_json_local(STATUS_FILE) or {"running": False, "start_time": None}

    # try worker results
    results_res = _worker_get("/results")
    if results_res["ok"]:
        results = results_res["data"].get("results", [])[:6]
    else:
        local = read_json_local(RESULTS_FILE) or []
        results = local[:6]

    # show template
    return templates.TemplateResponse("index.html", {"request": request, "results": results, "status": status, "worker_url": WORKER_URL})

@app.post("/start")
def start_analysis(trigger: Optional[str] = Form("manual")):
    """Startet Analyse über den Worker-Service (server->server POST)."""
    if not WORKER_URL:
        return JSONResponse({"error": "WORKER_URL not configured"}, status_code=500)

    # Call worker /trigger
    res = _worker_post("/trigger", params={"mode": trigger})
    if not res["ok"]:
        return JSONResponse({"status": "error", "detail": res.get("error")}, status_code=502)
    # forward worker response (status_code may be 200 or 409 if already running)
    code = res.get("status_code", 200)
    return JSONResponse(res.get("data", {}), status_code=code)

@app.get("/status")
def api_status():
    """Proxy to worker /status with fallback."""
    res = _worker_get("/status")
    if res["ok"]:
        return res["data"]
    # fallback to local file
    local = read_json_local(STATUS_FILE) or {"running": False, "start_time": None}
    return {"error": "worker_unreachable", "local_status": local, "detail": res.get("error")}

@app.get("/results")
def api_results():
    """Proxy to worker /results with fallback to local."""
    res = _worker_get("/results")
    if res["ok"]:
        return res["data"]
    local = read_json_local(RESULTS_FILE) or []
    return {"error": "worker_unreachable", "results": local, "detail": res.get("error")}

@app.get("/predictions")
def api_predictions():
    """Return latest predictions JSON from worker -> /predictions endpoint."""
    res = _worker_get("/predictions")
    if res["ok"]:
        return res["data"]
    # fallback: try read output CSV locally (best-effort)
    out_csv = os.path.join(os.path.dirname(BASE_DIR), "output", "predictions_next5d_classification.csv")
    if os.path.exists(out_csv):
        try:
            import csv
            rows = []
            with open(out_csv, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for r in reader:
                    rows.append(r)
            return {"predictions": rows, "source": "local_csv"}
        except Exception as e:
            return {"error": "failed_local_csv_parse", "detail": str(e)}
    return {"error": "worker_unreachable", "detail": res.get("error")}

@app.get("/healthz")
def healthz():
    # verify basic startup (always returns ok)
    return {"ok": True, "worker_url": WORKER_URL or None}

PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions.json")

@app.post("/api/upload_results")
async def upload_results(request: Request):
    """
    Wird vom Worker aufgerufen nachdem eine Analyse fertig ist.
    Speichert Ergebnisse, Status und Predictions lokal im Frontend.
    """
    try:
        body = await request.json()

        # 1) Ergebnisse speichern (run history / metadata)
        results = body.get("results", [])
        summary = body.get("summary", {})

        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # 1b) Predictions (Top list) speichern, falls mitgesendet
        preds = body.get("predictions")
        if preds is not None:
            try:
                with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
                    json.dump(preds, f, indent=2, default=str)
            except Exception as e:
                log.warning("failed to save predictions file: %s", e)

        # 2) Status aktualisieren
        status_data = {
            "running": False,
            "start_time": summary.get("start_time"),
            "end_time": summary.get("timestamp"),
            "runtime_seconds": summary.get("runtime_seconds"),
            "trigger": summary.get("trigger", "auto"),
            "accuracy": summary.get("accuracy"),
            "mean_5_day_return": summary.get("mean_5_day_return")
        }

        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2)

        log.info("Frontend hat neue Analyse-Ergebnisse gespeichert.")
        return {"ok": True}

    except Exception as e:
        log.error(f"upload_results failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# run via `uvicorn web.main:app` in production (Render)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("web.main:app", host="0.0.0.0", port=port)
