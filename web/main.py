# web/main.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import traceback
import sys
import os
import importlib

# === Pfad anpassen, damit dein Hauptskript gefunden wird ===
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

app = FastAPI(title="Aktienprognose API")

@app.get("/")
def root():
    return {
        "message": "✅ API läuft.",
        "hint": "Rufe /predict auf, um die Analyse zu starten."
    }

@app.get("/predict")
def predict():
    try:
        print("🚀 Starte Analyse-Run über Web-API...")

        # Modul nur bei Bedarf laden → verhindert Render-Startup-Timeout
        analysis_module = importlib.import_module("analysispredict_next5_FIXED")
        run_pipeline = getattr(analysis_module, "main")

        results = run_pipeline()  # Führt dein Analyse-Skript aus

        if isinstance(results, (list, dict)):
            return JSONResponse(content=results)
        return {"status": "ok", "note": "Run ausgeführt, keine direkte Ausgabe."}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            content={"error": str(e), "type": e.__class__.__name__},
            status_code=500
        )
