# web/main.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import traceback
import sys
import os

# === Pfad anpassen, damit dein Hauptskript gefunden wird ===
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from analysispredict_next5_FIXED import main as run_pipeline  # ⬅️ nutzt dein bestehendes Skript

app = FastAPI(title="Aktienprognose API")

@app.get("/")
def root():
    return {"message": "✅ API läuft. Besuche /predict, um eine Analyse zu starten."}

@app.get("/predict")
def predict():
    try:
        print("🚀 Starte Analyse-Run über Web-API...")
        results = run_pipeline()  # Führt dein Analyse-Skript aus
        if isinstance(results, (list, dict)):
            return JSONResponse(content=results)
        return {"status": "ok", "note": "Run ausgeführt, keine direkte Ausgabe."}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)
