# --- Basis-Image ---
FROM python:3.11-slim

# --- Verzeichnisstruktur einrichten ---
WORKDIR /app

# --- Systemabhängigkeiten installieren (z. B. für numpy, pandas, torch usw.) ---
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- Requirements kopieren und installieren ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Projektdateien kopieren ---
COPY . .

# --- Port-Variable für Render/Railway ---
ENV PORT=8000

# --- Startbefehl ---
CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000"]
