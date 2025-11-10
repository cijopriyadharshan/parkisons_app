# app.py
import os
import time
import threading
import subprocess
import tempfile
import requests
from flask import Flask, render_template, request, jsonify
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import joblib
import opensmile
from googletrans import Translator

# === LOAD MODEL ===
print("Loading model...")
artifacts = joblib.load('parkinsons_final.pkl')
model = artifacts['model']
scaler = artifacts['scaler']
selector = artifacts['selector']
threshold = artifacts['threshold']
print("Model loaded.")

# === openSMILE ===
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# === Translator ===
translator = Translator()

# === FastAPI ===
fastapi_app = FastAPI()
fastapi_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@fastapi_app.get("/health")
def health():
    return {"status": "ok"}

@fastapi_app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if len(contents) == 0 or len(contents) > 50 * 1024 * 1024:
            return {"error": "Invalid file"}

        suffix = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            tmp_in.write(contents); input_path = tmp_in.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            output_path = tmp_out.name

        cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        os.unlink(input_path)
        if result.returncode != 0:
            os.unlink(output_path)
            return {"error": "Audio conversion failed"}

        y, sr = librosa.load(output_path, sr=16000)
        os.unlink(output_path)
        if len(y) < 16000:
            return {"error": "Say 'Aaaah' for 5 seconds"}

        feats = smile.process_signal(y, sampling_rate=16000)
        feat_vec = np.zeros(754)
        feat_vec[:min(754, len(feats.columns))] = feats.iloc[0].values[:754]

        X = scaler.transform([feat_vec])
        X_sel = selector.transform(X)
        prob = model.predict_proba(X_sel)[0, 1]
        risk = "HIGH" if prob >= threshold else "LOW"

        return {"risk_score": round(prob, 3), "risk": risk}
    except:
        return {"error": "Processing failed"}

# === Flask ===
flask_app = Flask(__name__, template_folder="templates")
API_URL = "http://localhost:8000/predict"

@flask_app.route("/", methods=["GET", "POST"])
def index():
    lang = request.args.get("lang", "en")
    result = None
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            file.stream.seek(0, 2); size = file.stream.tell(); file.stream.seek(0)
            if size <= 50 * 1024 * 1024:
                try:
                    resp = requests.post(API_URL, files={'file': (file.filename, file.stream, file.content_type)}, timeout=180)
                    result = resp.json()
                except:
                    result = {"error": "Server error"}
            else:
                result = {"error": "File too large"}
        else:
            result = {"error": "No file"}
    return render_template("index.html", result=result, lang=lang)

@flask_app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    try:
        return {"translated": translator.translate(data.get("text", ""), dest=data.get("lang", "en")).text}
    except:
        return {"translated": data.get("text", "")}

# === Start FastAPI ===
def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

threading.Thread(target=run_fastapi, daemon=True).start()

# === Wait ===
for _ in range(60):
    try:
        requests.get("http://localhost:8000/health", timeout=2); break
    except:
        time.sleep(1)
else:
    exit(1)

# === Run with gunicorn on Render ===
if __name__ == "__main__":
    if os.getenv("RENDER"):
        subprocess.run(["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:5000", "app:flask_app"])
    else:
        flask_app.run(host="0.0.0.0", port=5000)
