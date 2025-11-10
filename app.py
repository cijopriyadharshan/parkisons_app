# app.py
import os
import threading
import subprocess
import tempfile
import librosa
import numpy as np
import joblib
import requests
from flask import Flask, render_template, request
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# LOAD MODEL
artifacts = joblib.load('parkinsons_final.pkl')
model = artifacts['model']
scaler = artifacts['scaler']
selector = artifacts['selector']
threshold = artifacts['threshold']

def extract_features(y, sr):
    if len(y) == 0: return np.zeros(754)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=8)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    feats = np.concatenate([mfcc.mean(axis=1), [zcr.mean(), rms.mean()]])
    full = np.zeros(754)
    full[:len(feats)] = feats
    return full

# FASTAPI
fastapi_app = FastAPI()
fastapi_app.add_middleware(CORSMiddleware, allow_origins=["*"])

@fastapi_app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        suffix = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            tmp_in.write(contents); input_path = tmp_in.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            output_path = tmp_out.name

        cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "22050", "-ac", "1", output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        os.unlink(input_path)
        if result.returncode != 0:
            os.unlink(output_path)
            return {"error": "FFmpeg failed"}

        y, sr = librosa.load(output_path, sr=22050)
        os.unlink(output_path)
        feats = extract_features(y, sr)
        X = scaler.transform([feats])
        X_sel = selector.transform(X)
        prob = model.predict_proba(X_sel)[0, 1]
        risk = "HIGH" if prob >= threshold else "LOW"
        return {"risk_score": round(prob, 3), "risk": risk}
    except Exception as e:
        return {"error": str(e)}

# FLASK
flask_app = Flask(__name__)
API_URL = "http://localhost:8000/predict"

@flask_app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["file"]
        if file.content_length > 50 * 1024 * 1024:
            result = {"error": "File too large. Max 50MB."}
        else:
            files = {'file': (file.filename, file.stream, file.content_type)}
            try:
                response = requests.post(API_URL, files=files, timeout=180)
                result = response.json()
            except:
                result = {"error": "Processing failed."}
    return render_template("index.html", result=result)

if __name__ == "__main__":
    threading.Thread(target=lambda: uvicorn.run(fastapi_app, host="0.0.0.0", port=8000), daemon=True).start()
    flask_app.run(host="0.0.0.0", port=5000)
