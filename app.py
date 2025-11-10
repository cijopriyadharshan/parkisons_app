# app.py
import os
import subprocess
import tempfile
import librosa
import numpy as np
import joblib
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
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
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Setup templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def index_post(request: Request, file: UploadFile = File(...)):
    result = None
    try:
        # Check file size (50MB limit)
        contents = await file.read()
        if len(contents) > 50 * 1024 * 1024:
            result = {"error": "File too large. Max 50MB."}
        else:
            # Process the file
            suffix = os.path.splitext(file.filename)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
                tmp_in.write(contents)
                input_path = tmp_in.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
                output_path = tmp_out.name

            # Convert audio using ffmpeg
            cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "22050", "-ac", "1", output_path]
            result_cmd = subprocess.run(cmd, capture_output=True, text=True)
            os.unlink(input_path)
            
            if result_cmd.returncode != 0:
                os.unlink(output_path)
                result = {"error": "FFmpeg failed: " + result_cmd.stderr}
            else:
                # Extract features and predict
                y, sr = librosa.load(output_path, sr=22050)
                os.unlink(output_path)
                feats = extract_features(y, sr)
                X = scaler.transform([feats])
                X_sel = selector.transform(X)
                prob = model.predict_proba(X_sel)[0, 1]
                risk = "HIGH" if prob >= threshold else "LOW"
                result = {"risk_score": round(prob, 3), "risk": risk}
    except Exception as e:
        result = {"error": str(e)}
    
    return templates.TemplateResponse("index.html", {"request": request, "result": result})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        suffix = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            tmp_in.write(contents)
            input_path = tmp_in.name
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
