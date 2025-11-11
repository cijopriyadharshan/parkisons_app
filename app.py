# app.py
import os
import subprocess
import tempfile
import librosa
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from deep_translator import GoogleTranslator

# === LOAD MODEL ===
artifacts = joblib.load('parkinsons_final.pkl')
model = artifacts['model']
scaler = artifacts['scaler']
selector = artifacts['selector']
threshold = artifacts['threshold']

# === TRANSLATOR ===
def translate_text(text, target_lang):
    if target_lang == 'en': return text
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text

# === EXTRACT FEATURES ===
def extract_features(y, sr):
    if len(y) == 0: return np.zeros(752)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)[0].mean()
    rms = librosa.feature.rms(y=y)[0].mean()
    feats = np.concatenate([mfcc_mean, mfcc_std, [zcr, rms]])
    full = np.zeros(752)
    full[:len(feats)] = feats
    return full

# === FLASK APP ===
app = Flask(__name__, template_folder='templates')
LANGUAGES = {'en': 'English', 'hi': 'हिंदी', 'ta': 'தமிழ்', 'bn': 'বাংলা'}

def process_upload(file):
    if not file or file.filename == '':
        return {"error": "No file selected"}

    content_length = getattr(file, "content_length", None)
    if content_length and content_length > 50 * 1024 * 1024:
        return {"error": "File too large (>50MB)"}

    input_path = output_path = None
    try:
        suffix = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            file.save(tmp_in.name)
            input_path = tmp_in.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            output_path = tmp_out.name

        cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "22050", "-ac", "1", output_path]
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)

        if process.returncode != 0:
            error = process.stderr.strip() or "Unknown error"
            print(f"FFmpeg failed: {error}")
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)
            return {"error": "Audio format not supported. Try .wav, .mp3, .amr"}

        y, sr = librosa.load(output_path, sr=22050)
        y = y[:5 * sr]
        feats = extract_features(y, sr)
        X = scaler.transform([feats])
        X_sel = selector.transform(X)
        prob = model.predict_proba(X_sel)[0, 1]
        risk = "HIGH" if prob >= threshold else "LOW"
        return {"risk_score": round(prob, 3), "risk": risk}
    except subprocess.TimeoutExpired:
        print("FFmpeg timed out")
        return {"error": "Audio processing timed out"}
    except Exception as e:
        print(f"Processing error: {e}")
        return {"error": "Processing failed"}
    finally:
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)

@app.route("/", methods=["GET", "POST"])
def index():
    lang = request.args.get('lang', 'en')
    result = None
    status = 200

    if request.method == "POST":
        result = process_upload(request.files.get("file"))
        status = 400 if result and result.get("error") else 200
        wants_json = request.headers.get("X-Requested-With") == "XMLHttpRequest" or \
            "application/json" in request.headers.get("Accept", "")
        if wants_json:
            return jsonify(result or {}), status

    return render_template("index.html", result=result, lang=lang, languages=LANGUAGES), status

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    return jsonify({"translated": translate_text(data.get('text', ''), data.get('lang', 'en'))})
