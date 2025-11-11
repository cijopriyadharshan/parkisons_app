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
try:
    artifacts = joblib.load('parkinsons_final.pkl')
    model = artifacts['model']
    scaler = artifacts['scaler']
    selector = artifacts['selector']
    threshold = artifacts['threshold']
    print("Model loaded successfully")
except Exception as e:
    print(f"MODEL LOAD FAILED: {e}")
    model = scaler = selector = threshold = None

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
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std = mfcc.std(axis=1)
        zcr = librosa.feature.zero_crossing_rate(y)[0].mean()
        rms = librosa.feature.rms(y=y)[0].mean()
        feats = np.concatenate([mfcc_mean, mfcc_std, [zcr, rms]])
        full = np.zeros(752)
        full[:len(feats)] = feats
        return full
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return np.zeros(752)

# === FLASK APP ===
app = Flask(__name__, template_folder='templates')
LANGUAGES = {'en': 'English', 'hi': 'हिंदी', 'ta': 'தமிழ்', 'bn': 'বাংলা'}

def process_upload(file):
    if not file or file.filename == '':
        return {"error": "No file selected"}

    if file.content_length and file.content_length > 50 * 1024 * 1024:
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
            error_msg = process.stderr.strip() or "Unknown FFmpeg error"
            print(f"FFmpeg failed: {error_msg}")
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)
            return {"error": "Unsupported audio format. Try .wav, .mp3, .amr"}

        y, sr = librosa.load(output_path, sr=22050)
        y = y[:5 * sr]
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)

        feats = extract_features(y, sr)
        X = scaler.transform([feats])
        X_sel = selector.transform(X)
        prob = model.predict_proba(X_sel)[0, 1]
        risk = "HIGH" if prob >= threshold else "LOW"
        print(f"Prediction: {prob:.3f} → {risk}")
        return {"risk_score": round(prob, 3), "risk": risk}

    except subprocess.TimeoutExpired:
        print("FFmpeg timed out")
        return {"error": "Audio processing timed out"}
    except Exception as e:
        print(f"CRITICAL ERROR in process_upload: {e}")
        import traceback
        traceback.print_exc()
        return {"error": "Processing failed"}
    finally:
        for path in [input_path, output_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass

@app.route("/", methods=["GET", "POST"])
def index():
    lang = request.args.get('lang', 'en')
    result = None
    status = 200

    if request.method == "POST":
        result = process_upload(request.files.get("file"))
        if result is None:
            result = {"error": "Internal server error"}
        status = 400 if result.get("error") else 200
        return jsonify(result), status  # ALWAYS JSON, NEVER None

    return render_template("index.html", result=result, lang=lang, languages=LANGUAGES), status

@app.route("/translate", methods=["POST"])
def translate():
    try:
        data = request.get_json()
        text = data.get('text', '')
        target = data.get('lang', 'en')
        return jsonify({"translated": translate_text(text, target)})
    except Exception as e:
        print(f"Translate error: {e}")
        return jsonify({"translated": text}), 400
