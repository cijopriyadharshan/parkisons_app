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
        print(f"Translation failed: {e}")
        return text  # fallback

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

@app.route("/", methods=["GET", "POST"])
def index():
    lang = request.args.get('lang', 'en')
    result = None
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == '':
            result = {"error": "No file selected"}
        elif file.content_length > 50 * 1024 * 1024:
            result = {"error": "File too large (>50MB)"}
        else:
            try:
                # Save uploaded file
                suffix = os.path.splitext(file.filename)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
                    file.save(tmp_in.name)
                    input_path = tmp_in.name

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
                    output_path = tmp_out.name

                # FFmpeg command with debug
                cmd = [
                    "ffmpeg", "-y", "-i", input_path,
                    "-ar", "22050", "-ac", "1", "-f", "wav", output_path
                ]
                process = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30
                )
                os.unlink(input_path)

                if process.returncode != 0:
                    error_msg = process.stderr or "Unknown FFmpeg error"
                    print(f"FFmpeg Error: {error_msg}")
                    os.unlink(output_path)
                    result = {"error": f"FFmpeg failed: {error_msg[:100]}"}
                else:
                    y, sr = librosa.load(output_path, sr=22050)
                    y = y[:5 * sr]  # 5 sec max
                    os.unlink(output_path)

                    feats = extract_features(y, sr)
                    X = scaler.transform([feats])
                    X_sel = selector.transform(X)
                    prob = model.predict_proba(X_sel)[0, 1]
                    risk = "HIGH" if prob >= threshold else "LOW"
                    result = {"risk_score": round(prob, 3), "risk": risk}
            except Exception as e:
                result = {"error": f"Processing error: {str(e)}"}
    return render_template("index.html", result=result, lang=lang, languages=LANGUAGES)

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data.get('text', '')
    target = data.get('lang', 'en')
    return jsonify({"translated": translate_text(text, target)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
