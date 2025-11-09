import streamlit as st
import joblib
import numpy as np
import io
import subprocess
import tempfile
import os
from deep_translator import GoogleTranslator
import time
import librosa

# === UI ===
st.markdown("""
<style>
    .stApp {background-image:url("bg.jpg");background-size:cover;background-attachment:fixed;color:#000;}
    .overlay {background:rgba(255,255,255,0.88);position:fixed;top:0;left:0;right:0;bottom:0;z-index:-1;backdrop-filter:blur(6px);}
    .title {font-size:3.8rem;font-weight:900;text-align:center;margin:2rem 0;color:#1a1a1a;}
    .subtitle {text-align:center;font-size:1.4rem;margin-bottom:2.5rem;color:#333;}
    .stFileUploader > div > div {background:rgba(255,255,255,0.95);border-radius:16px;padding:1.5rem;border:2px solid #007acc;}
    .stMetric {background:#fff;border-radius:12px;padding:1.2rem;box-shadow:0 4px 10px rgba(0,0,0,0.1);}
</style>
<div class="overlay"></div>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Parkinson Detector", layout="centered")

# === MODEL ===
@st.cache_resource
def load_model():
    artifacts = joblib.load('parkinsons_final.pkl')
    return artifacts['model'], artifacts['scaler'], artifacts['selector'], artifacts['threshold']
model, scaler, selector, threshold = load_model()

# === TRANSLATOR ===
@st.cache_resource
def get_translator(target='en'):
    return GoogleTranslator(source='auto', target=target)

# === AUDIO LOADER (AMR + ALL FORMATS) ===
def load_audio(file):
    try:
        file_bytes = file.read()
        file_name = file.name.lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp_in:
            tmp_in.write(file_bytes)
            input_path = tmp_in.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            output_path = tmp_out.name

        # Use ffmpeg directly
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "22050", "-ac", "1", "-f", "wav", output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        os.unlink(input_path)
        if result.returncode != 0:
            os.unlink(output_path)
            raise Exception(f"ffmpeg error: {result.stderr}")

        # Load with librosa
        y, sr = librosa.load(output_path, sr=22050)
        os.unlink(output_path)

        return y[:22050*5], 22050

    except Exception as e:
        st.error(f"Audio loading failed: {e}")
        st.info("Supported: AMR, WAV, MP3, M4A, OGG, FLAC")
        return None, None

# === FEATURES ===
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    feats = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), [zcr.mean(), rms.mean()]])
    full = np.zeros(754)
    full[:len(feats)] = feats
    return full

# === UI ===
langs = {'English':'en','Spanish':'es','Hindi':'hi','Arabic':'ar','French':'fr','German':'de','Chinese':'zh','Russian':'ru','Portuguese':'pt','Japanese':'ja','Korean':'ko','Italian':'it'}
lang = st.selectbox("Language", list(langs.keys()))
tr = get_translator(langs[lang])
t = lambda x: tr.translate(x)

st.markdown(f"<h1 class='title'>{t('Parkinson Detector')}</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='subtitle'>{t('AI-Powered Voice Analysis')}</p>", unsafe_allow_html=True)

audio = st.file_uploader(t("Upload voice: AMR, WAV, MP3, M4A, OGG, FLAC"), type=['amr','wav','mp3','m4a','ogg','flac'])

if audio and model:
    with st.spinner(t("Analyzing...")):
        for i in range(100):
            time.sleep(0.01)
            st.progress(i + 1)
        y, sr = load_audio(audio)
        if y is None: st.stop()
        X = scaler.transform([extract_features(y, sr)])
        X = selector.transform(X)
        prob = model.predict_proba(X)[0,1]
        pred = prob >= threshold
        col1,col2,col3 = st.columns([1,2,1])
        with col2:
            st.markdown(f"<h2 style='text-align:center;color:#007acc'>{t('Risk Score')}</h2>", unsafe_allow_html=True)
            st.metric("", f"{prob:.1%}")
        if pred:
            st.error(f"**{t('HIGH RISK')}** — {t('Consult a neurologist immediately')}")
        else:
            st.success(f"**{t('LOW RISK')}** — {t('Healthy voice pattern')}")
else:
    st.info(t("Upload your voice recording to begin."))
