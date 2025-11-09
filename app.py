
import streamlit as st
import joblib
import librosa
import numpy as np
import time
import io
from googletrans import Translator

# === CONFIG ===
st.set_page_config(page_title="Parkinson’s AI", layout="centered")
translator = Translator()

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    try:
        artifacts = joblib.load('parkinsons_final.pkl')
        return artifacts['model'], artifacts['scaler'], artifacts['selector'], artifacts['threshold']
    except:
        st.error("Model not found. Upload `parkinsons_final.pkl`")
        return None, None, None, None

model, scaler, selector, threshold = load_model()

# === ALL AUDIO FORMATS ===
def load_audio(file):
    audio_bytes = file.read()
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
    return y[:sr*5], sr

# === FEATURES ===
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    feats = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), [zcr.mean(), rms.mean()]])
    full = np.zeros(754); full[:len(feats)] = feats
    return full

# === LANGUAGES ===
LANGUAGES = {'English': 'en', 'Hindi': 'hi', 'Tamil': 'ta', 'Spanish': 'es'}
def translate(text, lang):
    if lang == 'en': return text
    try:
        return translator.translate(text, dest=lang).text
    except:
        return text

# === UI ===
st.markdown("<h1 style='text-align:center; color:#6a11cb;'>Parkinson’s Voice AI</h1>", unsafe_allow_html=True)
lang = st.sidebar.selectbox("Language", list(LANGUAGES.keys()))
lang_code = LANGUAGES[lang]

T = {
    'upload': translate("Upload any audio", lang_code),
    'high': translate("Parkinson’s Risk", lang_code),
    'low': translate("Healthy Voice", lang_code)
}

audio = st.file_uploader(T['upload'], type=['wav', 'mp3', 'm4a', 'ogg', 'flac'])

if audio and model:
    with st.spinner("Analyzing..."):
        y, sr = load_audio(audio)
        feats = extract_features(y, sr)
        X = scaler.transform([feats])
        X_sel = selector.transform(X)
        prob = model.predict_proba(X_sel)[0, 1]
        pred = prob >= threshold

        st.metric("Risk Score", f"{prob:.1%}")
        if pred:
            st.error(f"{T['high']} — See a doctor")
        else:
            st.success(f"{T['low']} — Keep monitoring")
