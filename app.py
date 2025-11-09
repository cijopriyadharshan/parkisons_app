import streamlit as st
import joblib
import librosa
import numpy as np
import io
from deep_translator import GoogleTranslator

# === UI ===
st.markdown(f"""
<style>
    .stApp {{background: url("bg.jpg") center/cover fixed; color: #000;}}
    .overlay {{background: rgba(255,255,255,0.88); position: fixed; inset: 0; z-index: -1; backdrop-filter: blur(6px);}}
    .title {{font-size: 3.8rem; font-weight: 900; text-align: center; color: #1a1a1a;}}
    .stFileUploader > div > div {{background: rgba(255,255,255,0.95); border: 2px solid #007acc; border-radius: 16px; padding: 1.5rem;}}
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

# === AUDIO LOADER (WAV, MP3, M4A, OGG, FLAC ONLY) ===
def load_audio(file):
    try:
        y, sr = librosa.load(io.BytesIO(file.read()), sr=22050)
        return y[:22050*5], 22050
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Supported: **WAV, MP3, M4A, OGG, FLAC** (No AMR)")
        return None, None

# === FEATURES ===
def extract_features(y, sr):
    import librosa
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    feats = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), [zcr.mean(), rms.mean()]])
    full = np.zeros(754)
    full[:len(feats)] = feats
    return full

# === UI ===
langs = {'English':'en','Spanish':'es','Hindi':'hi','Arabic':'ar'}
lang = st.selectbox("Language", list(langs.keys()))
tr = GoogleTranslator(source='auto', target=langs[lang])
t = lambda x: tr.translate(x)

st.markdown(f"<h1 class='title'>{t('Parkinson Detector')}</h1>", unsafe_allow_html=True)

audio = st.file_uploader("Upload voice (WAV, MP3, M4A, OGG, FLAC)", type=['wav','mp3','m4a','ogg','flac'])

if audio and model:
    with st.spinner("Analyzing..."):
        y, sr = load_audio(audio)
        if y is None: st.stop()
        prob = model.predict_proba(selector.transform(scaler.transform([extract_features(y, sr)])))[0,1]
        st.metric("Risk Score", f"{prob:.1%}")
        st.write("**HIGH RISK** — Consult a doctor" if prob >= threshold else "**LOW RISK** — Healthy pattern")
else:
    st.info("Upload a voice file to begin.")
