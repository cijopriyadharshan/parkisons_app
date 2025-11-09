import streamlit as st
import joblib
import librosa
import numpy as np
import io
from deep_translator import GoogleTranslator
from pydub import AudioSegment
import time

# === CONFIG ===
st.set_page_config(page_title="Parkinson Detector", layout="centered")
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #1e3c72, #2a5298); color: white; }
    .stApp { background: transparent; }
    .title { font-size: 3.5rem; font-weight: 900; text-align: center; margin: 1rem 0; 
             text-shadow: 0 0 10px rgba(255,255,255,0.3); }
    .subtitle { text-align: center; font-size: 1.3rem; opacity: 0.9; margin-bottom: 2rem; }
    .stFileUploader > div > div { background: rgba(255,255,255,0.1); border-radius: 15px; padding: 1rem; }
    .stMetric { font-size: 2rem !important; }
    .pulse { animation: pulse 2s infinite; }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .fade-in { animation: fadeIn 1.5s; }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    try:
        artifacts = joblib.load('parkinsons_final.pkl')
        return artifacts['model'], artifacts['scaler'], artifacts['selector'], artifacts['threshold']
    except Exception as e:
        st.error(f"Model not found: {e}")
        st.info("Upload `parkinsons_final.pkl`")
        return None, None, None, None

model, scaler, selector, threshold = load_model()

# === TRANSLATOR ===
@st.cache_resource
def get_translator(target='en'):
    return GoogleTranslator(source='auto', target=target)

# === AUDIO CONVERTER (AMR, M4A, OGG, etc.) ===
def convert_audio(file_bytes, original_format):
    try:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes), format=original_format)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io
    except Exception as e:
        st.error(f"Audio conversion failed: {e}")
        return None

# === AUDIO LOADER (ALL FORMATS) ===
def load_audio(file):
    file_bytes = file.read()
    file_name = file.name.lower()

    # Detect format
    if file_name.endswith('.amr'):
        wav_io = convert_audio(file_bytes, 'amr')
    elif file_name.endswith('.m4a'):
        wav_io = convert_audio(file_bytes, 'm4a')
    elif file_name.endswith('.ogg'):
        wav_io = convert_audio(file_bytes, 'ogg')
    elif file_name.endswith('.aac'):
        wav_io = convert_audio(file_bytes, 'aac')
    elif file_name.endswith('.wma'):
        wav_io = convert_audio(file_bytes, 'wma')
    elif file_name.endswith('.mp3'):
        wav_io = convert_audio(file_bytes, 'mp3')
    elif file_name.endswith('.wav'):
        wav_io = io.BytesIO(file_bytes)
    elif file_name.endswith('.flac'):
        wav_io = convert_audio(file_bytes, 'flac')
    else:
        st.error("Unsupported format. Try WAV, MP3, AMR, M4A, OGG, FLAC")
        return None, None

    if wav_io:
        y, sr = librosa.load(wav_io, sr=22050)
        return y[:sr*5], sr  # 5-second clip
    return None, None

# === FEATURE EXTRACTION ===
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    feats = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        [zcr.mean(), rms.mean()]
    ])
    full = np.zeros(754)
    full[:len(feats)] = feats
    return full

# === UI ===
langs = {
    'English': 'en', 'Spanish': 'es', 'Hindi': 'hi', 'Arabic': 'ar',
    'French': 'fr', 'German': 'de', 'Chinese': 'zh', 'Russian': 'ru',
    'Portuguese': 'pt', 'Japanese': 'ja', 'Korean': 'ko', 'Italian': 'it
