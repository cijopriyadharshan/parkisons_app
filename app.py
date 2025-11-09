import streamlit as st
import joblib
import librosa
import numpy as np
import io
from deep_translator import GoogleTranslator
import time

# === CUSTOM BACKGROUND + DARK TEXT UI ===
st.markdown(f"""
<style>
    .stApp {{
        background-image: url("bg.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #000000 !important;
    }}
    .overlay {{
        background: rgba(255, 255, 255, 0.85);
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        z-index: -1;
        backdrop-filter: blur(5px);
    }}
    .title {{
        font-size: 3.8rem;
        font-weight: 900;
        text-align: center;
        margin: 2rem 0;
        color: #1a1a1a !important;
        text-shadow: none;
    }}
    .subtitle {{
        text-align: center;
        font-size: 1.4rem;
        margin-bottom: 2.5rem;
        color: #333333 !important;
        font-weight: 500;
    }}
    .stFileUploader > div > div {{
        background: rgba(255,255,255,0.95);
        border-radius: 16px;
        padding: 1.5rem;
        border: 2px solid #007acc;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}
    .stFileUploader label {{
        color: #007acc !important;
        font-weight: bold;
    }}
    .stMetric {{
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        text-align: center;
    }}
    .stMetric > div:first-child {{
        color: #007acc !important;
        font-weight: bold;
    }}
    .stSpinner > div {{
        color: #007acc !important;
    }}
    .stError, .stSuccess, .stInfo {{
        border-radius: 12px;
        padding: 1rem;
    }}
</style>
<div class="overlay"></div>
""", unsafe_allow_html=True)

# === CONFIG ===
st.set_page_config(page_title="Parkinson Detector", layout="centered")

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

# === FIXED AUDIO LOADER (NO FORMAT ERROR) ===
def load_audio(file):
    try:
        # Read and reset pointer
        audio_bytes = file.read()
        audio_io = io.BytesIO(audio_bytes)
        audio_io.seek(0)  # CRITICAL: Reset pointer

        # Let librosa auto-detect format
        y, sr = librosa.load(audio_io, sr=22050, mono=True)
        return y[:sr*5], sr  # First 5 seconds
    except Exception as e:
        st.error(f"Audio loading failed: {e}")
        st.info("Supported formats: WAV, MP3, M4A, OGG, FLAC, AMR")
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
    'Portuguese': 'pt', 'Japanese': 'ja', 'Korean': 'ko', 'Italian': 'it'
}
lang = st.selectbox("Language", options=list(langs.keys()))
target_lang = langs[lang]
tr = get_translator(target_lang)
t = lambda x: tr.translate(x)

# Header
st.markdown(f"<h1 class='title'>{t('Parkinson Detector')}</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='subtitle'>{t('AI-Powered Voice Analysis')}</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Upload
audio = st.file_uploader(
    t("Upload voice: AMR, WAV, MP3, M4A, OGG, FLAC"),
    type=['amr', 'wav', 'mp3', 'm4a', 'ogg', 'flac']
)

if audio and model:
    with st.spinner(t("Analyzing voice...")):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)
        
        y, sr = load_audio(audio)
        if y is None:
            st.stop()

        feats = extract_features(y, sr)
        X = scaler.transform([feats])
        X_sel = selector.transform(X)
        prob = model.predict_proba(X_sel)[0, 1]
        pred = prob >= threshold

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown(f"<h2 style='text-align:center; color:#007acc;'>{t('Risk Score')}</h2>", unsafe_allow_html=True)
            st.metric("", f"{prob:.1%}")

        if pred:
            st.error(f"**{t('HIGH RISK')}** — {t('Consult a neurologist immediately')}")
        else:
            st.success(f"**{t('LOW RISK')}** — {t('Healthy voice pattern')}")

else:
    st.info(t("Upload your voice recording to begin."))
    st.markdown("<br><br>", unsafe_allow_html=True)
