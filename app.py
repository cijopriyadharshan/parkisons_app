import streamlit as st
import joblib
import librosa
import numpy as np
import io
from deep_translator import GoogleTranslator
import time

# === STATIC BACKGROUND (MANUAL UPLOAD) ===
st.markdown(f"""
<style>
    .stApp {{
        background-image: url("bg.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }}
    .overlay {{
        background: rgba(0, 0, 0, 0.5);
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        z-index: -1;
    }}
    .title {{
        font-size: 3.8rem;
        font-weight: 900;
        text-align: center;
        margin: 2rem 0;
        color: #fff;
        text-shadow: 0 0 15px rgba(255,255,255,0.5);
    }}
    .subtitle {{
        text-align: center;
        font-size: 1.4rem;
        margin-bottom: 2.5rem;
        color: #eee;
    }}
    .stFileUploader > div > div {{
        background: rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.3);
    }}
    .stMetric {{
        font-size: 2.5rem !important;
        text-align: center;
        background: rgba(0,0,0,0.3);
        border-radius: 15px;
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

# === AUDIO LOADER (NO FFMPEG) ===
def load_audio(file):
    try:
        audio_bytes = io.BytesIO(file.read())
        y, sr = librosa.load(audio_bytes, sr=22050)
        return y[:sr*5], sr
    except Exception as e:
        st.error(f"Audio loading failed: {e}")
        st.info("Supported: WAV, MP3, M4A, OGG, FLAC, AMR")
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
            st.markdown(f"<h2 style='text-align:center; color:#fff;'>{t('Risk Score')}</h2>", unsafe_allow_html=True)
            st.metric("", f"{prob:.1%}", delta=None)

        if pred:
            st.error(f"**{t('HIGH RISK')}** — {t('Consult a neurologist immediately')}")
        else:
            st.success(f"**{t('LOW RISK')}** — {t('Healthy voice pattern')}")

else:
    st.info(t("Upload your voice recording to begin."))
    st.markdown("<br><br>", unsafe_allow_html=True)
