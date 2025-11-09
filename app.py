import streamlit as st
import joblib
import numpy as np
import io
import soundfile as sf
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
        background: rgba(255, 255, 255, 0.88);
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        z-index: -1;
        backdrop-filter: blur(6px);
    }}
    .title {{
        font-size: 3.8rem;
        font-weight: 900;
        text-align: center;
        margin: 2rem 0;
        color: #1a1a1a !important;
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
        padding: 1.2rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        text-align: center;
    }}
    .stMetric > div:first-child {{
        color: #007acc !important;
        font-weight: bold;
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

# === AUDIO LOADER (FIXED — ALL FORMATS) ===
def load_audio(file):
    try:
        # Read file
        file_bytes = file.read()
        file_io = io.BytesIO(file_bytes)
        file_io.seek(0)

        # Detect format from extension
        file_name = file.name.lower()
        if file_name.endswith('.wav'):
            fmt = 'WAV'
        elif file_name.endswith('.mp3'):
            fmt = 'MP3'
        elif file_name.endswith('.m4a'):
            fmt = 'M4A'
        elif file_name.endswith('.ogg'):
            fmt = 'OGG'
        elif file_name.endswith('.flac'):
            fmt = 'FLAC'
        elif file_name.endswith('.amr'):
            fmt = 'AMR'
        else:
            fmt = None

        # Use soundfile (supports all via ffmpeg backend)
        y, sr = sf.read(file_io, format=fmt)
        y = y.astype(np.float32)
        if len(y.shape) > 1:
            y = y.mean(axis=1)  # Convert to mono
        if sr != 22050:
            # Resample manually if needed (soundfile doesn't resample)
            from scipy.signal import resample
            num_samples = int(len(y) * 22050 / sr)
            y = resample(y, num_samples)
            sr = 22050
        return y[:22050*5], 22050  # 5 seconds
    except Exception as e:
        st.error(f"Audio loading failed: {e}")
        st.info("Supported: WAV, MP3, M4A, OGG, FLAC, AMR")
        return None, None

# === FEATURE EXTRACTION ===
def extract_features(y, sr):
    import librosa
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
