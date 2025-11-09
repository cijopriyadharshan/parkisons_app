import streamlit as st
import joblib
import librosa
import numpy as np
import io
from deep_translator import GoogleTranslator
from pydub import AudioSegment
import time

# === ANIMATED BACKGROUND (NO UPLOAD NEEDED) ===
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(-45deg, #1e3c72, #2a5298, #0f2027, #203a43);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        color: white;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .title {
        font-size: 3.8rem;
        font-weight: 900;
        text-align: center;
        margin: 2rem 0;
        background: linear-gradient(90deg, #00dbde, #fc00ff);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        animation: fadeIn 2s ease-in-out;
    }
    .subtitle {
        text-align: center;
        font-size: 1.4rem;
        opacity: 0.9;
        margin-bottom: 2.5rem;
        animation: fadeIn 2.5s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stFileUploader > div > div {
        background: rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .stMetric {
        font-size: 2.5rem !important;
        text-align: center;
    }
    .pulse {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.08); }
        100% { transform: scale(1); }
    }
</style>
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

# === AUDIO CONVERTER ===
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

# === AUDIO LOADER ===
def load_audio(file):
    file_bytes = file.read()
    file_name = file.name.lower()

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
        return y[:sr*5], sr
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
    t("Upload voice: AMR, WAV, MP3, M4A, OGG, FLAC, AAC, WMA"),
    type=['amr', 'wav', 'mp3', 'm4a', 'ogg', 'flac', 'aac', 'wma']
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
        st.markdown(f"<h2 class='pulse' style='text-align:center;'>{t('Risk Score')}</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.metric("", f"{prob:.1%}", delta=None)

        if pred:
            st.error(f"**{t('HIGH RISK')}** — {t('Consult a neurologist immediately')}")
        else:
            st.success(f"**{t('LOW RISK')}** — {t('Healthy voice pattern')}")

else:
    st.info(t("Upload your voice recording to begin."))
    st.markdown("<br><br>", unsafe_allow_html=True)
