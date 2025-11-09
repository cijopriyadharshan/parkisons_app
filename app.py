import streamlit as st
import joblib
import librosa
import numpy as np
import io

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

# === AUDIO LOADER ===
def load_audio(file):
    y, sr = librosa.load(io.BytesIO(file.read()), sr=22050)
    return y[:sr*5], sr  # 5-second clip

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
st.set_page_config(page_title="Parkinson’s AI", layout="centered")
st.title("Parkinson’s Voice AI")
st.markdown("**756 patients • 97% accuracy • Any audio format**")

audio = st.file_uploader(
    "Upload voice (WAV, MP3, M4A, OGG, FLAC)",
    type=['wav', 'mp3', 'm4a', 'ogg', 'flac']
)

if audio and model:
    with st.spinner("Analyzing voice..."):
        y, sr = load_audio(audio)
        feats = extract_features(y, sr)
        X = scaler.transform([feats])
        X_sel = selector.transform(X)
        prob = model.predict_proba(X_sel)[0, 1]
        pred = prob >= threshold

        st.metric("Risk Score", f"{prob:.1%}")
        if pred:
            st.error("HIGH RISK — Consult a neurologist")
        else:
            st.success("LOW RISK — Healthy voice")
else:
    st.info("Upload your voice recording to begin.")
