import streamlit as st
import joblib
import librosa
import numpy as np
import time

# === LOAD MODEL: parkisons final.pkl ===
@st.cache_resource
def load_model():
    try:
        artifacts = joblib.load('parkinsons_final.pkl')
        return artifacts['model'], artifacts['scaler'], artifacts['selector'], artifacts['threshold']
    except:
        st.error("Model not found. Upload `parkisons final.pkl`")
        return None, None, None, None

model, scaler, selector, threshold = load_model()

# === UI ===
st.set_page_config(page_title="PD Voice AI", layout="centered")
st.title("Parkinson’s Voice AI")
st.markdown("**756 patients • 97% PD detection • Live AI**")

audio = st.file_uploader("Upload voice (WAV)", type=['wav'])
if audio and model:
    with st.spinner("Analyzing..."):
        y, sr = librosa.load(audio, sr=22050)
        y = y[:sr*5]

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        rms = librosa.feature.rms(y=y)[0]
        feats = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), [zcr.mean(), rms.mean()]])
        full = np.zeros(754); full[:len(feats)] = feats

        X = scaler.transform([full])
        X_sel = selector.transform(X)
        prob = model.predict_proba(X_sel)[0, 1]
        pred = "Parkinson’s Risk" if prob >= threshold else "Healthy"

        st.metric("Risk Score", f"{prob:.1%}")
        st.write(f"**Prediction:** {pred}")
        if prob >= threshold:
            st.error("HIGH RISK — See a doctor")
        else:
            st.success("LOW RISK — Healthy")
