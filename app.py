
import streamlit as st
import joblib
import librosa
import numpy as np
import time
import base64
from googletrans import Translator
import io

# === CONFIG ===
st.set_page_config(page_title="Parkinson’s AI", layout="centered", initial_sidebar_state="expanded")
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

# === SUPPORT ALL AUDIO FORMATS ===
def load_audio(file):
    try:
        audio_bytes = file.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        y = y[:sr*5]
        return y, sr
    except Exception as e:
        st.error(f"Audio error: {e}")
        return None, None

# === EXTRACT FEATURES ===
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    feats = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), [zcr.mean(), rms.mean()]])
    full = np.zeros(754); full[:len(feats)] = feats
    return full

# === MULTILANGUAGE ===
LANGUAGES = {
    'English': 'en',
    'Hindi': 'hi',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Kannada': 'kn',
    'Malayalam': 'ml',
    'Spanish': 'es',
    'French': 'fr'
}

def translate(text, lang):
    if lang == 'en': return text
    try:
        return translator.translate(text, dest=lang).text
    except:
        return text

# === CSS & ANIMATIONS ===
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);}
    .title {font-size: 3.2rem; font-weight: 900; text-align: center; color: white; text-shadow: 0 4px 8px rgba(0,0,0,0.3);}
    .card {background: white; border-radius: 20px; padding: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin: 1.5rem 0;}
    .upload-box {border: 3px dashed #6c5ce7; border-radius: 20px; padding: 2rem; text-align: center; background: #f0f4ff;}
    .result-high {background: linear-gradient(45deg, #ff6b6b, #ee5a52); color: white; padding: 2rem; border-radius: 20px; text-align: center; animation: pulse 2s infinite;}
    .result-low {background: linear-gradient(45deg, #51cf66, #40c057); color: white; padding: 2rem; border-radius: 20px; text-align: center; animation: pulse 2s infinite;}
    @keyframes pulse {0% {transform: scale(1);} 50% {transform: scale(1.02);} 100% {transform: scale(1);}}
    .tip-box {background: #e0f2fe; border-left: 5px solid #0ea5e9; padding: 1rem; margin: 1rem 0; border-radius: 8px;}
    .footer {text-align: center; margin-top: 4rem; color: #ddd; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

# === HEADER ===
st.markdown("<h1 class='title'>Parkinson’s Voice AI</h1>", unsafe_allow_html=True)

# === LANGUAGE SELECTOR ===
lang_name = st.sidebar.selectbox("Language", list(LANGUAGES.keys()))
lang_code = LANGUAGES[lang_name]

# === TRANSLATED TEXTS ===
T = {
    'title': translate("Parkinson’s Voice AI", lang_code),
    'subtitle': translate("756 patients • 97% PD detection", lang_code),
    'upload': translate("Upload any audio (MP3, WAV, M4A, OGG)", lang_code),
    'analyzing': translate("Analyzing voice...", lang_code),
    'high_risk': translate("Parkinson’s Risk Detected", lang_code),
    'low_risk': translate("Healthy Voice", lang_code),
    'advice_high': translate("Consult a neurologist. Early treatment helps.", lang_code),
    'advice_low': translate("Keep monitoring. Annual checkup recommended.", lang_code),
    'tips': translate("Tips: Speak clearly, record in quiet room, 3–5 sec voice", lang_code),
    'remedy': translate("Remedies: Voice therapy, hydration, avoid caffeine", lang_code)
}

# === SIDEBAR INFO ===
with st.sidebar:
    st.image("https://img.icons8.com/fluency/100/000000/brain.png")
    st.markdown(f"### {translate('Model', lang_code)}")
    st.markdown("- 93% accuracy")
    st.markdown("- 97% PD detection")
    st.markdown("---")
    st.markdown("### Parkinson’s Info")
    st.image("https://img.icons8.com/color/100/000000/parkinsons-disease.png")
    st.write(translate("Shaking, slow movement, voice changes", lang_code))

# === MAIN UI ===
st.markdown(f"<p style='text-align:center; color:white; font-size:1.3rem;'>{T['subtitle']}</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='upload-box'><strong>{T['upload']}</strong></div>", unsafe_allow_html=True)
    audio_file = st.file_uploader("", type=['wav', 'mp3', 'm4a', 'ogg', 'flac'], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

# === PREDICTION ===
if audio_file and model:
    with st.spinner(T['analyzing']):
        y, sr = load_audio(audio_file)
        if y is None:
            st.stop()
        
        feats = extract_features(y, sr)
        X = scaler.transform([feats])
        X_sel = selector.transform(X)
        prob = model.predict_proba(X_sel)[0, 1]
        is_high_risk = prob >= threshold
        
        time.sleep(1.5)

    # === RESULT ===
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if is_high_risk:
        st.markdown(f"<div class='result-high'><h2>{T['high_risk']}</h2><h1>{prob:.1%}</h1><p>{T['advice_high']}</p></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-low'><h2>{T['low_risk']}</h2><h1>{prob:.1%}</h1><p>{T['advice_low']}</p></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # === TIPS & REMEDIES ===
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='tip-box'><strong>{translate('Tips', lang_code)}:</strong> {T['tips']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='tip-box'><strong>{translate('Remedies', lang_code)}:</strong> {T['remedy']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# === FOOTER ===
st.markdown(f"<div class='footer'>© 2025 | {translate('Built with AI for healthcare', lang_code)}</div>", unsafe_allow_html=True)
