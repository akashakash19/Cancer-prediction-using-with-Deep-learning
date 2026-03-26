import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# =========================
# LOAD MODEL
# =========================
model = load_model("model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Cancer Predictor",
    page_icon="🧠",
    layout="wide"
)

# =========================
# CUSTOM STYLE
# =========================
st.markdown("""
<style>
.big-title {font-size:40px; font-weight:bold; color:#4CAF50;}
.card {
    padding:20px;
    border-radius:10px;
    background-color:#f5f5f5;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🧠 AI Breast Cancer Prediction</p>', unsafe_allow_html=True)
st.markdown("### Enter patient details")

# =========================
# FEATURES
# =========================
features = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]

# =========================
# AUTO FILL BUTTON
# =========================
if st.button("⚡ Auto Fill Sample Data"):
    input_data = [15]*30
else:
    input_data = []

# =========================
# INPUT UI (SLIDERS)
# =========================
cols = st.columns(3)

for i, feature in enumerate(features):
    val = cols[i % 3].slider(feature, 0.0, 1000.0, float(input_data[i] if input_data else 0))
    input_data.append(val)

# =========================
# PREDICTION
# =========================
if st.button("🔍 Predict"):

    input_array = np.array(input_data[:30]).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)
    prob = prediction[0][0]

    st.markdown("## 📊 Prediction Result")

    if prob > 0.5:
        st.success("✅ No Cancer (Benign)")
        st.progress(int(prob * 100))
        st.write(f"Confidence: {prob:.2f}")
    else:
        st.error("❗ Cancer Detected (Malignant)")
        st.progress(int((1 - prob) * 100))
        st.write(f"Confidence: {1 - prob:.2f}")

# =========================
# SIDEBAR INFO
# =========================
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
This app predicts whether a tumor is benign or malignant using a deep learning model.

- Input: 30 medical features
- Output: Cancer prediction
- Built with TensorFlow & Streamlit
""")

st.sidebar.success("Developed by FAHHH 🚀")
