import streamlit as st
import numpy as np
import pickle

# =========================
# LOAD MODEL & SCALER
# =========================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# =========================
# UI DESIGN
# =========================
st.set_page_config(page_title="Cancer Prediction", page_icon="🧠")

st.title("🧠 Breast Cancer Prediction System")
st.markdown("### Enter patient details below:")

# =========================
# INPUT FIELDS (BETTER UI)
# =========================
col1, col2 = st.columns(2)

input_data = []

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

# split inputs into 2 columns
for i in range(15):
    val = col1.number_input(features[i], value=0.0)
    input_data.append(val)

for i in range(15, 30):
    val = col2.number_input(features[i], value=0.0)
    input_data.append(val)

# =========================
# PREDICTION BUTTON
# =========================
if st.button("🔍 Predict"):

    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)

    probability = prediction[0][0]

    st.subheader("📊 Result:")

    if probability > 0.5:
        st.success(f"✅ No Cancer (Benign)\n\nConfidence: {probability:.2f}")
    else:
        st.error(f"❗ Cancer Detected (Malignant)\n\nConfidence: {1 - probability:.2f}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("👨‍💻 Developed by FAHHH | AI/ML Project")
