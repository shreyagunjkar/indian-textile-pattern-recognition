import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Textile AI Dashboard", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.main {
    background-color: #0f172a;
    color: white;
}
.sidebar .sidebar-content {
    background-color: #111827;
}
h1, h2, h3 {
    color: white;
}
.stButton>button {
    background: linear-gradient(90deg, #4f46e5, #3b82f6);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    box-shadow: 0 0 15px rgba(79, 70, 229, 0.7);
}
.card {
    background-color: #1f2937;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.5);
}
img {
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_my_model():
    return load_model("textile_model.h5")

model = load_my_model()

class_names = ['bandhani', 'kalamkari', 'paithani']
IMG_SIZE = 224

with st.sidebar:
    st.title("⚙️ Configuration Panel")
    st.write("Upload and analyze textile patterns")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

st.markdown("## 🧵 Textile Pattern AI Dashboard")
st.markdown("Analyze textile patterns using deep learning")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📷 Input Image")

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")

        st.image(img, width=350)  # FIXED SIZE, NO STRETCH

    else:
        st.info("Upload image from sidebar")

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤖 AI Prediction")

    if uploaded_file:
        if st.button("🚀 Run Analysis"):

            img_resized = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img_resized)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            class_id = np.argmax(prediction)
            confidence = float(np.max(prediction))

            st.markdown(f"### 🎯 Prediction: **{class_names[class_id]}**")
            st.markdown(f"### 📊 Confidence: {confidence*100:.2f}%")

            st.progress(int(confidence * 100))

            st.markdown("### 🔍 Class Breakdown")
            for i, cls in enumerate(class_names):
                st.write(cls)
                st.progress(int(prediction[0][i] * 100))

    else:
        st.info("Waiting for input...")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<h3 style='text-align: center;'>🚀 Computer Vision Mini Project</h3>",
    unsafe_allow_html=True
)