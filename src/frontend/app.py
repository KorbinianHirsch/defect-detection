"""Streamlit demo: upload a product photo, get an anomaly score + heatmap.

Talks to the FastAPI backend's /predict endpoint. Run both:

    uvicorn src.api.main:app --reload
    streamlit run src/frontend/app.py
"""

import base64
import os

import requests
import streamlit as st

API_URL = os.environ.get("DEFECT_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Defect Detection", page_icon="🔍", layout="centered")
st.title("🔍 Unsupervised Defect Detection")
st.caption(
    "PatchCore anomaly detection, trained only on defect-free images. "
    "Upload a product photo to get an anomaly score and a localization heatmap."
)

try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    st.caption(f"Backend: {API_URL} · category = `{health['category']}` · model loaded = {health['model_loaded']}")
except requests.RequestException:
    st.error(f"Cannot reach backend at {API_URL}. Is `uvicorn src.api.main:app` running?")
    st.stop()

uploaded = st.file_uploader("Upload a product image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded, caption="Original", use_container_width=True)

    with st.spinner("Running inference ..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
            st.stop()

    result = response.json()
    heatmap_bytes = base64.b64decode(result["heatmap_png_base64"])

    with col2:
        st.image(heatmap_bytes, caption="Anomaly heatmap", use_container_width=True)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Anomaly score", f"{result['score']:.2f}")
    m2.metric("Threshold", f"{result['threshold']:.2f}" if result["threshold"] is not None else "n/a")

    if result["is_anomaly"]:
        m3.metric("Verdict", "⚠️ Defect")
        st.error("Defect detected.")
    else:
        m3.metric("Verdict", "✅ OK")
        st.success("No defect detected.")
