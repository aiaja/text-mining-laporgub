import streamlit as st
import pandas as pd
import joblib
import os
import logging
from src import predict_category

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
MODEL_PATH = "models/naive_bayes_model.pkl"

@st.cache_resource
def load_model():
    """Load trained Naive Bayes model from file."""
    if not os.path.exists(MODEL_PATH):
        st.error("⚠️ Model tidak ditemukan. Jalankan train.py terlebih dahulu.")
        st.stop()
    
    try:
        model = joblib.load(MODEL_PATH)
        logging.info("✅ Model berhasil dimuat.")
        return model
    except Exception as e:
        logging.error(f"❌ Error saat memuat model: {e}")
        st.error("⚠️ Terjadi kesalahan saat memuat model.")
        st.stop()

def validate_prediction(predicted_category: str, expected_category: str) -> bool:
    """Check if the prediction is correct, otherwise show an alert."""
    if predicted_category != expected_category:
        st.warning("⚠️ Prediksi tidak sesuai dengan kategori yang diharapkan. Mohon periksa kembali!")
        return False
    return True

# Load Model
model = load_model()

# UI Aplikasi
st.title("🚀 LaporGub Complaint Classification Prototype")
st.write("Masukkan complaint Anda, dan sistem akan memprediksi kategori complaint.")

# Input Keluhan
complaint = st.text_area("📝 Masukkan complaint Anda:")

# Input kategori yang diharapkan (opsional)
expected_category = st.text_input("📌 Kategori yang diharapkan (opsional):")

# Tombol Prediksi
if st.button("🔍 Prediksi Kategori"):
    if complaint.strip():
        try:
            predicted_category = predict_category(model, complaint)
            st.success(f"✅ Kategori complaint: **{predicted_category}**")
            
            # Check prediction validity if expected category is provided
            if expected_category:
                if not validate_prediction(predicted_category, expected_category):
                    st.error("⚠️ Silakan tinjau kembali hasil prediksi!")

        except Exception as e:
            logging.error(f"❌ Error saat prediksi: {e}")
            st.error("⚠️ Terjadi kesalahan saat melakukan prediksi. Silakan coba lagi.")
    else:
        st.warning("⚠️ Mohon masukkan complaint terlebih dahulu.")

# Footer
st.markdown("---")
st.caption("© 2025 LaporGub AI | Prototype")
