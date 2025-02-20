import streamlit as st
import pandas as pd
import joblib
import os
from src import predict_category

# Load Model dan Data
model_path = 'models/naive_bayes_model.pkl'
if not os.path.exists(model_path):
    st.error("Model tidak ditemukan. Jalankan train.py terlebih dahulu.")
    st.stop()
model = joblib.load(model_path)

# UI Aplikasi
st.title("🚀 LaporGub Complaint Classification Prototype")
st.write("Masukkan complaint Anda, dan sistem akan memprediksi kategori complaint.")

# Input Keluhan
complaint = st.text_area("Masukkan complaint Anda:")

# Tombol Prediksi
if st.button("Prediksi Kategori"):
    if complaint.strip():
        category = predict_category(model, complaint)
        st.success(f"Kategori complaint: {category}")
    else:
        st.warning("Mohon masukkan complaint terlebih dahulu.")

# Footer
st.markdown("---")
