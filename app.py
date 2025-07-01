import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model dan preprocessing
model = joblib.load("model_svm.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
selected_features = joblib.load("selected_features.pkl")
imputer = joblib.load("knn_imputer.pkl")

st.set_page_config(page_title="Klasifikasi Topik Skripsi", layout="centered")

st.title("🎓 Klasifikasi Topik Skripsi Mahasiswa PTIK")
st.write("Masukkan nilai mahasiswa pada 15 fitur terpilih untuk memprediksi topik skripsi.")

if st.button("🔍 Prediksi Topik"):
    input_df = pd.DataFrame([input_data], columns=selected_features)

    # Imputasi nilai kosong
    imputed_input = imputer.transform(input_df)

    # Normalisasi
    scaled_input = scaler.transform(imputed_input)

    # Prediksi label
    pred = model.predict(scaled_input)
    label = label_encoder.inverse_transform(pred)[0]

    st.success(f"Hasil prediksi topik skripsi: **{label}** 🎯")
