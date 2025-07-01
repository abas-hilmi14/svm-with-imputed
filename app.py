import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model dan preprocessing
model = joblib.load("model_svm.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
selected_features = joblib.load("selected_features.pkl")
imputer = joblib.load("knn_imputer.pkl")  # ✅ Load imputer

st.set_page_config(page_title="Klasifikasi Topik Skripsi", layout="centered")

st.title("🎓 Klasifikasi Topik Skripsi Mahasiswa PTIK")
st.write("Masukkan nilai mahasiswa pada 15 fitur terpilih. Nilai kosong akan diisi otomatis oleh sistem.")

# Form input nilai-nilai (dapat dikosongkan)
input_data = {}
for feature in selected_features:
    val = st.number_input(
        f"{feature}",
        min_value=0.0,
        max_value=100.0,
        value=None,
        format="%.2f",
        step=0.1,
        placeholder="(opsional)"
    )
    input_data[feature] = np.nan if val is None else val

if st.button("🔍 Prediksi Topik"):
    # Buat DataFrame dari input user
    input_df = pd.DataFrame([input_data], columns=selected_features)

    # Imputasi nilai yang kosong
    imputed_array = imputer.transform(input_df)

    # Normalisasi
    scaled_input = scaler.transform(imputed_array)

    # Prediksi
    pred = model.predict(scaled_input)
    label = label_encoder.inverse_transform(pred)[0]

    st.success(f"Hasil prediksi topik skripsi: **{label}** 🎯")
