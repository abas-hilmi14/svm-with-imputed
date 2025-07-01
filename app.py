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

# Kolom yang digunakan oleh imputer
imputer_columns = imputer.feature_names_in_  # ← penting!

st.set_page_config(page_title="Klasifikasi Topik Skripsi", layout="centered")

st.title("🎓 Klasifikasi Topik Skripsi Mahasiswa PTIK")
st.write("Masukkan nilai mahasiswa pada 15 fitur terpilih. Nilai kosong akan diisi otomatis.")

# Buat input hanya untuk fitur yang dikenali oleh imputer
input_data = {}
for feature in imputer_columns:
    val = st.number_input(
        f"{feature}", min_value=0.0, max_value=100.0,
        value=None, format="%.2f", step=0.1, placeholder="(opsional)"
    )
    input_data[feature] = np.nan if val is None else val

if st.button("🔍 Prediksi Topik"):
    input_df = pd.DataFrame([input_data], columns=imputer_columns)

    # Imputasi
    imputed_array = imputer.transform(input_df)

    # Ambil fitur terpilih dari hasil imputasi
    df_imputed = pd.DataFrame(imputed_array, columns=imputer_columns)
    X_selected_input = df_imputed[selected_features]

    # Normalisasi
    scaled_input = scaler.transform(X_selected_input)

    # Prediksi
    pred = model.predict(scaled_input)
    label = label_encoder.inverse_transform(pred)[0]

    st.success(f"Hasil prediksi topik skripsi: **{label}** 🎯")
