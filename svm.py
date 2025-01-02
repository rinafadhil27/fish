import streamlit as st
import pickle
import numpy as np

# Load the scaler and model
scaler_path = "scaler_svm.sav"
model_path = "fish_svm.sav"

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit app title
st.title("Aplikasi Klasifikasi Ikan dengan SVM")

# Input features
st.header("Masukkan Fitur")

# Asumsi fitur: ['Length', 'Weight', 'W/L Ratio']
length = st.number_input("Length (cm):", min_value=0.0, step=0.1)
weight = st.number_input("Weight (gram):", min_value=0.0, step=0.1)
w_l_ratio = st.number_input("W/L Ratio:", min_value=0.0, step=0.01)

# Button untuk prediksi
if st.button("Prediksi"):
    # Data input pengguna
    input_features = np.array([[length, weight, w_l_ratio]])

    # Preprocessing dengan scaler
    input_scaled = scaler.transform(input_features)

    # Prediksi dengan model
    prediction = model.predict(input_scaled)

    # Tampilkan hasil
    st.subheader("Hasil Prediksi:")
    st.write(f"Jenis Ikan: {prediction[0]}")
