import streamlit as st
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

# 1. Memuat model Pipeline yang sudah disimpan
# Model ini mencakup preprocessor (scaling & encoding) dan Ridge Regression
model = joblib.load('XGB_model_final.pkl')

# 2. Judul dan Deskripsi Aplikasi
st.title("Laptop Price Predictor")
st.write("""
Aplikasi ini memprediksi **Harga Laptop (Euro)** berdasarkan spesifikasi teknis 
menggunakan model Ridge Regression yang telah dioptimalkan.
""")

st.header("Masukkan Spesifikasi Laptop")

# 3. Membuat Input Form untuk 10 Fitur Terpilih
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        # Fitur Kategorikal (Berdasarkan analisis EDA) 
        company = st.selectbox("Merek (Company)", 
                               ['Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'MSI', 'Razer', 'Other'])
        typename = st.selectbox("Tipe Laptop", 
                                ['Ultrabook', 'Notebook', 'Netbook', 'Gaming', '2 in 1 Convertible', 'Workstation'])
        cpu_family = st.selectbox("Keluarga CPU", 
                                  ['Core i7', 'Core i5', 'Core i3', 'Celeron Dual Core', 'Pentium Quad Core', 'Ryzen', 'Other'])
        opsys = st.selectbox("Sistem Operasi", 
                             ['Windows 10', 'macOS', 'Linux', 'No OS', 'Windows 7', 'Other'])
        gpu = st.selectbox("GPU Brand", ['Intel HD Graphics 620', 'Nvidia GeForce GTX 1050', 'AMD Radeon 530', 'Other'])

    with col2:
        # Fitur Numerik (Disesuaikan dengan preprocessing StandardScaler) [2]
        ram = st.number_input("RAM (GB)", min_value=2, max_value=64, value=8)
        ssd = st.selectbox("Kapasitas SSD (GB)", [8-11])
        weight = st.number_input("Berat Laptop (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
        clockspeed = st.number_input("CPU Clock Speed (GHz)", min_value=0.9, max_value=4.0, value=2.5, step=0.1)
        ppi = st.number_input("PPI (Pixels Per Inch)", value=141.21)

    submit_button = st.form_submit_button("Estimasi Harga")

# 4. Logika Prediksi
if submit_button:
    # Data harus dalam bentuk DataFrame agar sesuai dengan ColumnTransformer di Pipeline
    input_data = pd.DataFrame({
        'Company': [company],
        'TypeName': [typename],
        'Ram': [ram],
        'Gpu': [gpu],
        'OpSys': [opsys],
        'Weight': [weight],
        'PPI': [ppi],
        'SSD_GB': [ssd],
        'Cpu_Family': [cpu_family],
        'Cpu_Clockspeed': [clockspeed]
    })
    
    # Prediksi menggunakan pipeline (Otomatis melakukan scaling & encoding)
    prediksi = model.predict(input_data)
    
    # Menampilkan hasil
    st.success(f"Estimasi Harga Laptop: € {prediksi[0]:,.2f}")