import streamlit as st
import pickle
import pandas as pd

# Load model
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)

# Judul aplikasi
st.title("Prediksi Gender Berdasarkan Data Kesehatan")
st.write(
    "Masukkan data pengguna untuk mendapatkan prediksi apakah gendernya **Male** atau **Female** berdasarkan fitur kesehatan."
)

# Input fitur dari pengguna
age = st.number_input("Usia", min_value=0, max_value=120, value=30)
region = st.selectbox("Wilayah", options=["Urban", "Rural"])
cholesterol = st.number_input("Kolesterol", min_value=100, max_value=600, value=200)
systolic_bp = st.number_input("Tekanan Darah Sistolik", min_value=80, max_value=200, value=120)
diastolic_bp = st.number_input("Tekanan Darah Diastolik", min_value=50, max_value=120, value=80)

# Nama fitur yang sesuai dengan model saat training
feature_names = ["Age", "Region", "Cholesterol_Level", "Systolic_BP", "Diastolic_BP"]

# Tombol prediksi
if st.button("Prediksi"):
    try:
        # Load model
        model = load_model()
        
        # Preprocessing input (encoding untuk 'Region')
        region_encoded = 1 if region == "Urban" else 0
        input_data = pd.DataFrame(
            [[age, region_encoded, cholesterol, systolic_bp, diastolic_bp]],
            columns=feature_names,
        )

        # Prediksi
        prediction = model.predict(input_data)
        result = "Male" if prediction[0] == 1 else "Female"

        # Tampilkan hasil
        st.subheader("Hasil Prediksi:")
        st.write(f"Pengguna diprediksi: **{result}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
