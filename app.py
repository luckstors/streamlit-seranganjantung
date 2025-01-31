import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and label encoders
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)

def load_encoders():
    return {
        "Gender": {0: "Female", 1: "Male"},
        "Region": {"Urban": 1, "Rural": 0},
    }

# App Title
st.title("📊 Prediksi Gender Berdasarkan Fitur Kesehatan")
st.markdown("Aplikasi ini menggunakan model Machine Learning untuk memprediksi gender berdasarkan data kesehatan pengguna.")

# Sidebar Navigation
st.sidebar.title("Navigasi")
option = st.sidebar.radio("Pilih Halaman", ["Visualisasi Data", "Prediksi"])

# Load Dataset for Visualization
if option == "Visualisasi Data":
    st.header("Visualisasi Data")
    uploaded_file = st.file_uploader("Unggah file CSV untuk visualisasi", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("### Contoh Data")
        st.dataframe(data.head())

        st.write("### Distribusi Usia")
        fig, ax = plt.subplots()
        sns.histplot(data["Age"], kde=True, ax=ax, color="blue")
        st.pyplot(fig)

        st.write("### Korelasi Antar Fitur")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# Prediction Page
elif option == "Prediksi":
    st.header("Prediksi Gender")
    model = load_model()
    encoders = load_encoders()

    # User Input
    st.write("Masukkan informasi berikut untuk prediksi:")
    age = st.number_input("Umur", min_value=1, max_value=120, value=30)
    region = st.selectbox("Wilayah", options=["Urban", "Rural"])
    cholesterol = st.number_input("Kolesterol", min_value=100, max_value=600, value=200)
    systolic_bp = st.number_input("Tekanan Darah Sistolik", min_value=80, max_value=200, value=120)
    diastolic_bp = st.number_input("Tekanan Darah Diastolik", min_value=50, max_value=120, value=80)

    if st.button("Prediksi"):
        # Preprocessing Input
        input_data = pd.DataFrame(
            {
                "Age": [age],
                "Region": [encoders["Region"][region]],
                "Cholesterol_Level": [cholesterol],
                "Systolic_BP": [systolic_bp],
                "Diastolic_BP": [diastolic_bp],
            }
        )

        # Predict
        prediction = model.predict(input_data)
        gender = encoders["Gender"][prediction[0]]
        st.write(f"Prediksi Gender: **{gender}**")
