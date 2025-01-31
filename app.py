import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
def load_data():
    data = pd.read_csv("japan_heart_attack_dataset.csv")
    return data

# Load trained model
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

st.title("Prediksi Serangan Jantung")

# Load Data
data = load_data()
st.subheader("Preview Dataset")
st.write(data.head())

# Visualisasi
st.subheader("Visualisasi Data")
fig, ax = plt.subplots()
sns.countplot(data=data, x="target", ax=ax)
st.pyplot(fig)

# Input fitur untuk prediksi
st.subheader("Prediksi Risiko Serangan Jantung")
columns = data.drop(columns=["target"]).columns
input_data = {}
for col in columns:
    input_data[col] = st.number_input(f"{col}", value=float(data[col].median()))

# Prediksi
model = load_model()
if st.button("Prediksi"): 
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    st.write(f"Hasil Prediksi: {'Berisiko' if prediction[0] == 1 else 'Tidak Berisiko'}")
