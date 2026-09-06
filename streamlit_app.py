import streamlit as st
import pandas as pd
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt

# Load ONNX model
session = ort.InferenceSession("isolation_forest.onnx")

def preprocess(df):
    # Only keep the 3 sensor columns
    return df[["temperature", "turbidity", "pH"]].astype(np.float32).values

def predict(df):
    input_data = preprocess(df)
    inputs = {session.get_inputs()[0].name: input_data}
    outputs = session.run(None, inputs)
    return outputs[0]  # Isolation Forest outputs -1 (anomaly) or 1 (normal)

# Streamlit UI
st.title("💧 Water Quality Anomaly Detection")

uploaded_file = st.file_uploader("Upload sensor CSV/Excel", type=["csv", "xlsx"])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("📊 Raw Data Preview", df.head())

    # Run anomaly detection
    predictions = predict(df)
    df["Prediction"] = np.where(predictions == -1, "Anomaly", "Normal")

    st.write("✅ Processed Data with Predictions")
    st.dataframe(df)

    # Visualization
    st.subheader("Sensor Trends with Anomaly Flags")
    fig, ax = plt.subplots()
    ax.plot(df.index, df["temperature"], label="Temperature")
    ax.plot(df.index, df["turbidity"], label="Turbidity")
    ax.plot(df.index, df["pH"], label="pH")

    # Highlight anomalies
    anomalies = df[df["Prediction"] == "Anomaly"]
    ax.scatter(anomalies.index, anomalies["temperature"], color="red", label="Anomalies")

    ax.legend()
    st.pyplot(fig)
