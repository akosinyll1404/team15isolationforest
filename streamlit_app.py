import streamlit as st
import pandas as pd
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt

# Load ONNX model
session = ort.InferenceSession("isolation_forest.onnx")

def preprocess(df):
    # Match the training column names
    required_cols = ["ph", "temperature_degC", "turbidity_fnu"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}. Please check your file headers.")
        return None
    return df[required_cols].astype(np.float32).values

def predict(df):
    input_data = preprocess(df)
    if input_data is None:
        return None
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
    if predictions is not None:
        df["Prediction"] = np.where(predictions == -1, "Anomaly", "Normal")

        st.write("✅ Processed Data with Predictions")
        st.dataframe(df)

        # Visualization
        st.subheader("Sensor Trends with Anomaly Flags")
        fig, ax = plt.subplots()
        ax.plot(df.index, df["temperature_degC"], label="Temperature (°C)")
        ax.plot(df.index, df["turbidity_fnu"], label="Turbidity (FNU)")
        ax.plot(df.index, df["ph"], label="pH")

        # Highlight anomalies
        anomalies = df[df["Prediction"] == "Anomaly"]
        ax.scatter(anomalies.index, anomalies["temperature_degC"], color="red", label="Anomalies")

        ax.legend()
        st.pyplot(fig)
