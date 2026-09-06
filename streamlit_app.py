import streamlit as st
import pandas as pd
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt

# Load ONNX model
session = ort.InferenceSession("isolation_forest.onnx")

# Expected training columns
expected_cols = {
    "ph": ["ph", "pH", "ph_value"],
    "temperature_degC": ["temperature", "temp", "temperature_degC", "temp_c"],
    "turbidity_fnu": ["turbidity", "turbidity_fnu", "ntu"]
}

def auto_map_columns(df):
    """Try to automatically map dataset columns to expected names."""
    df.columns = df.columns.str.strip().str.lower()
    mapping = {}
    for target, aliases in expected_cols.items():
        for alias in aliases:
            if alias.lower() in df.columns:
                mapping[target] = alias.lower()
                break
    return mapping

def preprocess(df, mapping):
    try:
        # Select the mapped columns
        selected = df[[mapping["ph"], mapping["temperature_degC"], mapping["turbidity_fnu"]]]

        # Convert to numeric, coerce errors → NaN
        selected = selected.apply(pd.to_numeric, errors="coerce")

        # Fill NaN with 0 or forward-fill (choose strategy)
        selected = selected.fillna(0)

        return selected.astype(np.float32).values
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        return None

def predict(df, mapping):
    input_data = preprocess(df, mapping)
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

    # Try auto-mapping
    auto_mapping = auto_map_columns(df)

    st.subheader("🔧 Column Mapping")
    ph_col = st.selectbox("Select pH column", df.columns, index=df.columns.get_loc(auto_mapping.get("ph", df.columns[0])))
    temp_col = st.selectbox("Select Temperature column", df.columns, index=df.columns.get_loc(auto_mapping.get("temperature_degC", df.columns[0])))
    turb_col = st.selectbox("Select Turbidity column", df.columns, index=df.columns.get_loc(auto_mapping.get("turbidity_fnu", df.columns[0])))

    # Final mapping (user can override auto-map)
    final_mapping = {
        "ph": ph_col,
        "temperature_degC": temp_col,
        "turbidity_fnu": turb_col
    }

    # Run anomaly detection
    predictions = predict(df, final_mapping)
    if predictions is not None:
        df["Prediction"] = np.where(predictions == -1, "Anomaly", "Normal")

        st.write("✅ Processed Data with Predictions")
        st.dataframe(df)

        # Download results
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results as CSV", csv, "predictions.csv", "text/csv")

        # Visualization
        st.subheader("📈 Sensor Trends with Anomaly Flags")
        fig, ax = plt.subplots()
        ax.plot(df.index, df[temp_col], label="Temperature (°C)")
        ax.plot(df.index, df[turb_col], label="Turbidity (FNU)")
        ax.plot(df.index, df[ph_col], label="pH")

        # Highlight anomalies
        anomalies = df[df["Prediction"] == "Anomaly"]
        ax.scatter(anomalies.index, anomalies[temp_col], color="red", label="Anomalies")

        ax.legend()
        st.pyplot(fig)
