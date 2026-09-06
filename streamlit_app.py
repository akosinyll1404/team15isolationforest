import streamlit as st
import pandas as pd
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt

# Load ONNX model
session = ort.InferenceSession("isolation_forest.onnx")

# Expected training columns and aliases
expected_cols = {
    "ph": ["ph", "pH", "ph_value"],
    "temperature_degC": ["temperature", "temp", "temperature_degC", "temp_c"],
    "turbidity_fnu": ["turbidity", "turbidity_fnu", "ntu"]
}

def auto_map_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    mapping = {}
    for target, aliases in expected_cols.items():
        for alias in aliases:
            if alias.lower() in df.columns:
                mapping[target] = alias.lower()
                break
    return mapping

def engineer_features(df, mapping, window=180):
    df = df.copy()
    for col in [mapping["ph"], mapping["temperature_degC"], mapping["turbidity_fnu"]]:
        roll_mean = df[col].rolling(window, min_periods=window//3).mean()
        roll_std = df[col].rolling(window, min_periods=window//3).std()
        df[f"{col}_roll_z"] = (df[col] - roll_mean) / roll_std
        df[f"{col}_diff1"] = df[col].diff()
    df = df.bfill().ffill()
    return df

def preprocess(df, mapping):
    df = engineer_features(df, mapping)
    feature_cols = [
        mapping["ph"], f"{mapping['ph']}_roll_z", f"{mapping['ph']}_diff1",
        mapping["temperature_degC"], f"{mapping['temperature_degC']}_roll_z", f"{mapping['temperature_degC']}_diff1",
        mapping["turbidity_fnu"], f"{mapping['turbidity_fnu']}_roll_z", f"{mapping['turbidity_fnu']}_diff1"
    ]
    selected = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    arr = selected.to_numpy(dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    st.write("Input shape:", arr.shape)   # should be (n_samples, 9)
    st.write("Input dtype:", arr.dtype)   # should be float32
    return arr

def predict(df, mapping):
    input_data = preprocess(df, mapping)
    if input_data is None:
        return None
    inputs = {session.get_inputs()[0].name: input_data}
    outputs = session.run(None, inputs)
    preds = outputs[0].flatten()  # ensure 1D
    return preds

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

    # Auto-map
    auto_mapping = auto_map_columns(df)

    st.subheader("🔧 Column Mapping")
    ph_col = st.selectbox("Select pH column", df.columns, index=df.columns.get_loc(auto_mapping.get("ph", df.columns[0])) if "ph" in auto_mapping else 0)
    temp_col = st.selectbox("Select Temperature column", df.columns, index=df.columns.get_loc(auto_mapping.get("temperature_degC", df.columns[0])) if "temperature_degC" in auto_mapping else 0)
    turb_col = st.selectbox("Select Turbidity column", df.columns, index=df.columns.get_loc(auto_mapping.get("turbidity_fnu", df.columns[0])) if "turbidity_fnu" in auto_mapping else 0)

    final_mapping = {
        "ph": ph_col,
        "temperature_degC": temp_col,
        "turbidity_fnu": turb_col
    }

# Run anomaly detection
predictions = predict(df, final_mapping)
if predictions is not None and len(predictions) == len(df):
    df["Prediction"] = np.where(predictions == -1, "Anomaly", "Normal")

    st.write("✅ Processed Data with Predictions")
    st.dataframe(df)

    # Download full results
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results as CSV", csv, "predictions.csv", "text/csv")

    # Download anomalies only
    anomalies = df[df["Prediction"] == "Anomaly"]
    if not anomalies.empty:
        csv_anom = anomalies.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Anomalies Only", csv_anom, "anomalies.csv", "text/csv")

    # Handle datetime if present
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df.set_index("datetime", inplace=True)

    # Visualization controls
    st.subheader("📈 Sensor Trends")
    view_mode = st.radio("Select view mode:", ["Raw Values", "Anomalies", "Deviation"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # --- pH ---
    axes[0].plot(df.index, df[ph_col], color="black", label="pH")
    if view_mode == "Anomalies":
        axes[0].scatter(anomalies.index, anomalies[ph_col], color="red", label="Anomalies")
    elif view_mode == "Deviation":
        axes[0].plot(df.index, df[f"{ph_col}_roll_z"], color="blue", label="Deviation (z-score)")
    axes[0].set_ylabel("pH")
    axes[0].legend(loc="upper right")
    axes[0].set_title("pH over time")

    # --- Turbidity ---
    axes[1].plot(df.index, df[turb_col], color="black", label="Turbidity (FNU)")
    if view_mode == "Anomalies":
        axes[1].scatter(anomalies.index, anomalies[turb_col], color="red", label="Anomalies")
    elif view_mode == "Deviation":
        axes[1].plot(df.index, df[f"{turb_col}_roll_z"], color="blue", label="Deviation (z-score)")
    axes[1].set_ylabel("Turbidity (FNU)")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Turbidity over time")

    # --- Temperature ---
    axes[2].plot(df.index, df[temp_col], color="black", label="Temperature (°C)")
    if view_mode == "Anomalies":
        axes[2].scatter(anomalies.index, anomalies[temp_col], color="red", label="Anomalies")
    elif view_mode == "Deviation":
        axes[2].plot(df.index, df[f"{temp_col}_roll_z"], color="blue", label="Deviation (z-score)")
    axes[2].set_ylabel("Temperature (°C)")
    axes[2].legend(loc="upper right")
    axes[2].set_title("Temperature over time")

    plt.xlabel("Datetime")
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.error("Prediction length mismatch. Check preprocessing and model input.")
