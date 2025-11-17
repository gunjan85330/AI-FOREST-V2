import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import os

st.set_page_config(page_title="AI-FOREST Wildfire Risk App", layout="wide")

st.title(" AI-FOREST Wildfire Risk & Simulation App")

# ============================================================
# STEP 1 — DOWNLOAD LARGE PARQUET FILE FROM GOOGLE DRIVE
# ============================================================

PARQUET_URL = "https://drive.google.com/uc?id=1W2NwAwKVtz-jQcAZaKUAT4aDpUj-GNlv"

PARQUET_PATH = "master_with_lags.parquet"

@st.cache_resource
def download_parquet():
    if not os.path.exists(PARQUET_PATH):
        st.write("Downloading dataset (~300MB)...")
        gdown.download(PARQUET_URL, PARQUET_PATH, quiet=False)
    return pd.read_parquet(PARQUET_PATH)

df = download_parquet()
st.success("Dataset loaded successfully!")


# ============================================================
# STEP 2 — LOAD MODEL FROM LOCAL REPO
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("models/baseline_risk_model_improved.pkl")

model = load_model()
st.success("Model loaded successfully!")


# ============================================================
# RISK PREDICTION PANEL
# ============================================================

st.header("📊 Fire Risk Prediction")

lat = st.number_input("Latitude:", value=29.7)
lon = st.number_input("Longitude:", value=80.3)

if st.button("Predict Risk"):
    sample = df.iloc[:1].copy()
    sample["lat"] = lat
    sample["lon"] = lon

    prob = model.predict_proba(sample)[0][1]
    st.subheader(f" Predicted Fire Risk: **{prob:.4f}**")


# ============================================================
# FIRE SPREAD SIMULATION PANEL
# ============================================================

st.header(" Fire Spread Simulation")

# Pre-compute neighbors
def build_neighbors(df):
    lat_step = df.lat.diff().abs().median()
    lon_step = df.lon.diff().abs().median()

    df["lat_r"] = (df.lat / lat_step).round().astype(int)
    df["lon_r"] = (df.lon / lon_step).round().astype(int)

    cell_to_idx = {(r.lat_r, r.lon_r): i for i, r in df.iterrows()}

    neighbors = [[] for _ in range(len(df))]
    for i, r in df.iterrows():
        for dlat, dlon in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            key = (r.lat_r + dlat, r.lon_r + dlon)
            if key in cell_to_idx:
                neighbors[i].append(cell_to_idx[key])
    return neighbors

neighbors = build_neighbors(df)

start_lat = st.number_input("Ignition Point Latitude", value=29.7)
start_lon = st.number_input("Ignition Point Longitude", value=80.3)
steps = st.slider("Simulation Steps", 1, 20, 10)
spread_factor = st.slider("Spread Speed Factor", 0.0, 1.0, 0.5)

if st.button("Run Simulation"):
    dist = np.abs(df.lat - start_lat) + np.abs(df.lon - start_lon)
    start_idx = dist.idxmin()

    burning = {start_idx}
    burned_history = [burning.copy()]

    for _ in range(steps):
        new_fire = set()
        for cell in burning:
            for n in neighbors[cell]:
                risk = model.predict_proba(df.iloc[[n]])[0][1]
                if risk > spread_factor:
                    new_fire.add(n)
        burning = burning.union(new_fire)
        burned_history.append(burning.copy())

    st.subheader(" Simulation Completed")
    st.write(f"Total burned cells after {steps} steps: {len(burning)}")

    st.write("Simulation (step-by-step):")
    for t, burnset in enumerate(burned_history):
        st.write(f"Step {t}: {len(burnset)} burning cells")
