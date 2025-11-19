import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import os
from sklearn.neighbors import KDTree

st.set_page_config(page_title="AI FOREST Wildfire Risk", layout="wide")
st.title("AI-FOREST Wildfire Prediction & Spread Simulation")

# ============================================================
# 1) LOAD LARGE PARQUET FROM GOOGLE DRIVE
# ============================================================

PARQUET_URL = "https://drive.google.com/uc?id=1W2NwAwKVtz-jQcAZaKUAT4aDpUj-GNlv"
PARQUET_PATH = "master_with_lags.parquet"

@st.cache_resource
def load_dataset():
    if not os.path.exists(PARQUET_PATH):
        st.write("Downloading dataset (~300MB)...")
        gdown.download(PARQUET_URL, PARQUET_PATH, quiet=False)

    df = pd.read_parquet(PARQUET_PATH)

    # Clean coordinates
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    return df

df = load_dataset()
st.success("Dataset loaded successfully!")

# ============================================================
# 2) LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("models/baseline_risk_model_improved.pkl")

model = load_model()
st.success("Model loaded successfully!")

# ============================================================
# 3) FEATURES USED BY MODEL
# ============================================================

FEATURES = [
 'temp_c','rain_mm','wind_u','wind_v','pressure','wind','humidity','vpd',
 'fire_today_lag_1','fire_today_lag_3','fire_today_lag_7',
 'temp_c_lag_1','temp_c_lag_3','temp_c_lag_7',
 'rain_mm_lag_1','rain_mm_lag_3','rain_mm_lag_7',
 'humidity_lag_1','humidity_lag_3','humidity_lag_7',
 'vpd_lag_1','vpd_lag_3','vpd_lag_7',
 'wind_lag_1','wind_lag_3','wind_lag_7',
 'wind_u_lag_1','wind_u_lag_3','wind_u_lag_7',
 'wind_v_lag_1','wind_v_lag_3','wind_v_lag_7',
 'pressure_lag_1','pressure_lag_3','pressure_lag_7'
]

# ============================================================
# 4) KD-TREE NEIGHBOR SEARCH (VERY FAST)
# ============================================================

@st.cache_resource
def build_kdtree(df):
    tree = KDTree(df[["lat", "lon"]], leaf_size=40)
    return tree

tree = build_kdtree(df)
st.success("KDTree built successfully!")

# ============================================================
# 5) FIRE RISK PREDICTION
# ============================================================

st.header("Fire Risk Prediction")

lat = st.number_input("Latitude", value=29.7)
lon = st.number_input("Longitude", value=80.3)

if st.button("Predict Fire Risk"):

    dist = ((df["lat"] - lat).abs() + (df["lon"] - lon).abs())
    idx = dist.idxmin()

    sample = df.loc[idx, FEATURES].to_frame().T.fillna(0)
    prob = model.predict_proba(sample)[0][1]

    st.subheader(f"Predicted Fire Risk: {prob:.4f}")

# ============================================================
# 6) FIRE SPREAD SIMULATION (ADVANCED)
# ============================================================

st.header("Fire Spread Simulation")

start_lat = st.number_input("Ignition Latitude", value=29.7)
start_lon = st.number_input("Ignition Longitude", value=80.3)
steps = st.slider("Steps", 1, 50, 15)
spread_factor = st.slider("Base Spread Threshold", 0.0, 1.0, 0.05)

import random

if st.button("Run Simulation"):

    # Find starting cell
    dist = ((df["lat"] - start_lat).abs() + (df["lon"] - start_lon).abs())
    start_idx = dist.idxmin()

    burning = {start_idx}
    history = [burning.copy()]

    for step in range(steps):
        new_fire = set()

        for cell in burning:

            # Get up to 100 nearest neighbors
            _, inds = tree.query(df.loc[[cell], ["lat", "lon"]], k=100)
            neighbors = inds[0][1:]

            for n in neighbors:

                # Weather features
                sample = df.loc[n, FEATURES].to_frame().T.fillna(0)
                risk = model.predict_proba(sample)[0][1]

                # Dynamic threshold: fire spreads more easily with more burning around
                local_factor = spread_factor * (1 - min(len(burning) / 300, 0.8))

                # Probabilistic ignition
                random_threshold = random.uniform(0, 1)

                # Combined ignition rule
                if risk > local_factor and (risk + random.uniform(0,0.1)) > random_threshold:
                    new_fire.add(n)

        # Update burned area
        burning = burning.union(new_fire)
        history.append(burning.copy())

    st.subheader("Simulation Complete")
    st.write(f"Total burned cells after {steps} steps: {len(burning)}")
    
    for t, burnset in enumerate(history):
        st.write(f"Step {t}: {len(burnset)} burning cells")
