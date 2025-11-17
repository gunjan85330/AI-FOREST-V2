import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import os

st.set_page_config(page_title="AI FOREST Wildfire Risk", layout="wide")

st.title(" AI-FOREST Wildfire Prediction & Spread Simulation")


# ============================================================
# 1) DOWNLOAD LARGE PARQUET FILE FROM GOOGLE DRIVE
# ============================================================

PARQUET_URL = "https://drive.google.com/uc?id=1W2NwAwKVtz-jQcAZaKUAT4aDpUj-GNlv"
PARQUET_PATH = "master_with_lags.parquet"

@st.cache_resource
def load_dataset():
    if not os.path.exists(PARQUET_PATH):
        st.write(" Downloading dataset (~300MB)...")
        gdown.download(PARQUET_URL, PARQUET_PATH, quiet=False)

    df = pd.read_parquet(PARQUET_PATH)

    # Clean lat/lon
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
# 3) MODEL FEATURES
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
# 4) PREDICT RISK PANEL
# ============================================================

st.header(" Fire Risk Prediction")

lat = st.number_input("Latitude", value=29.7)
lon = st.number_input("Longitude", value=80.3)

if st.button("Predict Fire Risk"):

    # Find nearest grid cell
    dist = ((df["lat"] - lat).abs() + (df["lon"] - lon).abs())
    idx = dist.idxmin()

    # Use only required model features
    sample = df.loc[idx, FEATURES].to_frame().T
    sample = sample.fillna(0)

    prob = model.predict_proba(sample)[0][1]

    st.subheader(f" Predicted Fire Risk: **{prob:.4f}**")


# ============================================================
# 5) BUILD NEIGHBOR GRID FOR SIMULATION
# ============================================================

def build_neighbors(df):
    # Convert to numeric, force invalid values to NaN
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # Replace inf / -inf with nan
    df["lat"].replace([np.inf, -np.inf], np.nan, inplace=True)
    df["lon"].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop unusable rows
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # Detect spacing
    lat_step = df["lat"].diff().abs().median()
    lon_step = df["lon"].diff().abs().median()

    # If spacing fails, force fallback
    if pd.isna(lat_step) or lat_step == 0:
        lat_step = 0.01
    if pd.isna(lon_step) or lon_step == 0:
        lon_step = 0.01

    # SAFE integer assignments
    df["lat_r"] = (df["lat"] / lat_step).replace([np.inf, -np.inf], 0).fillna(0).round().astype(int)
    df["lon_r"] = (df["lon"] / lon_step).replace([np.inf, -np.inf], 0).fillna(0).round().astype(int)

    # Build mapping
    cell_to_idx = {(r.lat_r, r.lon_r): i for i, r in df.iterrows()}

    # Build neighbor list
    neighbors = [[] for _ in range(len(df))]
    directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

    for i, r in df.iterrows():
        for dlat, dlon in directions:
            key = (r.lat_r + dlat, r.lon_r + dlon)
            if key in cell_to_idx:
                neighbors[i].append(cell_to_idx[key])

    return neighbors, df



# ============================================================
# 6) FIRE SPREAD SIMULATION
# ============================================================

st.header(" Fire Spread Simulation")

start_lat = st.number_input("Ignition Latitude", value=29.7)
start_lon = st.number_input("Ignition Longitude", value=80.3)
steps = st.slider("Steps", 1, 20, 10)
spread_factor = st.slider("Spread Factor (0–1)", 0.0, 1.0, 0.5)

if st.button("Run Simulation"):

    # Find starting cell
    dist = ((df["lat"] - start_lat).abs() + (df["lon"] - start_lon).abs())
    start_idx = dist.idxmin()

    burning = {start_idx}
    history = [burning.copy()]

    for _ in range(steps):
        new_fire = set()

        for cell in burning:
            for n in neighbors[cell]:
                sample = df.loc[n, FEATURES].to_frame().T
                sample = sample.fillna(0)

                risk = model.predict_proba(sample)[0][1]

                if risk > spread_factor:
                    new_fire.add(n)

        burning = burning.union(new_fire)
        history.append(burning.copy())

    st.subheader(" Simulation Complete")
    st.write(f"Total burned cells: {len(burning)}")

    for t, burnset in enumerate(history):
        st.write(f"Step {t}: {len(burnset)} burning cells")
