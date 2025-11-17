import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Load Data & Model
# ---------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_parquet("data/master_with_lags.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_resource
def load_model():
    return joblib.load("models/baseline_risk_model_improved.pkl")

df = load_data()
model = load_model()

# ---------------------------------------------------
# Build Neighbors
# ---------------------------------------------------

def build_neighbors(df_day):
    df_day = df_day.reset_index(drop=True)

    # Detect grid spacing
    lat_step = df_day["lat"].sort_values().diff().abs().replace(0, np.nan).min()
    lon_step = df_day["lon"].sort_values().diff().abs().replace(0, np.nan).min()

    df_day["lat_r"] = df_day["lat"].round(3)
    df_day["lon_r"] = df_day["lon"].round(3)

    df_sorted = df_day.sort_values(["lat_r", "lon_r"]).reset_index(drop=True)

    coord_to_index = {
        (row.lat_r, row.lon_r): i
        for i, row in df_sorted.iterrows()
    }

    # 8-direction neighbors
    directions = [
        (lat_step, 0), (-lat_step, 0),
        (0, lon_step), (0, -lon_step),
        (lat_step, lon_step), (lat_step, -lon_step),
        (-lat_step, lon_step), (-lat_step, -lon_step),
    ]

    neighbors = [[] for _ in range(len(df_sorted))]

    for i, row in df_sorted.iterrows():
        lat = row.lat_r
        lon = row.lon_r
        for dlat, dlon in directions:
            neigh = (round(lat + dlat, 3), round(lon + dlon, 3))
            if neigh in coord_to_index:
                neighbors[i].append(coord_to_index[neigh])

    return df_sorted, neighbors

# ---------------------------------------------------
# Fire Simulation
# ---------------------------------------------------

def simulate_fire(df_day, start_indices, neighbors, steps=10, spread_factor=1.5):
    n = len(df_day)
    burning = np.zeros((steps+1, n), dtype=int)

    for idx in start_indices:
        burning[0][idx] = 1

    rp = df_day["risk_prob"].values
    vpd = df_day["vpd"].values
    wind = df_day["wind"].values

    # enhance probabilities
    risk_enhanced = np.minimum(1.0, (rp * 5)**0.7)

    vpd_norm = (vpd - vpd.min()) / (vpd.max() - vpd.min() + 1e-6)
    wind_norm = (wind - wind.min()) / (wind.max() - wind.min() + 1e-6)

    MIN_PROB = 0.01

    for t in range(steps):
        for i in range(n):
            if burning[t][i] == 1:
                burning[t+1][i] = 1
                continue

            if not any(burning[t][j] == 1 for j in neighbors[i]):
                continue

            prob = risk_enhanced[i] * spread_factor
            prob *= (1 + vpd_norm[i])
            prob *= (1 + wind_norm[i])
            prob = max(prob, MIN_PROB)
            prob = min(prob, 0.95)

            if np.random.rand() < prob:
                burning[t+1][i] = 1

    return burning

# ---------------------------------------------------
# Convert Burning → Grid
# ---------------------------------------------------

def build_grid_data(df_day):
    lats = sorted(df_day["lat"].unique())
    lons = sorted(df_day["lon"].unique())

    lat_to_idx = {lat: i for i, lat in enumerate(lats)}
    lon_to_idx = {lon: i for i, lon in enumerate(lons)}

    H = len(lats)
    W = len(lons)

    return lats, lons, lat_to_idx, lon_to_idx, H, W

def burning_to_grid(burning_t, df_day, lat_to_idx, lon_to_idx, H, W):
    grid = np.zeros((H, W))
    for i, row in df_day.iterrows():
        if burning_t[i] == 1:
            r = lat_to_idx[row.lat]
            c = lon_to_idx[row.lon]
            grid[r, c] = 1
    return grid

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------

st.title(" AI FOREST – Wildfire Risk & Spread Simulation")

# Choose date
dates = sorted(df["date"].unique())
selected_date = st.selectbox("Select Date", dates)

df_day = df[df["date"] == selected_date].copy().reset_index(drop=True)

# Prepare features
drop_cols = ["latitude", "longitude", "lat", "lon", "lat_r", "lon_r", "date",
             "cell_id", "fire_today", "fire_tomorrow"]
feature_cols = [c for c in df_day.columns if c not in drop_cols]

df_day["risk_prob"] = model.predict_proba(df_day[feature_cols])[:,1]

# Show risk map (as table values)
st.subheader(" Risk Map Preview (Top 10 highest risk locations)")
st.dataframe(df_day.nlargest(10, "risk_prob")[["lat","lon","risk_prob"]])

# Build neighbors
df_sorted, neighbors = build_neighbors(df_day)
lats, lons, lat_to_idx, lon_to_idx, H, W = build_grid_data(df_sorted)

# ignition
selected_ignition = st.selectbox(" Choose Ignition Method", ["Highest Risk Cell", "Manual Coordinates"])

if selected_ignition == "Highest Risk Cell":
    start_idx = df_sorted["risk_prob"].idxmax()
else:
    lat_in = st.number_input("Enter Latitude:", value=float(df_sorted["lat"].iloc[0]))
    lon_in = st.number_input("Enter Longitude:", value=float(df_sorted["lon"].iloc[0]))
    start_idx = df_sorted[((df_sorted.lat - lat_in)**2 + (df_sorted.lon - lon_in)**2).idxmin()]

start_indices = [start_idx]

# Simulation parameters
steps = st.slider("Simulation Steps", 1, 15, 10)
spread_factor = st.slider("Spread Factor", 0.5, 10.0, 1.5)

# Run simulation
burning = simulate_fire(df_sorted, start_indices, neighbors, steps=steps, spread_factor=spread_factor)

# Show results
st.subheader(" Fire Spread Over Time")

for t in range(steps+1):
    grid = burning_to_grid(burning[t], df_sorted, lat_to_idx, lon_to_idx, H, W)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(grid, cmap="hot", interpolation="nearest")
    ax.set_title(f"Step {t}")
    st.pyplot(fig)

st.success("Simulation Completed!")
