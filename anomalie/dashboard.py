# dashboard.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    layout="wide"
)

st.title("📊 Market Anomaly Detection Dashboard")


# =========================
# Data loading
# =========================
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["SEANCE"] = pd.to_datetime(df["SEANCE"])
    return df


data_path = st.sidebar.text_input(
    "Path to detected data CSV",
    "data/detected_anomalies.csv"
)

df = load_data(data_path)


# =========================
# Sidebar filters
# =========================
codes = df["CODE"].unique()
selected_code = st.sidebar.selectbox("Select Asset (CODE)", codes)

df_asset = df[df["CODE"] == selected_code]


# =========================
# Price chart with anomalies
# =========================
st.subheader(f"Price evolution – {selected_code}")

fig = go.Figure()

# Price line
fig.add_trace(
    go.Scatter(
        x=df_asset["SEANCE"],
        y=df_asset["CLOTURE"],
        mode="lines",
        name="Price"
    )
)

# Anomaly markers
anomalies = df_asset[df_asset["is_anomaly"]]

fig.add_trace(
    go.Scatter(
        x=anomalies["SEANCE"],
        y=anomalies["CLOTURE"],
        mode="markers",
        marker=dict(color="red", size=8),
        name="Anomaly"
    )
)

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Price",
    legend=dict(orientation="h")
)

st.plotly_chart(fig, use_container_width=True)


# =========================
# Top 5 anomalies today
# =========================
st.subheader("🚨 Top 5 Anomalies Today")

today = df["SEANCE"].dt.date.max()

top_anomalies = (
    df[
        (df["SEANCE"].dt.date == today)
        & (df["is_anomaly"])
    ]
    .sort_values("SEANCE", ascending=False)
    .head(5)
)

if top_anomalies.empty:
    st.info("No anomalies detected today.")
else:
    st.dataframe(
        top_anomalies[
            [
                "SEANCE",
                "CODE",
                "volume_anomaly",
                "price_anomaly",
                "pattern_anomaly",
            ]
        ]
    )


# =========================
# Alert section
# =========================
st.subheader("⚠️ Live Alerts")

if not anomalies.empty:
    st.warning(
        f"{len(anomalies)} anomalies detected for {selected_code}"
    )
else:
    st.success("No anomalies detected for this asset.")
