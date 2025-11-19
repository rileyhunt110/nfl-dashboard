import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="NFL Player Dashboard",
    page_icon="🏈",
    layout="wide",
)

data_path = Path("data") / "Basic_Stats.csv"

@st.cache
def load_basic_stats(path):
    df = pd.read_csv(path)
    return df

if not data_path.exists():
    st.error("Could not find Basic_Stats.csv in the data/ folder.")
    st.stop()

df = load_basic_stats(data_path)

st.title("🏈 NFL Player Overview Dashboard")
st.markdown(
    "This dashboard lets you explore basic information for NFL players "
    "using the `Basic_Stats.csv` file."
)
st.write("---")

# --- Sidebar filters ---
st.sidebar.header("Filters")

# Filter by Team
teams = sorted(df["Current Team"].dropna().unique())
selected_teams = st.sidebar.multiselect(
    "Current Team",
    teams,
    default=teams
)
df_filtered = df[df["Current Team"].isin(selected_teams)]

# Filter by Position
positions = sorted(df["Position"].dropna().unique())
selected_positions = st.sidebar.multiselect(
    "Position",
    positions,
    default=positions
)
df_filtered = df_filtered[df_filtered["Position"].isin(selected_positions)]

# -------------------------------
# Summary Metrics
# -------------------------------
st.subheader("📌 Summary Statistics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Players Shown", len(df_filtered))

# Numeric fields available
if "Age" in df_filtered.columns:
    col2.metric("Avg Age", f"{df_filtered['Age'].mean():.1f}")

if "Height (inches)" in df_filtered.columns:
    col3.metric("Avg Height (in)", f"{df_filtered['Height (inches)'].mean():.1f}")

if "Weight (lbs)" in df_filtered.columns:
    col4.metric("Avg Weight (lbs)", f"{df_filtered['Weight (lbs)'].mean():.1f}")

st.write("---")

# -------------------------------
# Player Table Preview
# -------------------------------
st.subheader("🧾 Player Table")

st.dataframe(df_filtered)

with st.expander("Show column names"):
    st.write(list(df.columns))