import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="NFL Player Dashboard",
    page_icon="🏈",
    layout="wide",
)

# ----------------------------------------------------------
# LOAD BASIC PLAYER STATS
# ----------------------------------------------------------
basic_path = Path("data") / "Basic_Stats.csv"

@st.cache
def load_basic_stats(path):
    return pd.read_csv(path)

if not basic_path.exists():
    st.error("Basic_Stats.csv not found in /data folder.")
    st.stop()

df = load_basic_stats(basic_path)

def fmt_int(value, default="N/A"):
    """Format a value as a whole number string (no decimals)."""
    v = pd.to_numeric(value, errors="coerce")
    if pd.isna(v):
        return default
    return str(int(round(v)))


# ----------------------------------------------------------
# LOAD PASSING CAREER STATS
# ----------------------------------------------------------
passing_path = Path("data") / "Career_Stats_Passing.csv"

@st.cache
def load_passing_stats(path):
    return pd.read_csv(path)

if passing_path.exists():
    passing_df = load_passing_stats(passing_path)
else:
    st.warning("Career_Stats_Passing.csv not found in /data folder.")
    passing_df = None

# ----------------------------------------------------------
# TITLE
# ----------------------------------------------------------
st.title("NFL Player Overview Dashboard")
st.markdown("Explore NFL player information and career statistics.")

st.write("---")

# ----------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------
st.sidebar.header("Filters")

# Filter: Team
teams = sorted(df["Current Team"].dropna().unique())
selected_teams = st.sidebar.multiselect(
    "Current Team",
    teams,
    default=teams
)

df_filtered = df[df["Current Team"].isin(selected_teams)]

# Filter: Position
positions = sorted(df["Position"].dropna().unique())
selected_positions = st.sidebar.multiselect(
    "Position",
    positions,
    default=positions
)

df_filtered = df_filtered[df_filtered["Position"].isin(selected_positions)]

# ----------------------------------------------------------
# SUMMARY METRICS
# ----------------------------------------------------------
st.subheader("Summary Statistics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Players Shown", len(df_filtered))

# Safely compute Avg Age
if "Age" in df_filtered.columns:
    age_series = pd.to_numeric(df_filtered["Age"], errors="coerce")
    if age_series.notna().any():
        col2.metric("Avg Age", f"{age_series.mean():.1f}")
    else:
        col2.metric("Avg Age", "N/A")

# Safely compute Avg Height
if "Height (inches)" in df_filtered.columns:
    h_series = pd.to_numeric(df_filtered["Height (inches)"], errors="coerce")
    if h_series.notna().any():
        col3.metric("Avg Height (in)", f"{h_series.mean():.1f}")
    else:
        col3.metric("Avg Height (in)", "N/A")

# Safely compute Avg Weight
if "Weight (lbs)" in df_filtered.columns:
    w_series = pd.to_numeric(df_filtered["Weight (lbs)"], errors="coerce")
    if w_series.notna().any():
        col4.metric("Avg Weight (lbs)", f"{w_series.mean():.1f}")
    else:
        col4.metric("Avg Weight (lbs)", "N/A")


st.write("---")

# ----------------------------------------------------------
# PLAYER LOOKUP
# ----------------------------------------------------------
st.subheader("Player Lookup")

player_list = sorted(df_filtered["Name"].unique())
selected_player = st.selectbox("Select a Player", player_list)

player_data = df_filtered[df_filtered["Name"] == selected_player].iloc[0]

# ----------------------------------------------------------
# PLAYER PROFILE
# ----------------------------------------------------------
st.markdown(f"## {player_data['Name']} — {player_data['Position']}")

col1, col2 = st.columns(2)

with col1:
    st.write("### Player Info")
    st.write(f"**Team:** {player_data['Current Team']}")
    st.write(f"**Number:** {fmt_int(player_data['Number'])}")
    st.write(f"**Age:** {fmt_int(player_data['Age'])}")
    st.write(f"**Experience:** {player_data['Experience']} years")
    st.write(f"**Status:** {player_data['Current Status']}")

with col2:
    st.write("### Physical Attributes")
    st.write(f"**Height:** {fmt_int(player_data['Height (inches)'])} inches")
    st.write(f"**Weight:** {fmt_int(player_data['Weight (lbs)'])} lbs")
    st.write(f"**College:** {player_data['College']}")
    st.write(f"**Birth Place:** {player_data['Birth Place']}")
    st.write(f"**Birthday:** {player_data['Birthday']}")


# ----------------------------------------------------------
# PASSING CAREER STATISTICS
# ----------------------------------------------------------
st.write("---")

if passing_df is not None:

    player_id = player_data["Player Id"]
    passing_player = passing_df[passing_df["Player Id"] == player_id]

    # Rename Sacks → Times Sacked for clarity
    if "Sacks" in passing_player.columns:
        passing_player = passing_player.rename(columns={"Sacks": "Times Sacked"})

    # Clean commas from numeric fields (e.g., "4,523")
    if "Passing Yards" in passing_player.columns:
        passing_player["Passing Yards"] = passing_player["Passing Yards"].astype(str).str.replace(",", "")

    # Clean "T" suffix from longest pass (e.g., "65T")
    if "Longest Pass" in passing_player.columns:
        passing_player["Longest Pass"] = passing_player["Longest Pass"].astype(str).str.replace("T", "", regex=False)

    numeric_cols_list = [
        "Games Played",
        "Passes Attempted",
        "Passes Completed",
        "Completion Percentage",
        "Pass Attempts Per Game",
        "Passing Yards",
        "Passing Yards Per Attempt",
        "Passing Yards Per Game",
        "TD Passes",
        "Percentage of TDs per Attempts",
        "Ints",
        "Int Rate",
        "Longest Pass",
        "Passes Longer than 20 Yards",
        "Passes Longer than 40 Yards",
        "Times Sacked",
        "Sacked Yards Lost",
        "Passer Rating",
    ]

    for col in numeric_cols_list:
        if col in passing_player.columns:
            passing_player[col] = pd.to_numeric(passing_player[col], errors="coerce")

    # Convert integer-like fields to int
    int_fields = [
        "Games Played",
        "Passes Attempted",
        "Passes Completed",
        "TD Passes",
        "Ints",
        "Longest Pass",
        "Passes Longer than 20 Yards",
        "Passes Longer than 40 Yards",
        "Times Sacked",
        "Sacked Yards Lost",
    ]

    for col in int_fields:
        if col in passing_player.columns:
            passing_player[col] = pd.to_numeric(passing_player[col], errors="coerce").fillna(0).astype(int)



    if passing_player.empty:
        st.markdown("")
    else:
        st.markdown("### Career Passing Summary")
        if "Player Id" in passing_player.columns:
            display_df = passing_player.drop(columns=["Player Id", "Name", "Position"])
        else:
            display_df = passing_player

        # Clean formatting for display (no trailing .0)
        format_dict = {}

        # Integer-like fields – force them to show as whole numbers
        int_like_cols = [
            "Games Played", "Passes Attempted", "Passes Completed",
            "TD Passes", "Ints", "Longest Pass",
            "Passes Longer than 20 Yards", "Passes Longer than 40 Yards",
            "Times Sacked", "Sacked Yards Lost"
        ]

        for col in int_like_cols:
            if col in display_df.columns:
                format_dict[col] = "{:.0f}"

        # Percentage-like fields – show 1 decimal
        percent_like_cols = [
            "Completion Percentage",
            "Percentage of TDs per Attempts",
            "Int Rate"
        ]

        for col in percent_like_cols:
            if col in display_df.columns:
                format_dict[col] = "{:.1f}"

        # Yardage / averages – show 1 decimal if needed
        one_decimal_cols = [
            "Passing Yards Per Attempt",
            "Passing Yards Per Game",
            "Pass Attempts Per Game",
            "Passer Rating"
        ]

        for col in one_decimal_cols:
            if col in display_df.columns:
                format_dict[col] = "{:.1f}"

        # Make a copy to format for display
        formatted_df = display_df.copy()

        # Integer-like fields – no decimals
        int_like_cols = [
            "Games Played", "Passes Attempted", "Passes Completed",
            "TD Passes", "Ints", "Longest Pass",
            "Passes Longer than 20 Yards", "Passes Longer than 40 Yards",
            "Times Sacked", "Sacked Yards Lost"
        ]

        for col in int_like_cols:
            if col in formatted_df.columns:
                formatted_df[col] = pd.to_numeric(formatted_df[col], errors="coerce").round(0).astype("Int64").astype(str)

        # Percentage-like fields – 1 decimal
        percent_like_cols = [
            "Completion Percentage",
            "Percentage of TDs per Attempts",
            "Int Rate"
        ]

        for col in percent_like_cols:
            if col in formatted_df.columns:
                formatted_df[col] = pd.to_numeric(formatted_df[col], errors="coerce").round(1).astype(str)

        # One-decimal numeric fields
        one_decimal_cols = [
            "Passing Yards Per Attempt",
            "Passing Yards Per Game",
            "Pass Attempts Per Game",
            "Passer Rating"
        ]

        for col in one_decimal_cols:
            if col in formatted_df.columns:
                formatted_df[col] = pd.to_numeric(formatted_df[col], errors="coerce").round(1).astype(str)

        st.dataframe(formatted_df)

        # Sort by year for plotting
        if "Year" in passing_player.columns:
            passing_player = passing_player.sort_values(by="Year")

        # List of numeric columns
        numeric_cols = [
            "Games Played",
            "Passes Attempted",
            "Passes Completed",
            "Completion Percentage",
            "Pass Attempts Per Game",
            "Passing Yards",
            "Passing Yards Per Attempt",
            "Passing Yards Per Game",
            "TD Passes",
            "Percentage of TDs per Attempts",
            "Ints",
            "Int Rate",
            "Times Sacked",
            "Sacked Yards Lost",
            "Passer Rating",
            "Passes Longer than 20 Yards",
            "Passes Longer than 40 Yards",
            "Longest Pass",
        ]

        numeric_cols = [c for c in numeric_cols if c in passing_player.columns]

        st.markdown("### Plot a Passing Metric Over Time")

        default_metric = "Passing Yards" if "Passing Yards" in numeric_cols else numeric_cols[0]

        metric = st.selectbox(
            "Select Metric",
            options=numeric_cols,
            index=numeric_cols.index(default_metric)
        )

        # Line chart
        if "Year" in passing_player.columns:
            passing_player["Year"] = pd.to_numeric(passing_player["Year"], errors="coerce").fillna(0).astype(int)


            # Sort by Year
            plot_df = passing_player.sort_values(by="Year")

            # Ensure metric column is numeric
            plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
            metric_values = plot_df[metric].dropna()

            if metric_values.empty:
                st.info("No valid numeric values available for this metric.")
            else:
                y_min = metric_values.min()
                y_max = metric_values.max()

                if y_max == y_min:
                    # Flat line – just plot without padding
                    padded_series = plot_df.set_index("Year")[metric]
                else:
                    padding = (y_max - y_min) * 0.15  # 15% padding
                    y_min_adj = max(0, y_min - padding)
                    y_max_adj = y_max + padding
                    padded_series = plot_df.set_index("Year")[metric].clip(y_min_adj, y_max_adj)

                st.line_chart(
                    padded_series,
                    use_container_width=True
                )

        else:
            st.info("No Year column available for line chart.")

# ----------------------------------------------------------
# RAW COLUMN DEBUGGER
# ----------------------------------------------------------
with st.expander("Show raw basic stats columns"):
    st.write(list(df.columns))

with st.expander("Show raw passing stats columns"):
    st.write(list(passing_df.columns))