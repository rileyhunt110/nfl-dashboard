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
# LOAD RUSHING CAREER STATS
# ----------------------------------------------------------
rushing_path = Path("data") / "Career_Stats_Rushing.csv"

@st.cache
def load_rushing_stats(path):
    return pd.read_csv(path)

if rushing_path.exists():
    rushing_df = load_rushing_stats(rushing_path)
else:
    st.warning("Career_Stats_Rushing.csv not found in /data folder.")
    rushing_df = None

# ----------------------------------------------------------
# LOAD RECEIVING CAREER STATS
# ----------------------------------------------------------
receiving_path = Path("data") / "Career_Stats_Receiving.csv"

@st.cache
def load_receiving_stats(path):
    return pd.read_csv(path)

if receiving_path.exists():
    receiving_df = load_receiving_stats(receiving_path)
else:
    st.warning("Career_Stats_Receiving.csv not found in /data folder.")
    receiving_df = None

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

st.write("---")

# ----------------------------------------------------------
# PASSING CAREER STATISTICS
# ----------------------------------------------------------
def show_passing_stats():
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
# RUSHING CAREER STATISTICS
# ----------------------------------------------------------
def show_rushing_stats():
    if rushing_df is not None:

        player_id = player_data["Player Id"]
        rushing_player = rushing_df[rushing_df["Player Id"] == player_id]

        # Clean commas from numeric fields (e.g., "4,523")
        if "Rushing Yards" in rushing_player.columns:
            rushing_player["Rushing Yards"] = rushing_player["Rushing Yards"].astype(str).str.replace(",", "")

        # Clean "T" suffix from Longest Rushing Run (e.g., "65T")
        if "Longest Rushing Run" in rushing_player.columns:
            rushing_player["Longest Rushing Run"] = rushing_player["Longest Rushing Run"].astype(str).str.replace("T", "", regex=False)

        # Numeric columns we care about (will filter by existence)
        rush_numeric_cols_list = [
            "Games Played",
            "Rushing Attempts",
            "Rushing Attempts Per Game",
            "Rushing Yards",
            "Yards Per Carry",
            "Rushing Yards Per Game",
            "Rushing TDs",
            "Longest Rushing Run",
            "Rushing First Downs",
            "Percentage of Rushing First Downs",
            "Rushing More Than 20 Yards",
            "Rushing More Than 40 Yards",
            "Fumbles"
        ]

        # Convert to numeric where possible
        for col in rush_numeric_cols_list:
            if col in rushing_player.columns:
                rushing_player[col] = pd.to_numeric(rushing_player[col], errors="coerce")

        if rushing_player.empty:
            st.markdown("")
        else:
            st.markdown("### Career Rushing Summary")
            if "Player Id" in rushing_player.columns:
                rush_display_df = rushing_player.drop(columns=["Player Id", "Name", "Position"])
            else:
                rush_display_df = rushing_player

            # Clean formatting for display (no trailing .0)
            rush_format_dict = {}

            # Integer-like fields – force them to show as whole numbers
            rush_int_like_cols = [
                "Games Played",
                "Rushing Attempts",
                "Rushing Yards",
                "Rushing TDs",
                "Longest Rushing Run",
                "Rushing First Downs",
                "Rushing More Than 20 Yards",
                "Rushing More Than 40 Yards",
                "Fumbles"
            ]

            for col in rush_int_like_cols:
                if col in rush_display_df.columns:
                    rush_format_dict[col] = "{:.0f}"

            # Percentage-like fields – show 1 decimal
            rush_percent_like_cols = [
                "Percentage of Rushing First Downs"
            ]

            for col in rush_percent_like_cols:
                if col in rush_display_df.columns:
                    rush_format_dict[col] = "{:.1f}"

            # Yardage / averages – show 1 decimal if needed
            rush_one_decimal_cols = [
                "Rushing Attempts Per Game",
                "Yards Per Carry",
                "Rushing Yards Per Game"
            ]

            for col in rush_one_decimal_cols:
                if col in rush_display_df.columns:
                    rush_format_dict[col] = "{:.1f}"

            # Make a copy to format for display
            rush_formatted_df = rush_display_df.copy()

            # Integer-like fields – no decimals
            rush_int_like_cols = [
                "Games Played",
                "Rushing Attempts",
                "Rushing Yards",
                "Rushing TDs",
                "Longest Rushing Run",
                "Rushing First Downs",
                "Rushing More Than 20 Yards",
                "Rushing More Than 40 Yards",
                "Fumbles"
            ]

            for col in rush_int_like_cols:
                if col in rush_formatted_df.columns:
                    rush_formatted_df[col] = pd.to_numeric(rush_formatted_df[col], errors="coerce").round(0).astype("Int64").astype(str)

            # Percentage-like fields – show 1 decimal
            rush_percent_like_cols = [
                "Percentage of Rushing First Downs"
            ]

            for col in rush_percent_like_cols:
                if col in rush_display_df.columns:
                    rush_formatted_df[col] = pd.to_numeric(rush_formatted_df[col], errors="coerce").round(1).astype(str)

            # Yardage / averages – show 1 decimal if needed
            rush_one_decimal_cols = [
                "Rushing Attempts Per Game",
                "Yards Per Carry",
                "Rushing Yards Per Game"
            ]

            for col in rush_one_decimal_cols:
                if col in rush_display_df.columns:
                    rush_formatted_df[col] = pd.to_numeric(rush_formatted_df[col], errors="coerce").round(1).astype(str)

            st.dataframe(rush_formatted_df)

            # ---------------- Plot rushing metric over time ----------------
            if "Year" in rushing_player.columns:
                rushing_player = rushing_player.sort_values(by="Year")

                
            # Numeric columns we care about (will filter by existence)
            rush_numeric_cols = [
                "Games Played",
                "Rushing Attempts",
                "Rushing Attempts Per Game",
                "Rushing Yards",
                "Yards Per Carry",
                "Rushing Yards Per Game",
                "Rushing TDs",
                "Longest Rushing Run",
                "Rushing First Downs",
                "Percentage of Rushing First Downs",
                "Rushing More Than 20 Yards",
                "Rushing More Than 40 Yards",
                "Fumbles"
            ]

            rush_numeric_cols_list = [c for c in rush_numeric_cols_list if c in rushing_player.columns]

            st.markdown("### Plot a Rushing Metric Over Time")

            rush_default_metric = "Rushing Yards" if "Rushing Yards" in rush_numeric_cols_list else rush_numeric_cols_list[0]

            rush_metric = st.selectbox(
                "Select Metric",
                options=rush_numeric_cols,
                index=rush_numeric_cols.index(rush_default_metric)
            )

            # Line chart
            if "Year" in rushing_player.columns:
                rushing_player["Year"] = pd.to_numeric(rushing_player["Year"], errors="coerce").fillna(0).astype(int)


                # Sort by Year
                plot_df = rushing_player.sort_values(by="Year")

                # Ensure metric column is numeric
                plot_df[rush_metric] = pd.to_numeric(plot_df[rush_metric], errors="coerce")
                rush_metric_values = plot_df[rush_metric].dropna()

                if rush_metric_values.empty:
                    st.info("No valid numeric values available for this metric.")
                else:
                    y_min = rush_metric_values.min()
                    y_max = rush_metric_values.max()

                    if y_max == y_min:
                        # Flat line – just plot without padding
                        padded_series = plot_df.set_index("Year")[rush_metric]
                    else:
                        padding = (y_max - y_min) * 0.15  # 15% padding
                        y_min_adj = max(0, y_min - padding)
                        y_max_adj = y_max + padding
                        padded_series = plot_df.set_index("Year")[rush_metric].clip(y_min_adj, y_max_adj)

                    st.line_chart(
                        padded_series,
                        use_container_width=True
                    )

            else:
                st.info("No Year column available for line chart.")

# ----------------------------------------------------------
# RECEIVING CAREER STATISTICS
# ----------------------------------------------------------
def show_receiving_stats():
    if receiving_df is not None:

        player_id = player_data["Player Id"]
        receiving_player = receiving_df[receiving_df["Player Id"] == player_id].copy()

        if receiving_player.empty:
            st.info("No receiving stats available for this player.")
            return

        # Clean commas in yardage (e.g., "1,234")
        if "Receiving Yards" in receiving_player.columns:
            receiving_player["Receiving Yards"] = (receiving_player["Receiving Yards"].astype(str).str.replace(",", "", regex=False))

        # Clean "T" suffix on longest reception (e.g., "80T")
        if "Longest Reception" in receiving_player.columns:
            receiving_player["Longest Reception"] = (receiving_player["Longest Reception"].astype(str).str.replace("T", "", regex=False))

        # Numeric candidates (we'll only use the ones that actually exist)
        recv_numeric_candidates = [
            "Games Played",
            "Receptions",
            "Receiving Yards",
            "Yards Per Reception",
            "Yards Per Game",
            "Receiving TDs",
            "Longest Reception",
            "First Downs Receptions",
            "Receptions More Than 20 Yards",
            "Receptions More Than 40 Yards",
            "Fumbles",
        ]

        for col in recv_numeric_candidates:
            if col in receiving_player.columns:
                receiving_player[col] = pd.to_numeric(receiving_player[col], errors="coerce")

        # Drop ID-ish columns for display
        recv_display_df = receiving_player.copy()
        for c in ["Player Id", "Name", "Position"]:
            if c in recv_display_df.columns:
                recv_display_df = recv_display_df.drop(columns=[c])

        st.markdown("### Career Receiving Summary")

        # ---------- FORMAT RECEIVING TABLE ----------
        formatted_recv_df = recv_display_df.copy()

        # Integer-like fields
        recv_int_fields = [
            "Games Played",
            "Receptions",
            "Receiving Yards",
            "Receiving TDs",
            "Longest Reception",
            "First Downs Receptions",
            "Receptions More Than 20 Yards",
            "Receptions More Than 40 Yards",
            "Fumbles",
        ]

        for col in recv_int_fields:
            if col in formatted_recv_df.columns:
                formatted_recv_df[col] = (
                    pd.to_numeric(formatted_recv_df[col], errors="coerce")
                    .round(0)
                    .astype("Int64")
                    .astype(str)
                )

        # One-decimal fields (remove trailing .0)
        recv_one_decimal_fields = [
            "Yards Per Reception",
            "Yards Per Game",
        ]

        for col in recv_one_decimal_fields:
            if col in formatted_recv_df.columns:
                formatted_recv_df[col] = (
                    pd.to_numeric(formatted_recv_df[col], errors="coerce")
                    .round(1)
                    .apply(lambda x: str(x).rstrip("0").rstrip(".") if "." in str(x) else str(x))
                )

        # Percentage-like fields
        recv_percent_fields = [
            "Percentage of First Downs"
        ]

        for col in recv_percent_fields:
            if col in formatted_recv_df.columns:
                formatted_recv_df[col] = (
                    pd.to_numeric(formatted_recv_df[col], errors="coerce")
                    .round(1)
                    .astype(str)
                )

        st.dataframe(formatted_recv_df)


        # ---------- Plot receiving metric over time ----------
        if "Year" in receiving_player.columns:
            receiving_player["Year"] = pd.to_numeric(
                receiving_player["Year"], errors="coerce"
            ).astype("Int64")

            plot_recv_df = receiving_player.dropna(subset=["Year"]).copy()
            plot_recv_df = plot_recv_df.sort_values(by="Year")

            recv_numeric_cols = [
                c for c in recv_numeric_candidates if c in plot_recv_df.columns
            ]

            if not recv_numeric_cols:
                st.info("No numeric receiving metrics found to plot.")
                return

            st.markdown("### Plot a Receiving Metric Over Time")

            default_recv_metric = (
                "Receiving Yards"
                if "Receiving Yards" in recv_numeric_cols
                else recv_numeric_cols[0]
            )

            recv_metric = st.selectbox(
                "Select Receiving Metric",
                options=recv_numeric_cols,
                index=recv_numeric_cols.index(default_recv_metric),
            )

            plot_recv_df[recv_metric] = pd.to_numeric(
                plot_recv_df[recv_metric], errors="coerce"
            )
            metric_values = plot_recv_df[recv_metric].dropna()

            if metric_values.empty:
                st.info("No valid numeric values available for this receiving metric.")
                return

            y_min = metric_values.min()
            y_max = metric_values.max()

            if y_max == y_min:
                padded_series = plot_recv_df.set_index("Year")[recv_metric]
            else:
                padding = (y_max - y_min) * 0.15
                y_min_adj = max(0, y_min - padding)
                y_max_adj = y_max + padding
                padded_series = (
                    plot_recv_df.set_index("Year")[recv_metric]
                    .clip(y_min_adj, y_max_adj)
                )

            st.line_chart(padded_series, use_container_width=True)

        else:
            st.info("No 'Year' column in receiving stats to plot over time.")
    else:
        st.info("Receiving stats file not loaded.")


# ----------------------------------------------------------
# POSITION-BASED STAT ORDERING
# ----------------------------------------------------------
pos = str(player_data["Position"]).upper()

if pos.startswith("QB"):
    # Quarterbacks → Passing, then Rushing, then Receiving (if any)
    show_passing_stats()
    show_rushing_stats()
    show_receiving_stats()

elif pos.startswith(("RB", "HB", "FB")):
    # Backs → Rushing, then Receiving, then Passing
    show_rushing_stats()
    show_receiving_stats()
    show_passing_stats()

elif pos.startswith(("WR", "TE")):
    # Receivers → Receiving, then Rushing (jet sweeps, etc.), then Passing (trick plays)
    show_receiving_stats()
    show_rushing_stats()
    show_passing_stats()

else:
    # Other positions → Passing, Rushing, Receiving as fallback
    show_passing_stats()
    show_rushing_stats()
    show_receiving_stats()


# ----------------------------------------------------------
# RAW COLUMN DEBUGGER
# ----------------------------------------------------------
with st.expander("Show raw basic stats columns"):
    st.write(list(df.columns))

with st.expander("Show raw passing stats columns"):
    st.write(list(passing_df.columns))

with st.expander("Show raw rushing stats columns"):
    st.write(list(rushing_df.columns))

with st.expander("Show raw receiving stats columns"):
    st.write(list(receiving_df.columns))