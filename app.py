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
# GENERIC CSV LOADER
# ----------------------------------------------------------
@st.cache
def load_csv(path: Path):
    return pd.read_csv(path)


# ----------------------------------------------------------
# LOAD CAREER STATS FILES
# ----------------------------------------------------------
passing_path = Path("data") / "Career_Stats_Passing.csv"
passing_df = load_csv(passing_path) if passing_path.exists() else None
if passing_df is None:
    st.warning("Career_Stats_Passing.csv not found in /data folder.")

rushing_path = Path("data") / "Career_Stats_Rushing.csv"
rushing_df = load_csv(rushing_path) if rushing_path.exists() else None
if rushing_df is None:
    st.warning("Career_Stats_Rushing.csv not found in /data folder.")

receiving_path = Path("data") / "Career_Stats_Receiving.csv"
receiving_df = load_csv(receiving_path) if receiving_path.exists() else None
if receiving_df is None:
    st.warning("Career_Stats_Receiving.csv not found in /data folder.")

defensive_path = Path("data") / "Career_Stats_Defensive.csv"
defensive_df = load_csv(defensive_path) if defensive_path.exists() else None

fg_path = Path("data") / "Career_Stats_Field_Goal_Kickers.csv"
fg_df = load_csv(fg_path) if fg_path.exists() else None

fumbles_path = Path("data") / "Career_Stats_Fumbles.csv"
fumbles_df = load_csv(fumbles_path) if fumbles_path.exists() else None

kick_return_path = Path("data") / "Career_Stats_Kick_Return.csv"
kick_return_df = load_csv(kick_return_path) if kick_return_path.exists() else None

kickoff_path = Path("data") / "Career_Stats_Kickoff.csv"
kickoff_df = load_csv(kickoff_path) if kickoff_path.exists() else None

ol_path = Path("data") / "Career_Stats_Offensive_Line.csv"
ol_df = load_csv(ol_path) if ol_path.exists() else None

punt_return_path = Path("data") / "Career_Stats_Punt_Return.csv"
punt_return_df = load_csv(punt_return_path) if punt_return_path.exists() else None

punting_path = Path("data") / "Career_Stats_Punting.csv"
punting_df = load_csv(punting_path) if punting_path.exists() else None


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
# GENERIC CAREER STATS SECTION
# ----------------------------------------------------------
def show_generic_career_stats(
    title,
    source_df,
    player_data_row,
    id_col="Player Id",
    year_col="Year",
    drop_cols=None,
    int_fields=None,
    one_decimal_fields=None,
    percent_fields=None,
    key_prefix="",
    default_metric_name=None,
):
    """
    Generic section:
      - filters source_df by Player Id
      - converts all NaN/NA in value columns to 0
      - shows a formatted table
      - plots a numeric metric over time (Year) with padding
      - dropdown metrics are ONLY numeric columns shown in the table
    """
    if source_df is None:
        return

    if id_col not in source_df.columns:
        return

    player_id = player_data_row[id_col]
    player_rows = source_df[source_df[id_col] == player_id].copy()

    if player_rows.empty:
        return  # no stats for this player in this category

    st.write("---")
    st.subheader(title)

    # Clean obvious numeric-like text in non-ID/Name/Position columns
    for col in player_rows.columns:
        if col not in [id_col, "Name", "Position", "Team", year_col]:
            if player_rows[col].dtype == object:
                player_rows[col] = (
                    player_rows[col]
                    .astype(str)
                    .str.replace(",", "", regex=False)   # remove commas
                    .str.replace("T", "", regex=False)   # remove "T" suffix
                )

    # Identify "value" columns (everything except IDs/labels/year)
    value_cols = [
        c for c in player_rows.columns
        if c not in [id_col, "Name", "Position", "Team", year_col]
    ]

    # Convert value columns to numeric and fill NaN with 0
    for col in value_cols:
        player_rows[col] = pd.to_numeric(player_rows[col], errors="coerce").fillna(0)

    # Convert year column to numeric (but don't fill NaN with 0 here yet)
    if year_col in player_rows.columns:
        player_rows[year_col] = pd.to_numeric(
            player_rows[year_col], errors="coerce"
        ).astype("Int64")

    # If after conversion, the player still somehow has no non-zero data, we can bail quietly
    if value_cols:
        # If every row and every value col is 0, it's still "valid", but it's all zeros now
        pass  # you said "delete or make it all zeros" — we've chosen "make it all zeros"

    # ---------- TABLE DISPLAY ----------
    display_df = player_rows.copy()

    cols_to_drop = [c for c in [id_col, "Name", "Position"] if c in display_df.columns]
    if drop_cols:
        cols_to_drop.extend(drop_cols)
    cols_to_drop = list(set(cols_to_drop))  # dedupe

    display_df = display_df.drop(columns=cols_to_drop, errors="ignore")

    # Also fill NaNs with 0 in the display table to avoid any <NA> visuals
    for col in display_df.columns:
        if col != year_col:
            if pd.api.types.is_numeric_dtype(display_df[col]):
                display_df[col] = display_df[col].fillna(0)

    # Formatting
    formatted_df = display_df.copy()

    # Integer-like fields
    if int_fields:
        for col in int_fields:
            if col in formatted_df.columns:
                formatted_df[col] = (
                    pd.to_numeric(formatted_df[col], errors="coerce")
                    .fillna(0)
                    .round(0)
                    .astype("Int64")
                    .astype(str)
                )

    # One-decimal fields (strip trailing .0)
    if one_decimal_fields:
        for col in one_decimal_fields:
            if col in formatted_df.columns:
                formatted_df[col] = (
                    pd.to_numeric(formatted_df[col], errors="coerce")
                    .fillna(0)
                    .round(1)
                    .apply(
                        lambda x: (
                            str(x).rstrip("0").rstrip(".")
                            if "." in str(x)
                            else str(x)
                        )
                    )
                )

    # Percentage-like fields: keep 1 decimal, including trailing 0
    if percent_fields:
        for col in percent_fields:
            if col in formatted_df.columns:
                formatted_df[col] = (
                    pd.to_numeric(formatted_df[col], errors="coerce")
                    .fillna(0)
                    .round(1)
                    .astype(str)
                )

    st.markdown("#### Career Summary")
    st.dataframe(formatted_df)

    # ---------- PLOT OVER TIME ----------
    if year_col not in player_rows.columns:
        st.info(f"No '{year_col}' column available to plot over time.")
        return

    plot_df = player_rows.dropna(subset=[year_col]).copy()
    plot_df = plot_df.sort_values(by=year_col)

    if plot_df.empty:
        st.info("No rows with a valid year to plot.")
        return

    # Dropdown options: ONLY numeric columns that are in the displayed table (not Year)
    numeric_cols = [
        c for c in display_df.columns
        if c != year_col and pd.api.types.is_numeric_dtype(display_df[c])
    ]

    if not numeric_cols:
        st.info("No numeric columns found to plot for this category.")
        return

    st.markdown("#### Plot a Metric Over Time")

    # Choose default metric: prefer default_metric_name if valid,
    # otherwise fall back to the first numeric column.
    if default_metric_name and default_metric_name in numeric_cols:
        default_metric = default_metric_name
    else:
        default_metric = numeric_cols[0]

    metric = st.selectbox(
        "Select Metric",
        options=numeric_cols,
        index=numeric_cols.index(default_metric),
        key=f"{key_prefix}_metric_select",
    )


    # Use numeric data from plot_df (already NaNs→0 in value columns)
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce").fillna(0)
    metric_values = plot_df[metric]

    if metric_values.empty:
        st.info("No valid numeric values available for this metric.")
        return

    y_min = metric_values.min()
    y_max = metric_values.max()

    if y_max == y_min:
        padded_series = plot_df.set_index(year_col)[metric]
    else:
        padding = (y_max - y_min) * 0.15
        y_min_adj = y_min - padding
        y_max_adj = y_max + padding
        padded_series = plot_df.set_index(year_col)[metric].clip(y_min_adj, y_max_adj)

    st.line_chart(padded_series, use_container_width=True)


# ----------------------------------------------------------
# SPECIALIZED WRAPPERS: PASSING
# ----------------------------------------------------------

def show_passing_stats():
    if passing_df is None:
        return

    passing_local = passing_df.copy()
    if "Sacks" in passing_local.columns:
        passing_local = passing_local.rename(columns={"Sacks": "Times Sacked"})

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

    one_decimal_fields = [
        "Passing Yards Per Attempt",
        "Passing Yards Per Game",
        "Pass Attempts Per Game",
        "Passer Rating",
    ]

    percent_fields = [
        "Completion Percentage",
        "Percentage of TDs per Attempts",
        "Int Rate",
    ]

    show_generic_career_stats(
        title="Career Passing Statistics",
        source_df=passing_local,
        player_data_row=player_data,
        int_fields=int_fields,
        one_decimal_fields=one_decimal_fields,
        percent_fields=percent_fields,
        key_prefix="passing",
        default_metric_name="Passing Yards",
    )

# ----------------------------------------------------------
# SPECIALIZED WRAPPERS: RUSHING
# ----------------------------------------------------------

def show_rushing_stats():
    if rushing_df is None:
        return

    int_fields = [
        "Games Played",
        "Rushing Attempts",
        "Rushing Yards",
        "Rushing TDs",
        "Longest Rushing Run",
        "Rushing First Downs",
        "Rushing More Than 20 Yards",
        "Rushing More Than 40 Yards",
        "Fumbles",
    ]

    one_decimal_fields = [
        "Rushing Attempts Per Game",
        "Yards Per Carry",
        "Rushing Yards Per Game",
    ]

    percent_fields = [
        "Percentage of Rushing First Downs",
    ]

    show_generic_career_stats(
        title="Career Rushing Statistics",
        source_df=rushing_df,
        player_data_row=player_data,
        int_fields=int_fields,
        one_decimal_fields=one_decimal_fields,
        percent_fields=percent_fields,
        key_prefix="rushing",
        default_metric_name="Rushing Yards",
    )

# ----------------------------------------------------------
# SPECIALIZED WRAPPERS: RECEIVING
# ----------------------------------------------------------

def show_receiving_stats():
    if receiving_df is None:
        return

    int_fields = [
        "Games Played",
        "Receptions",
        "Receiving Yards",
        "Receiving TDs",
        "Longest Reception",
        "Receptions Longer than 20 Yards",
        "Receptions Longer than 40 Yards",
        "First Down Receptions",
        "Fumbles",
    ]

    one_decimal_fields = [
        "Yards Per Reception",
        "Yards Per Game",
    ]

    show_generic_career_stats(
        title="Career Receiving Statistics",
        source_df=receiving_df,
        player_data_row=player_data,
        int_fields=int_fields,
        one_decimal_fields=one_decimal_fields,
        percent_fields=None,
        key_prefix="receiving",
        default_metric_name="Receiving Yards",
    )

# ----------------------------------------------------------
# SPECIALIZED WRAPPERS: DEFENSE
# ----------------------------------------------------------

def show_defensive_stats():
    if defensive_df is None:
        return

    int_fields = [
        "Games Played",
        "Total Tackles",
        "Solo Tackles",
        "Assisted Tackles",
        "Sacks",
        "Safties",
        "Passes Defended",
        "Ints",
        "Ints for TDs",
        "Int Yards",
        "Longest Int Return",
    ]

    one_decimal_fields = [
        "Yards Per Int",
    ]

    show_generic_career_stats(
        title="Defensive Career Statistics",
        source_df=defensive_df,
        player_data_row=player_data,
        int_fields=int_fields,
        one_decimal_fields=one_decimal_fields,
        percent_fields=None,
        key_prefix="defensive",
        default_metric_name="Total Tackles",
    )

# ----------------------------------------------------------
# SPECIALIZED WRAPPERS: KICKER
# ----------------------------------------------------------

def show_kicker_stats():
    if fg_df is None:
        return

    int_fields = [
        "Games Played",
        "Kicks Blocked",
        "Longest FG Made",
        "FGs Made",
        "FGs Attempted",
        "FGs Made 20-29 Yards",
        "FGs Attempted 20-29 Yards",
        "FGs Made 30-39 Yards",
        "FGs Attempted 30-39 Yards",
        "FGs Made 40-49 Yards",
        "FGs Attempted 40-49 Yards",
        "FGs Made 50+ Yards",
        "FGs Attempted 50+ Yards",
        "Extra Points Attempted",
        "Extra Points Made",
        "Extra Points Blocked",
    ]

    percent_fields = [
        "FG Percentage",
        "FG Percentage 20-29 Yards",
        "FG Percentage 30-39 Yards",
        "FG Percentage 40-49 Yards",
        "FG Percentage 50+ Yards",
        "Percentage of Extra Points Made",
    ]

    show_generic_career_stats(
        title="Field Goal Kicking Career Statistics",
        source_df=fg_df,
        player_data_row=player_data,
        int_fields=int_fields,
        one_decimal_fields=None,
        percent_fields=percent_fields,
        key_prefix="fg",
        default_metric_name="FGs Made",
    )

# ----------------------------------------------------------
# SPECIALIZED WRAPPERS: KICK RETURN
# ----------------------------------------------------------

def show_kick_return_stats():
    if kick_return_df is None:
        return

    int_fields = [
        "Games Played",
        "Returns",
        "Yards Returned",
        "Longest Return",
        "Returns for TDs",
        "Returns Longer than 20 Yards",
        "Returns Longer than 40 Yards",
        "Fair Catches",
        "Fumbles",
    ]

    one_decimal_fields = [
        "Yards Per Return",
    ]

    show_generic_career_stats(
        title="Kick Return Career Statistics",
        source_df=kick_return_df,
        player_data_row=player_data,
        int_fields=int_fields,
        one_decimal_fields=one_decimal_fields,
        percent_fields=None,
        key_prefix="kick_return",
        default_metric_name="Yards Returned",
    )

# ----------------------------------------------------------
# SPECIALIZED WRAPPERS: KICKOFF
# ----------------------------------------------------------

def show_kickoff_stats():
    if kickoff_df is None:
        return

    int_fields = [
        "Games Played",
        "Kickoffs",
        "Kickoff Yards",
        "Out of Bounds Kickoffs",
        "Touchbacks",
        "Kickoffs Returned",
        "Kickoffs Resulting in TDs",
        "On Sides Kicks",
        "On Sides Kicks Returned",
    ]

    one_decimal_fields = [
        "Yards Per Kickoff",
        "Average Returned Yards",
    ]

    percent_fields = [
        "Touchback Percentage",
    ]

    show_generic_career_stats(
        title="Kickoff Career Statistics",
        source_df=kickoff_df,
        player_data_row=player_data,
        int_fields=int_fields,
        one_decimal_fields=one_decimal_fields,
        percent_fields=percent_fields,
        key_prefix="kickoff",
        default_metric_name="Kickoff Yards",
    )

# ----------------------------------------------------------
# SPECIALIZED WRAPPERS: OFFENSIIVE LINE
# ----------------------------------------------------------

def show_offensive_line_stats():
    if ol_df is None:
        return

    int_fields = [
        "Games Played",
        "Games Started",
    ]

    show_generic_career_stats(
        title="Offensive Line Career Statistics",
        source_df=ol_df,
        player_data_row=player_data,
        int_fields=int_fields,
        one_decimal_fields=None,
        percent_fields=None,
        key_prefix="offensive_line",
    )

# ----------------------------------------------------------
# SPECIALIZED WRAPPERS: PUNT RETURN
# ----------------------------------------------------------

def show_punt_return_stats():
    if punt_return_df is None:
        return

    int_fields = [
        "Games Played",
        "Returns",
        "Yards Returned",
        "Longest Return",
        "Returns for TDs",
        "Returns Longer than 20 Yards",
        "Returns Longer than 40 Yards",
        "Fair Catches",
        "Fumbles",
    ]

    one_decimal_fields = [
        "Yards Per Return",
    ]

    show_generic_career_stats(
        title="Punt Return Career Statistics",
        source_df=punt_return_df,
        player_data_row=player_data,
        int_fields=int_fields,
        one_decimal_fields=one_decimal_fields,
        percent_fields=None,
        key_prefix="punt_return",
        default_metric_name="Yards Returned",
    )

# ----------------------------------------------------------
# SPECIALIZED WRAPPERS: PUNTING
# ----------------------------------------------------------

def show_punting_stats():
    if punting_df is None:
        return

    int_fields = [
        "Games Played",
        "Punts",
        "Gross Punting Yards",
        "Net Punting Yards",
        "Longest Punt",
        "Punts Blocked",
        "Out of Bounds Punts",
        "Downed Punts",
        "Punts Inside 20 Yard Line",
        "Touchbacks",
        "Fair Catches",
        "Punts Returned",
        "Yards Returned on Punts",
        "TDs Returned on Punt",
    ]

    one_decimal_fields = [
        "Gross Punting Average",
        "Net Punting Average",
    ]

    show_generic_career_stats(
        title="Punting Career Statistics",
        source_df=punting_df,
        player_data_row=player_data,
        int_fields=int_fields,
        one_decimal_fields=one_decimal_fields,
        percent_fields=None,
        key_prefix="punting",
        default_metric_name="Gross Punting Yards",
    )

# ----------------------------------------------------------
# SPECIALIZED WRAPPERS: OFFHAND DEFENSE
# ----------------------------------------------------------

def show_off_defensive_stats():
    if defensive_df is None:
        return

    int_fields = [
        "Games Played",
        "Total Tackles",
        "Solo Tackles",
        "Assisted Tackles"
    ]

    show_generic_career_stats(
        title="Defensive Career Statistics",
        source_df=defensive_df,
        player_data_row=player_data,
        drop_cols=[
            "Sacks",
            "Safties",
            "Ints",
            "Ints for TDs",
            "Int Yards",
            "Yards Per Int",
            "Longest Int Return"
                   ],
        int_fields=int_fields,
        percent_fields=None,
        key_prefix="defensive",
        default_metric_name="Total Tackles",
    )

# ----------------------------------------------------------
# SPECIALIZED WRAPPERS: OFFHAND PASSING
# ----------------------------------------------------------

def show_off_passing_stats():
    if passing_df is None:
        return

    int_fields = [
        "Games Played",
        "Passes Attempted",
        "Passes Completed",
        "TD Passes",
        "Ints",
        "Longest Pass"
    ]

    percent_fields = [
        "Completion Percentage",
    ]

    show_generic_career_stats(
        title="Career Passing Statistics",
        source_df=passing_df,
        player_data_row=player_data,
        drop_cols=[
            "Pass Attempts Per Game",
            "Passing Yards Per Attempt",
            "Passing Yards Per Game",
            "Passer Rating",
            "Percentage of TDs per Attempts",
            "Int Rate",
            "Passes Longer than 20 Yards",
            "Passes Longer than 40 Yards",
            "Sacks",
            "Sacked Yards Lost"
        ],
        int_fields=int_fields,
        percent_fields=percent_fields,
        key_prefix="passing",
        default_metric_name="Passing Yards",
    )

# ----------------------------------------------------------
# SPECIALIZED WRAPPERS: OFFHAND RUSHING
# ----------------------------------------------------------

def show_off_rushing_stats():
    if rushing_df is None:
        return

    int_fields = [
        "Games Played",
        "Rushing Attempts",
        "Rushing Yards",
        "Rushing TDs",
        "Longest Rushing Run",
        "Rushing First Downs",
        "Fumbles"
    ]

    one_decimal_fields = [
        "Yards Per Carry"
    ]

    show_generic_career_stats(
        title="Career Rushing Statistics",
        source_df=rushing_df,
        player_data_row=player_data,
        drop_cols=[
            "Rushing Attempts Per Game",
            "Rushing Yards Per Game",
            "Percentage of Rushing First Downs",
            "Rushing More Than 20 Yards",
            "Rushing More Than 40 Yards"
        ], 
        int_fields=int_fields,
        one_decimal_fields=one_decimal_fields,
        key_prefix="rushing",
        default_metric_name="Rushing Yards",
    )

# ----------------------------------------------------------
# SPECIALIZED WRAPPERS: OFFHAND RECEIVING
# ----------------------------------------------------------

def show_off_receiving_stats():
    if receiving_df is None:
        return

    int_fields = [
        "Games Played",
        "Receptions",
        "Receiving Yards",
        "Receiving TDs",
        "Longest Reception",
        "First Down Receptions",
        "Fumbles"
    ]

    one_decimal_fields = [
        "Yards Per Reception"
    ]

    show_generic_career_stats(
        title="Career Receiving Statistics",
        source_df=receiving_df,
        player_data_row=player_data,
        drop_cols=[
            "Receptions Longer than 20 Yards",
            "Receptions Longer than 40 Yards",
            "Yards Per Game"
        ], 
        int_fields=int_fields,
        one_decimal_fields=one_decimal_fields,
        percent_fields=None,
        key_prefix="receiving",
        default_metric_name="Receiving Yards",
    )

# ----------------------------------------------------------
# POSITION-BASED STAT ORDERING
# ----------------------------------------------------------
pos = str(player_data["Position"]).upper()

if pos.startswith("QB"):
    # Quarterbacks → Passing, then Rushing, then Receiving
    show_passing_stats()
    show_rushing_stats()
    show_off_receiving_stats()
    show_off_defensive_stats()

elif pos.startswith(("RB", "HB", "FB")):
    # Backs → Rushing, then Receiving, then Passing
    show_rushing_stats()
    show_receiving_stats()
    show_off_passing_stats()
    show_kick_return_stats()
    show_punt_return_stats()
    show_off_defensive_stats()

elif pos.startswith(("WR", "TE")):
    # Receivers → Receiving, then Rushing, then Passing
    show_receiving_stats()
    show_off_rushing_stats()
    show_off_passing_stats()
    show_kick_return_stats()
    show_punt_return_stats()
    show_off_defensive_stats()

elif pos.startswith(("G", "OG", "OL", "OT", "T", "LS", "C")):
    # Everyone else → default order, but still see the big 3 if they have data
    show_offensive_line_stats()
    show_off_rushing_stats()
    show_off_receiving_stats()
    show_off_defensive_stats()
    
elif pos.startswith(("K")):
    show_kicker_stats()
    show_kickoff_stats()
    show_punting_stats()
    show_off_defensive_stats()
    show_off_passing_stats()

elif pos.startswith(("P")):
    show_punting_stats()
    show_off_rushing_stats()
    show_kicker_stats()
    show_kickoff_stats()
    show_off_defensive_stats()
    show_off_passing_stats()

elif pos.startswith("CB", "DB", "FS", "SS", "SAF"):
    show_defensive_stats()
    show_kick_return_stats()
    show_punt_return_stats()
    show_off_receiving_stats()

else:
    show_defensive_stats()

# ----------------------------------------------------------
# RAW COLUMN DEBUGGERS
# ----------------------------------------------------------
with st.expander("Show raw basic stats columns"):
    st.write(list(df.columns))

if passing_df is not None:
    with st.expander("Show raw passing stats columns"):
        st.write(list(passing_df.columns))

if rushing_df is not None:
    with st.expander("Show raw rushing stats columns"):
        st.write(list(rushing_df.columns))

if receiving_df is not None:
    with st.expander("Show raw receiving stats columns"):
        st.write(list(receiving_df.columns))

if defensive_df is not None:
    with st.expander("Show raw defensive stats columns"):
        st.write(list(defensive_df.columns))

if fg_df is not None:
    with st.expander("Show raw field goal kicker stats columns"):
        st.write(list(fg_df.columns))

if fumbles_df is not None:
    with st.expander("Show raw fumbles stats columns"):
        st.write(list(fumbles_df.columns))

if kick_return_df is not None:
    with st.expander("Show raw kick return stats columns"):
        st.write(list(kick_return_df.columns))

if kickoff_df is not None:
    with st.expander("Show raw kickoff stats columns"):
        st.write(list(kickoff_df.columns))

if ol_df is not None:
    with st.expander("Show raw offensive line stats columns"):
        st.write(list(ol_df.columns))

if punt_return_df is not None:
    with st.expander("Show raw punt return stats columns"):
        st.write(list(punt_return_df.columns))

if punting_df is not None:
    with st.expander("Show raw punting stats columns"):
        st.write(list(punting_df.columns))
