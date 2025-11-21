import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
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
# HELPERS
# ----------------------------------------------------------

import altair as alt

def plot_hist(series, bins=20):
    """Plot a histogram using Altair so we can control axis ticks."""
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        st.info("No data available for this distribution.")
        return

    df = pd.DataFrame({"Value": series})

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                "Value:Q",
                bin=alt.Bin(maxbins=bins),
                axis=alt.Axis(format="d", tickMinStep=1, title="Value")
            ),
            y=alt.Y("count()", title="Count"),
        )
        .properties(height=300)
    )

    st.altair_chart(chart, use_container_width=True)


def clean_age_column(series):
    """Cleans age values by extracting leading integers and dropping junk."""
    cleaned = []

    for val in series.astype(str):
        # Strip spaces
        v = val.strip()

        # Extract leading digits
        digits = ""
        for ch in v:
            if ch.isdigit():
                digits += ch
            else:
                break

        # Convert to int if valid
        if digits.isdigit():
            age = int(digits)
            if 15 < age < 60:  # realistic NFL ages
                cleaned.append(age)
            else:
                cleaned.append(np.nan)
        else:
            cleaned.append(np.nan)

    return pd.Series(cleaned)

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
# LOAD GAME LOGS FILES
# ----------------------------------------------------------
qb_logs_path = Path("data") / "Game_Logs_Quarterback.csv"
qb_logs_df = load_csv(qb_logs_path) if qb_logs_path.exists() else None

rb_logs_path = Path("data") / "Game_Logs_Runningback.csv"
rb_logs_df = load_csv(rb_logs_path) if rb_logs_path.exists() else None

wrte_logs_path = Path("data") / "Game_Logs_Wide_Receiver_and_Tight_End.csv"
wrte_logs_df = load_csv(wrte_logs_path) if wrte_logs_path.exists() else None

def_line_logs_path = Path("data") / "Game_Logs_Defensive_Lineman.csv"
def_line_logs_df = load_csv(def_line_logs_path) if def_line_logs_path.exists() else None

ol_logs_path = Path("data") / "Game_Logs_Offensive_Line.csv"
ol_logs_df = load_csv(ol_logs_path) if ol_logs_path.exists() else None

kicker_logs_path = Path("data") / "Game_Logs_Kickers.csv"
kicker_logs_df = load_csv(kicker_logs_path) if kicker_logs_path.exists() else None

punter_logs_path = Path("data") / "Game_Logs_Punters.csv"
punter_logs_df = load_csv(punter_logs_path) if punter_logs_path.exists() else None

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
# SUMARRY STATISTICS
# ----------------------------------------------------------

st.subheader("Summary Statistics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Players Shown", len(df_filtered))

# Safely compute Avg Age
if "Age" in df_filtered.columns:
    age_series = clean_age_column(df_filtered["Age"])
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

# ----------------------------------------------------------
# LEAGUE OVERVIEW (AGGREGATES OVER FILTERED DATA)
# ----------------------------------------------------------
st.write("---")
st.subheader("League Overview")

tab_pos, tab_team, tab_dist = st.tabs(
    ["By Position", "By Team", "Distributions"]
)

# ---- Players by Position ----
with tab_pos:
    if "Position" in df_filtered.columns:
        pos_counts = (
            df_filtered["Position"]
            .fillna("Unknown")
            .value_counts()
            .sort_values(ascending=False)
        )
        st.markdown("**Players by Position (filtered)**")

        pos_df = pos_counts.reset_index()
        pos_df.columns = ["Position", "Count"]

        pos_chart = (
            alt.Chart(pos_df)
            .mark_bar()
            .encode(
                x=alt.X("Position:N", sort="-y", title="Position"),
                y=alt.Y("Count:Q", title="Players"),
            )
            .properties(height=300)
        )
        st.altair_chart(pos_chart, use_container_width=True)
    else:
        st.info("No Position column available.")


# ---- Players by Team ----
with tab_team:
    if "Current Team" in df_filtered.columns:
        team_counts = (
            df_filtered["Current Team"]
            .fillna("Unknown")
            .value_counts()
            .sort_values(ascending=False)
        )

        st.markdown("**Players by Team (filtered)**")

        team_df = team_counts.reset_index()
        team_df.columns = ["Team", "Count"]

        team_chart = (
            alt.Chart(team_df)
            .mark_bar()
            .encode(
                x=alt.X("Team:N", sort="-y", title="Team"),
                y=alt.Y("Count:Q", title="Players"),
            )
            .properties(height=300)
        )
        st.altair_chart(team_chart, use_container_width=True)
    else:
        st.info("No Current Team column available.")



# ---- Distributions: Age, Height, Weight ----
with tab_dist:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Age Distribution**")
        if "Age" in df_filtered.columns:
            plot_hist(clean_age_column(df_filtered["Age"]), bins=30)
        else:
            st.info("No Age column available.")

    with c2:
        st.markdown("**Height (in) Distribution**")
        if "Height (inches)" in df_filtered.columns:
            plot_hist(df_filtered["Height (inches)"], bins=20)
        else:
            st.info("No Height (inches) column available.")

    with c3:
        st.markdown("**Weight (lbs) Distribution**")
        if "Weight (lbs)" in df_filtered.columns:
            plot_hist(df_filtered["Weight (lbs)"], bins=40)
        else:
            st.info("No Weight (lbs) column available.")

st.write("---")

# ----------------------------------------------------------
# PLAYER LOOKUP
# ----------------------------------------------------------
st.subheader("Player Lookup")

player_list = sorted(df_filtered["Name"].unique())

# Add a placeholder option at the top
player_options = ["-- Select a player --"] + player_list

selected_player = st.selectbox(
    "Select a Player",
    player_options,
    index=0,  # default to the placeholder
)

# If no real player is selected yet, show a hint and stop the script
if selected_player == "-- Select a player --":
    st.info("Select a player from the dropdown above to view their profile and stats.")
    st.stop()

player_data = df_filtered[df_filtered["Name"] == selected_player].iloc[0]

# ----------------------------------------------------------
# OPTIONAL: PLAYER COMPARISON SELECTION
# ----------------------------------------------------------
compare_mode = st.checkbox("Compare with another player")

player_data_compare = None

if compare_mode:
    # Second player list, excluding the primary player
    compare_candidates = [name for name in player_list if name != selected_player]
    compare_options = ["-- Select player to compare --"] + compare_candidates

    selected_player_compare = st.selectbox(
        "Select Player to Compare",
        compare_options,
        index=0,
        key="compare_player_select",
    )

    if selected_player_compare != "-- Select player to compare --":
        player_data_compare = df_filtered[df_filtered["Name"] == selected_player_compare].iloc[0]

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
# GENERIC GAME LOGS SECTION (with season + rolling averages)
# ----------------------------------------------------------
def show_generic_game_logs(
    title,
    source_df,
    player_data_row,
    id_col="Player Id",
    drop_label_cols=None,
    default_metric_name=None,
    key_prefix="",
):
    """
    Generic game logs viewer:
      - filters source_df by Player Id
      - lets user choose a season (Year) if available
      - cleans '--', '-', '' to NaN then 0 for numeric stats
      - sorts games (Year+Week or Game Date if available)
      - adds a 'Game #' index
      - shows full game log table
      - lets user pick a numeric metric to plot vs Game #
      - optional rolling averages (3-game, 5-game)
    """
    if source_df is None:
        st.info("Game log data is not available for this category.")
        return

    if id_col not in source_df.columns:
        st.info(f"Game logs are missing '{id_col}' column.")
        return

    player_id = player_data_row[id_col]
    logs = source_df[source_df[id_col] == player_id].copy()

    if logs.empty:
        st.info("No game logs found for this player in this category.")
        return

    # Replace common "no data" placeholders
    logs = logs.replace({"--": np.nan, "-": np.nan, "": np.nan})

    # --- Coerce Year/Week if present (for sorting + filtering) ---
    has_year = "Year" in logs.columns
    has_week = "Week" in logs.columns

    if has_year:
        logs["Year"] = pd.to_numeric(logs["Year"], errors="coerce")

    if has_week:
        logs["Week"] = pd.to_numeric(logs["Week"], errors="coerce")

    # --- Season (Year) filter, if we have a Year column ---
    if has_year:
        years = (
            logs["Year"]
            .dropna()
            .astype(int)
            .sort_values()
            .unique()
            .tolist()
        )

        if years:
            year_options = ["All Seasons"] + [str(y) for y in years]
            selected_year = st.selectbox(
                "Season (Year)",
                year_options,
                index=0,
                key=f"{key_prefix}_gamelog_year",
            )

            if selected_year != "All Seasons":
                year_int = int(selected_year)
                logs = logs[logs["Year"] == year_int].copy()

                if logs.empty:
                    st.info(f"No game logs found for the {year_int} season.")
                    return

    # --- Sort games after filtering ---
    if has_year and has_week:
        logs = logs.sort_values(by=["Year", "Week"])
    elif has_year:
        logs = logs.sort_values(by="Year")
    elif "Game Date" in logs.columns:
        logs = logs.sort_values(by="Game Date")

    # Add a simple game index for plotting
    logs.insert(0, "Game #", range(1, len(logs) + 1))

    # Identify label columns we *don't* want treated as metrics
    base_label_cols = [
        id_col,
        "Name",
        "Position",
        "Team",
        "Opponent",
        "Opp",
        "Game Date",
        "Date",
        "Home or Away",
        "Location",
        "Result",
        "Outcome",
        "Season",
        "Week",
        "Year",
    ]
    if drop_label_cols:
        base_label_cols.extend(drop_label_cols)

    label_cols = [c for c in base_label_cols if c in logs.columns]

    # Convert value columns (everything else except labels + Game #) to numeric
    value_cols = [
        c for c in logs.columns
        if c not in label_cols + ["Game #"]
    ]

    for col in value_cols:
        logs[col] = pd.to_numeric(logs[col], errors="coerce")

    # Fill NaNs in numeric with 0 for display
    if value_cols:
        logs[value_cols] = logs[value_cols].fillna(0)

    st.markdown(f"### {title}")
    st.dataframe(logs)

    # Numeric columns for plotting (only within value_cols)
    numeric_cols = [
        c for c in value_cols
        if pd.api.types.is_numeric_dtype(logs[c])
    ]

    if not numeric_cols:
        st.info("No numeric stats available to plot from these game logs.")
        return

    st.markdown("#### Plot a Game Log Metric Over Time")

    if default_metric_name and default_metric_name in numeric_cols:
        default_metric = default_metric_name
    else:
        default_metric = numeric_cols[0]

    metric = st.selectbox(
        "Select metric",
        options=numeric_cols,
        index=numeric_cols.index(default_metric),
        key=f"{key_prefix}_gamelog_metric",
    )

    # Build base plot df
    plot_df = logs[["Game #", metric]].dropna()
    if plot_df.empty:
        st.info("No valid values to plot for this metric.")
        return

    # --- Rolling average selector ---
    rolling_choice = st.selectbox(
        "Rolling average (games)",
        ["None", "3-game", "5-game"],
        index=0,
        key=f"{key_prefix}_gamelog_roll",
    )

    plot_df = plot_df.set_index("Game #")

    if rolling_choice != "None":
        if rolling_choice.startswith("3"):
            window = 3
        else:
            window = 5

        roll_col = f"{metric} (Rolling {window})"
        plot_df[roll_col] = (
            plot_df[metric]
            .rolling(window=window, min_periods=1)
            .mean()
        )

    st.line_chart(plot_df, use_container_width=True)


# ----------------------------------------------------------
# GAME LOG WRAPPERS PER GROUP / POSITION
# ----------------------------------------------------------
def show_qb_game_logs():
    if qb_logs_df is None:
        st.info("Quarterback game logs file is not available.")
        return

    show_generic_game_logs(
        title="Quarterback Game Logs",
        source_df=qb_logs_df,
        player_data_row=player_data,
        default_metric_name="Passing Yards",
        key_prefix="qb_logs",
    )


def show_rb_game_logs():
    if rb_logs_df is None:
        st.info("Running back game logs file is not available.")
        return

    show_generic_game_logs(
        title="Running Back Game Logs",
        source_df=rb_logs_df,
        player_data_row=player_data,
        default_metric_name="Rushing Yards",
        key_prefix="rb_logs",
    )


def show_wrte_game_logs():
    if wrte_logs_df is None:
        st.info("WR/TE game logs file is not available.")
        return

    show_generic_game_logs(
        title="Wide Receiver / Tight End Game Logs",
        source_df=wrte_logs_df,
        player_data_row=player_data,
        default_metric_name="Receiving Yards",
        key_prefix="wrte_logs",
    )


def show_def_line_game_logs():
    if def_line_logs_df is None:
        st.info("Defensive line game logs file is not available.")
        return

    show_generic_game_logs(
        title="Defensive Lineman Game Logs",
        source_df=def_line_logs_df,
        player_data_row=player_data,
        default_metric_name="Total Tackles",
        key_prefix="defline_logs",
    )


def show_ol_game_logs():
    if ol_logs_df is None:
        st.info("Offensive line game logs file is not available.")
        return

    show_generic_game_logs(
        title="Offensive Line Game Logs",
        source_df=ol_logs_df,
        player_data_row=player_data,
        default_metric_name="Games Played",
        key_prefix="ol_logs",
    )


def show_kicker_game_logs():
    if kicker_logs_df is None:
        st.info("Kicker game logs file is not available.")
        return

    show_generic_game_logs(
        title="Kicker Game Logs",
        source_df=kicker_logs_df,
        player_data_row=player_data,
        default_metric_name="FGs Made",  # falls back if not present
        key_prefix="kicker_logs",
    )


def show_punter_game_logs():
    if punter_logs_df is None:
        st.info("Punter game logs file is not available.")
        return

    show_generic_game_logs(
        title="Punter Game Logs",
        source_df=punter_logs_df,
        player_data_row=player_data,
        default_metric_name="Punts",
        key_prefix="punter_logs",
    )


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
# CAREER SUMMARY HELPER
# ----------------------------------------------------------
def get_player_career_summary(player_data_row):
    """Compute key career totals across all stat tables for a player."""
    summary = {}
    player_id = player_data_row["Player Id"]

    # ----- Passing -----
    if passing_df is not None and "Player Id" in passing_df.columns:
        p = passing_df[passing_df["Player Id"] == player_id].copy()
        if not p.empty:
            if "Passing Yards" in p.columns:
                p["Passing Yards"] = (
                    p["Passing Yards"].astype(str).str.replace(",", "", regex=False)
                )
                val = pd.to_numeric(p["Passing Yards"], errors="coerce").fillna(0).sum()
                summary["Passing Yards"] = int(val)

            if "TD Passes" in p.columns:
                val = pd.to_numeric(p["TD Passes"], errors="coerce").fillna(0).sum()
                summary["Passing TDs"] = int(val)

            if "Ints" in p.columns:
                val = pd.to_numeric(p["Ints"], errors="coerce").fillna(0).sum()
                summary["Interceptions Thrown"] = int(val)

    # ----- Rushing -----
    if rushing_df is not None and "Player Id" in rushing_df.columns:
        r = rushing_df[rushing_df["Player Id"] == player_id].copy()
        if not r.empty:
            if "Rushing Yards" in r.columns:
                r["Rushing Yards"] = (
                    r["Rushing Yards"].astype(str).str.replace(",", "", regex=False)
                )
                val = pd.to_numeric(r["Rushing Yards"], errors="coerce").fillna(0).sum()
                summary["Rushing Yards"] = int(val)

            if "Rushing TDs" in r.columns:
                val = pd.to_numeric(r["Rushing TDs"], errors="coerce").fillna(0).sum()
                summary["Rushing TDs"] = int(val)

    # ----- Receiving -----
    if receiving_df is not None and "Player Id" in receiving_df.columns:
        rec = receiving_df[receiving_df["Player Id"] == player_id].copy()
        if not rec.empty:
            if "Receiving Yards" in rec.columns:
                rec["Receiving Yards"] = (
                    rec["Receiving Yards"].astype(str).str.replace(",", "", regex=False)
                )
                val = pd.to_numeric(rec["Receiving Yards"], errors="coerce").fillna(0).sum()
                summary["Receiving Yards"] = int(val)

            if "Receiving TDs" in rec.columns:
                val = pd.to_numeric(rec["Receiving TDs"], errors="coerce").fillna(0).sum()
                summary["Receiving TDs"] = int(val)

    # ----- Defense -----
    if defensive_df is not None and "Player Id" in defensive_df.columns:
        d = defensive_df[defensive_df["Player Id"] == player_id].copy()
        if not d.empty:
            if "Total Tackles" in d.columns:
                val = pd.to_numeric(d["Total Tackles"], errors="coerce").fillna(0).sum()
                summary["Total Tackles"] = int(val)

            if "Sacks" in d.columns:
                val = pd.to_numeric(d["Sacks"], errors="coerce").fillna(0).sum()
                summary["Sacks"] = float(val)

            if "Ints" in d.columns:
                val = pd.to_numeric(d["Ints"], errors="coerce").fillna(0).sum()
                summary["Interceptions"] = int(val)

    # ----- Kick / Punt Returns -----
    if kick_return_df is not None and "Player Id" in kick_return_df.columns:
        kr = kick_return_df[kick_return_df["Player Id"] == player_id].copy()
        if not kr.empty:
            if "Yards Returned" in kr.columns:
                kr["Yards Returned"] = (
                    kr["Yards Returned"].astype(str).str.replace(",", "", regex=False)
                )
                val = pd.to_numeric(kr["Yards Returned"], errors="coerce").fillna(0).sum()
                summary["Kick Return Yards"] = int(val)

            if "Returns for TDs" in kr.columns:
                val = pd.to_numeric(kr["Returns for TDs"], errors="coerce").fillna(0).sum()
                summary["Kick Return TDs"] = int(val)

    if punt_return_df is not None and "Player Id" in punt_return_df.columns:
        pr = punt_return_df[punt_return_df["Player Id"] == player_id].copy()
        if not pr.empty:
            if "Yards Returned" in pr.columns:
                pr["Yards Returned"] = (
                    pr["Yards Returned"].astype(str).str.replace(",", "", regex=False)
                )
                val = pd.to_numeric(pr["Yards Returned"], errors="coerce").fillna(0).sum()
                summary["Punt Return Yards"] = int(val)

            if "Returns for TDs" in pr.columns:
                val = pd.to_numeric(pr["Returns for TDs"], errors="coerce").fillna(0).sum()
                summary["Punt Return TDs"] = int(val)

    # ----- Kicking / Punting -----
    if fg_df is not None and "Player Id" in fg_df.columns:
        k = fg_df[fg_df["Player Id"] == player_id].copy()
        if not k.empty:
            if "FGs Made" in k.columns:
                val = pd.to_numeric(k["FGs Made"], errors="coerce").fillna(0).sum()
                summary["FGs Made"] = int(val)

            if "Extra Points Made" in k.columns:
                val = pd.to_numeric(k["Extra Points Made"], errors="coerce").fillna(0).sum()
                summary["XPs Made"] = int(val)

    if punting_df is not None and "Player Id" in punting_df.columns:
        pnt = punting_df[punting_df["Player Id"] == player_id].copy()
        if not pnt.empty:
            if "Punts" in pnt.columns:
                val = pd.to_numeric(pnt["Punts"], errors="coerce").fillna(0).sum()
                summary["Punts"] = int(val)

            if "Gross Punting Yards" in pnt.columns:
                pnt["Gross Punting Yards"] = (
                    pnt["Gross Punting Yards"].astype(str).str.replace(",", "", regex=False)
                )
                val = pd.to_numeric(pnt["Gross Punting Yards"], errors="coerce").fillna(0).sum()
                summary["Gross Punting Yards"] = int(val)

    # Optional: combined touchdowns (rough version)
    td_total = 0
    for key in ["Passing TDs", "Rushing TDs", "Receiving TDs", "Kick Return TDs", "Punt Return TDs"]:
        if key in summary:
            td_total += summary[key]
    if td_total > 0:
        summary["Total TDs (approx)"] = td_total

    return summary

# ----------------------------------------------------------
# COMPARISON: BUILD A PER-YEAR DF FOR A PLAYER
# ----------------------------------------------------------
def get_player_stat_timeseries(source_df, player_row, id_col="Player Id", year_col="Year"):
    if source_df is None or id_col not in source_df.columns or year_col not in source_df.columns:
        return None

    pid = player_row[id_col]
    df = source_df[source_df[id_col] == pid].copy()
    if df.empty:
        return None

    # Clean numeric-like text
    for col in df.columns:
        if col not in [id_col, "Name", "Position", "Team", year_col]:
            if df[col].dtype == object:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("T", "", regex=False)
                )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure Year numeric
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df.dropna(subset=[year_col])
    df[year_col] = df[year_col].astype(int)

    return df


# ----------------------------------------------------------
# COMPARISON: GENERIC TWO-PLAYER CHART
# ----------------------------------------------------------
def show_two_player_comparison_chart(
    source_df,
    player1_row,
    player2_row,
    default_metric_name,
    title,
    key_prefix,
):
    df1 = get_player_stat_timeseries(source_df, player1_row)
    df2 = get_player_stat_timeseries(source_df, player2_row)

    if df1 is None and df2 is None:
        st.info("No stats available for either player in this category.")
        return
    if df1 is None:
        st.info(f"No stats available for {player1_row['Name']} in this category.")
        return
    if df2 is None:
        st.info(f"No stats available for {player2_row['Name']} in this category.")
        return

    # Decide which columns are numeric and useful
    numeric_cols = [
        c
        for c in df1.columns
        if c not in ["Player Id", "Name", "Position", "Team", "Year"]
        and pd.api.types.is_numeric_dtype(df1[c])
    ]

    if not numeric_cols:
        st.info("No numeric stats available to compare for this category.")
        return

    # Pick default metric
    if default_metric_name in numeric_cols:
        metric = default_metric_name
    else:
        metric = numeric_cols[0]

    metric = st.selectbox(
        f"{title} – Select Metric to Compare",
        options=numeric_cols,
        index=numeric_cols.index(metric),
        key=f"{key_prefix}_compare_metric",
    )

    name1 = player1_row["Name"]
    name2 = player2_row["Name"]

    # Build combined dataframe
    df1_m = df1[["Year", metric]].copy()
    df1_m["Player"] = name1

    df2_m = df2[["Year", metric]].copy()
    df2_m["Player"] = name2

    combined = pd.concat([df1_m, df2_m], ignore_index=True)

    # Drop rows where metric is NaN
    combined = combined.dropna(subset=[metric])
    if combined.empty:
        st.info("No valid numeric values to plot for this metric.")
        return

    # Altair line chart
    chart = (
        alt.Chart(combined)
        .mark_line(point=True)
        .encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y(f"{metric}:Q", title=metric),
            color=alt.Color("Player:N", title="Player"),
            tooltip=["Player", "Year", metric],
        )
        .properties(height=350, title=title)
    )

    st.altair_chart(chart, use_container_width=True)


# ----------------------------------------------------------
# CAREER SUMMARY CARDS
# ----------------------------------------------------------
summary = get_player_career_summary(player_data)

if summary:
    st.subheader("Career Summary")

    # Choose a display order for keys (if present)
    preferred_order = [
        "Passing Yards",
        "Passing TDs",
        "Interceptions Thrown",
        "Rushing Yards",
        "Rushing TDs",
        "Receiving Yards",
        "Receiving TDs",
        "Total TDs (approx)",
        "Total Tackles",
        "Sacks",
        "Interceptions",
        "Kick Return Yards",
        "Kick Return TDs",
        "Punt Return Yards",
        "Punt Return TDs",
        "FGs Made",
        "XPs Made",
        "Punts",
        "Gross Punting Yards",
    ]

    # Build a list of (label, value) for keys that exist and are > 0
    cards = []
    for key in preferred_order:
        if key in summary and summary[key] is not None:
            try:
                val = float(summary[key])
            except Exception:
                continue
            if val > 0:
                cards.append((key, summary[key]))

    if cards:
        # Show in rows of up to 4 cards
        n = len(cards)
        idx = 0
        while idx < n:
            row_cards = cards[idx: idx + 4]
            cols = st.columns(len(row_cards))
            for col, (label, value) in zip(cols, row_cards):
                col.metric(label, fmt_int(value) if isinstance(value, (int, float)) else str(value))
            idx += 4

# ----------------------------------------------------------
# COMPARISON SUMMARY CARDS (SIDE-BY-SIDE)
# ----------------------------------------------------------
if player_data_compare is not None:
    st.subheader("Comparison: Career Summary")

    summary1 = get_player_career_summary(player_data)
    summary2 = get_player_career_summary(player_data_compare)

    name1 = player_data["Name"]
    name2 = player_data_compare["Name"]

    # Use the same key order as before
    preferred_order = [
        "Passing Yards",
        "Passing TDs",
        "Interceptions Thrown",
        "Rushing Yards",
        "Rushing TDs",
        "Receiving Yards",
        "Receiving TDs",
        "Total TDs (approx)",
        "Total Tackles",
        "Sacks",
        "Interceptions",
        "Kick Return Yards",
        "Kick Return TDs",
        "Punt Return Yards",
        "Punt Return TDs",
        "FGs Made",
        "XPs Made",
        "Punts",
        "Gross Punting Yards",
    ]

    # Build a list of (label, val1, val2) for things where one or both have > 0
    rows = []
    for key in preferred_order:
        v1 = summary1.get(key, 0)
        v2 = summary2.get(key, 0)
        try:
            n1 = float(v1)
            n2 = float(v2)
        except Exception:
            continue

        if n1 > 0 or n2 > 0:
            rows.append((key, int(n1), int(n2)))

    if rows:
        # Show them in chunks of up to 4 metrics per row
        for i in range(0, len(rows), 4):
            chunk = rows[i : i + 4]
            cols = st.columns(len(chunk))
            for col, (label, v1, v2) in zip(cols, chunk):
                col.markdown(f"**{label}**")
                col.write(f"{name1}: {fmt_int(v1)}")
                col.write(f"{name2}: {fmt_int(v2)}")

# ----------------------------------------------------------
# TWO-PLAYER COMPARISON CHART
# ----------------------------------------------------------
if player_data_compare is not None:
    st.write("---")
    st.subheader("Two-Player Stat Comparison")

    category = st.radio(
        "Stat Category",
        [
            "Passing",
            "Rushing",
            "Receiving",
            "Defense",
            "Kick Return",
            "Punt Return",
            "Field Goals",
            "Kickoffs",
            "Punting",
        ],
        index=0,
        horizontal=True,
        key="compare_category_radio",
    )

    # ------- PASSING -------
    if category == "Passing":
        if passing_df is None:
            st.info("No passing data available.")
        else:
            show_two_player_comparison_chart(
                source_df=passing_df,
                player1_row=player_data,
                player2_row=player_data_compare,
                default_metric_name="Passing Yards",
                title="Passing Comparison",
                key_prefix="compare_passing",
            )

    # ------- RUSHING -------
    elif category == "Rushing":
        if rushing_df is None:
            st.info("No rushing data available.")
        else:
            show_two_player_comparison_chart(
                source_df=rushing_df,
                player1_row=player_data,
                player2_row=player_data_compare,
                default_metric_name="Rushing Yards",
                title="Rushing Comparison",
                key_prefix="compare_rushing",
            )

    # ------- RECEIVING -------
    elif category == "Receiving":
        if receiving_df is None:
            st.info("No receiving data available.")
        else:
            show_two_player_comparison_chart(
                source_df=receiving_df,
                player1_row=player_data,
                player2_row=player_data_compare,
                default_metric_name="Receiving Yards",
                title="Receiving Comparison",
                key_prefix="compare_receiving",
            )

    # ------- DEFENSE -------
    elif category == "Defense":
        if defensive_df is None:
            st.info("No defensive data available.")
        else:
            show_two_player_comparison_chart(
                source_df=defensive_df,
                player1_row=player_data,
                player2_row=player_data_compare,
                default_metric_name="Total Tackles",  # falls back if not present
                title="Defensive Comparison",
                key_prefix="compare_defense",
            )

    # ------- KICK RETURN -------
    elif category == "Kick Return":
        if kick_return_df is None:
            st.info("No kick return data available.")
        else:
            show_two_player_comparison_chart(
                source_df=kick_return_df,
                player1_row=player_data,
                player2_row=player_data_compare,
                default_metric_name="Yards Returned",
                title="Kick Return Comparison",
                key_prefix="compare_kick_return",
            )

    # ------- PUNT RETURN -------
    elif category == "Punt Return":
        if punt_return_df is None:
            st.info("No punt return data available.")
        else:
            show_two_player_comparison_chart(
                source_df=punt_return_df,
                player1_row=player_data,
                player2_row=player_data_compare,
                default_metric_name="Yards Returned",
                title="Punt Return Comparison",
                key_prefix="compare_punt_return",
            )

    # ------- FIELD GOALS / KICKER -------
    elif category == "Field Goals":
        if fg_df is None:
            st.info("No field goal kicking data available.")
        else:
            show_two_player_comparison_chart(
                source_df=fg_df,
                player1_row=player_data,
                player2_row=player_data_compare,
                default_metric_name="FGs Made",
                title="Field Goal Kicking Comparison",
                key_prefix="compare_fg",
            )

    # ------- KICKOFFS -------
    elif category == "Kickoffs":
        if kickoff_df is None:
            st.info("No kickoff data available.")
        else:
            show_two_player_comparison_chart(
                source_df=kickoff_df,
                player1_row=player_data,
                player2_row=player_data_compare,
                default_metric_name="Kickoff Yards",
                title="Kickoff Comparison",
                key_prefix="compare_kickoff",
            )

    # ------- PUNTING -------
    elif category == "Punting":
        if punting_df is None:
            st.info("No punting data available.")
        else:
            show_two_player_comparison_chart(
                source_df=punting_df,
                player1_row=player_data,
                player2_row=player_data_compare,
                default_metric_name="Gross Punting Yards",
                title="Punting Comparison",
                key_prefix="compare_punting",
            )

# ----------------------------------------------------------
# POSITION-BASED STAT ORDERING
# ----------------------------------------------------------
# ----------------------------------------------------------
# TABS FOR STATS
# ----------------------------------------------------------
tab_offense, tab_defense, tab_special, tab_logs = st.tabs(
    ["Offense", "Defense", "Special Teams", "Game Logs"]
)

pos = str(player_data["Position"]).upper()

if pos.startswith("QB"):
    with tab_offense:
        show_passing_stats()
        show_rushing_stats()
        show_off_receiving_stats()
    with tab_defense:
        show_off_defensive_stats()
    with tab_logs:
        show_qb_game_logs()

elif pos.startswith(("RB", "HB", "FB")):
    with tab_offense:
        show_rushing_stats()
        show_receiving_stats()
        show_off_passing_stats()
        show_kick_return_stats()
        show_punt_return_stats()
    with tab_defense:
        show_off_defensive_stats()
    with tab_logs:
        show_rb_game_logs()

elif pos.startswith(("WR", "TE")):
    with tab_offense:
        show_receiving_stats()
        show_off_rushing_stats()
        show_off_passing_stats()
    with tab_special:
        show_kick_return_stats()
        show_punt_return_stats()
    with tab_defense:
        show_off_defensive_stats()
    with tab_logs:
        show_wrte_game_logs()

elif pos.startswith(("G", "OG", "OL")):
    with tab_offense:
        show_offensive_line_stats()
        show_off_rushing_stats()
        show_off_receiving_stats()
    with tab_defense:
        show_off_defensive_stats()
    with tab_logs:
        show_ol_game_logs()

elif pos.startswith(("OT", "T")):
    with tab_offense:
        show_offensive_line_stats()
        show_off_rushing_stats()
        show_off_receiving_stats()
    with tab_defense:
        show_off_defensive_stats()
    with tab_logs:
        show_ol_game_logs()

elif pos.startswith(("LS", "C")):
    with tab_offense:
        show_offensive_line_stats()
        show_off_rushing_stats()
        show_off_receiving_stats()
    with tab_defense:
        show_off_defensive_stats()
    with tab_logs:
        show_ol_game_logs()
    
if pos.startswith(("K")):
    with tab_special:
        show_kicker_stats()
        show_kickoff_stats()
        show_punting_stats()
    with tab_defense:
        show_off_defensive_stats()
    with tab_offense:
        show_off_passing_stats()
    with tab_logs:
        show_kicker_game_logs()

elif pos.startswith(("P")):
    with tab_special:
        show_punting_stats()
        show_kicker_stats()
        show_kickoff_stats()
    with tab_offense:
        show_off_rushing_stats()
        show_off_passing_stats()
    with tab_defense:
        show_off_defensive_stats()
    with tab_logs:
        show_punter_game_logs()

if pos.startswith(("CB", "DB", "FS")):
    with tab_defense:
        show_defensive_stats()
    with tab_special:
        show_kick_return_stats()
        show_punt_return_stats()
    with tab_offense:
        show_off_receiving_stats()

elif pos.startswith(("SS", "SAF")):
    with tab_defense:
        show_defensive_stats()
    with tab_special:
        show_kick_return_stats()
        show_punt_return_stats()
    with tab_offense:
        show_off_receiving_stats()

if pos.startswith(("DE", "DL", "DT")):
    with tab_defense:
        show_off_defensive_stats()
    with tab_offense:
        show_offensive_line_stats()
        show_off_rushing_stats()
        show_off_receiving_stats()

if pos.startswith(("ILB", "LB", "MLB")):
    with tab_defense:
        show_off_defensive_stats()
    with tab_offense:
        show_offensive_line_stats()
        show_off_rushing_stats()
        show_off_receiving_stats()

if pos.startswith(("NT","OLB")):
    with tab_defense:
        show_off_defensive_stats()
    with tab_offense:
        show_offensive_line_stats()
        show_off_rushing_stats()
        show_off_receiving_stats()

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

with st.expander("DEBUG: Raw Age Values"):
    st.write(df_filtered["Age"].head(50).tolist())
