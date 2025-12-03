import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
# POSITION GROUPING (for smarter similarity)
# ----------------------------------------------------------
def position_group(pos_raw: str) -> str:
    if not isinstance(pos_raw, str):
        return "OTHER"
    pos = pos_raw.upper().strip()

    # Quarterbacks
    if pos.startswith("QB"):
        return "QB"

    # Running backs / fullbacks
    if pos.startswith(("RB", "HB", "FB")):
        return "RB"

    # Wide receivers / tight ends
    if pos.startswith(("WR", "TE", "SE", "FL")):
        return "WRTE"

    # Offensive line
    if pos in {"C", "G", "OG", "OT", "T", "LT", "RT", "OL"}:
        return "OL"

    # Defensive line
    if pos in {"DE", "DL", "DT", "NT"}:
        return "DL"

    # Linebackers
    if pos in {"LB", "MLB", "ILB", "OLB"}:
        return "LB"

    # Defensive backs
    if pos in {"CB", "DB", "FS", "SS", "SAF"}:
        return "DB"

    # Specialists
    if pos.startswith("K"):
        return "K"
    if pos.startswith("P"):
        return "P"
    if pos in {"KR", "PR"}:
        return "RET"

    return "OTHER"


# LEAGUE LEADERS HELPERS
def compute_leaders(
    stat_df,
    value_col,
    top_n=10,
    label=None,
    year=None,
):
    """
    Aggregate a stat per player and return top N.
    If `year` is provided and the dataframe has a 'Year' column,
    it filters to that season before aggregating.

    Also respects current sidebar filters via df_filtered (Player Id subset).
    """
    if stat_df is None:
        return None

    if value_col not in stat_df.columns:
        return None

    df = stat_df.copy()

    # Optional single-season filter
    if year is not None and "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df = df[df["Year"] == year]

    # Minimal subset
    cols_to_keep = ["Player Id", "Name", value_col]
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_to_keep].copy()

    # Clean numeric field (commas, 'T', etc.)
    df[value_col] = (
        df[value_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("T", "", regex=False)
    )
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    # Group by player (we do NOT group by Year after filtering; we want
    # per-player totals within the selected year or across all years)
    group_cols = [c for c in ["Player Id", "Name"] if c in df.columns]
    if not group_cols:
        return None

    grouped = (
        df.groupby(group_cols, as_index=False)[value_col]
        .sum()
    )

    # Bring in team & position from basic stats (df_filtered)
    join_cols = [c for c in ["Player Id"] if c in grouped.columns and c in df_filtered.columns]
    if join_cols:
        meta_cols = [c for c in ["Player Id", "Position", "Current Team"] if c in df_filtered.columns]
        meta = df_filtered[meta_cols].drop_duplicates(subset=join_cols)
        leaders = grouped.merge(meta, on=join_cols, how="left")
    else:
        leaders = grouped

    # Respect current filters (df_filtered has only filtered players already)
    if "Player Id" in leaders.columns and "Player Id" in df_filtered.columns:
        allowed_ids = set(df_filtered["Player Id"])
        leaders = leaders[leaders["Player Id"].isin(allowed_ids)]

    # Drop 0 rows
    leaders = leaders[leaders[value_col] > 0]

    if leaders.empty:
        return None

    leaders = leaders.sort_values(by=value_col, ascending=False).head(top_n)

    # Nice display / chart dataframe
    leaders = leaders.copy()
    leaders.rename(columns={value_col: label or value_col}, inplace=True)

    return leaders

def show_leaderboard(title, leaders_df, stat_col):
    """Show a leaderboard table + horizontal bar chart, with optional jump-to-player."""
    if leaders_df is None or leaders_df.empty:
        st.info(f"No data available for {title.lower()} under current filters.")
        return

    st.markdown(f"**{title}**")

    # Display table (Name, Team, Position, Stat)
    display_cols = []
    for c in ["Name", "Current Team", "Position", stat_col]:
        if c in leaders_df.columns:
            display_cols.append(c)

    st.dataframe(leaders_df[display_cols])

    # Build chart
    if "Name" in leaders_df.columns and stat_col in leaders_df.columns:
        chart_df = leaders_df[["Name", stat_col]].copy()
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                y=alt.Y("Name:N", sort="-x", title="Player"),
                x=alt.X(f"{stat_col}:Q", title=stat_col),
                tooltip=["Name", stat_col],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

# TEAM DASHBOARD HELPERS
def compute_team_stat_totals(
    stat_df,
    team_player_ids,
    value_col,
    top_n=5,
    label=None,
):
    """
    For a given career stats dataframe (e.g., passing_df) and
    a set of player IDs on the selected team, compute:
      - total stat for the team
      - top N players on that team for this stat
    Returns (total_value, top_df_or_None).
    """
    if stat_df is None:
        return 0, None
    if value_col not in stat_df.columns:
        return 0, None
    if not team_player_ids:
        return 0, None

    df_stats = stat_df.copy()
    if "Player Id" not in df_stats.columns:
        return 0, None

    df_stats = df_stats[df_stats["Player Id"].isin(team_player_ids)].copy()
    if df_stats.empty:
        return 0, None

    # Clean numeric
    df_stats[value_col] = (
        df_stats[value_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("T", "", regex=False)
    )
    df_stats[value_col] = pd.to_numeric(df_stats[value_col], errors="coerce").fillna(0)

    # Group by player (sum across seasons)
    group_cols = [c for c in ["Player Id", "Name"] if c in df_stats.columns]
    grouped = (
        df_stats.groupby(group_cols, as_index=False)[value_col]
        .sum()
    )

    total = grouped[value_col].sum()

    # Attach position & current team from basic stats (global df)
    join_cols = [c for c in ["Player Id"] if c in grouped.columns and c in df.columns]
    if join_cols:
        meta_cols = [c for c in ["Player Id", "Position", "Current Team"] if c in df.columns]
        meta = df[meta_cols].drop_duplicates(subset=join_cols)
        top_df = grouped.merge(meta, on=join_cols, how="left")
    else:
        top_df = grouped

    top_df = top_df[top_df[value_col] > 0]
    if top_df.empty:
        return total, None

    top_df = top_df.sort_values(by=value_col, ascending=False).head(top_n)
    top_df = top_df.copy()
    top_df.rename(columns={value_col: label or value_col}, inplace=True)

    return total, top_df

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

@st.cache
def build_cluster_feature_df():
    """
    Build a per-player career feature matrix for clustering.
    Uses totals from passing, rushing, receiving, defense, returns, kicking, punting.
    """
    if "Player Id" not in df.columns:
        return pd.DataFrame(), []

    rows = []

    for _, row in df.iterrows():
        pid = row["Player Id"]
        name = row.get("Name", "")
        pos = row.get("Position", "")
        team = row.get("Current Team", "")

        # Base feature dict
        feats = {
            "Player Id": pid,
            "Name": name,
            "Position": pos,
            "Current Team": team,
            # Offense
            "PassYds": 0.0,
            "PassTD": 0.0,
            "IntThrown": 0.0,
            "RushYds": 0.0,
            "RushTD": 0.0,
            "RecYds": 0.0,
            "RecTD": 0.0,
            # Defense
            "Tackles": 0.0,
            "Sacks": 0.0,
            "DefInts": 0.0,
            # Special teams
            "FGMade": 0.0,
            "XPMade": 0.0,
            "Punts": 0.0,
            "PuntYds": 0.0,
            "KR_Yds": 0.0,
            "PR_Yds": 0.0,
        }

        # ----- Passing -----
        if passing_df is not None and "Player Id" in passing_df.columns:
            p = passing_df[passing_df["Player Id"] == pid].copy()
            if not p.empty:
                if "Passing Yards" in p.columns:
                    p["Passing Yards"] = (
                        p["Passing Yards"].astype(str).str.replace(",", "", regex=False)
                    )
                    feats["PassYds"] = pd.to_numeric(
                        p["Passing Yards"], errors="coerce"
                    ).fillna(0).sum()

                if "TD Passes" in p.columns:
                    feats["PassTD"] = pd.to_numeric(
                        p["TD Passes"], errors="coerce"
                    ).fillna(0).sum()

                if "Ints" in p.columns:
                    feats["IntThrown"] = pd.to_numeric(
                        p["Ints"], errors="coerce"
                    ).fillna(0).sum()

        # ----- Rushing -----
        if rushing_df is not None and "Player Id" in rushing_df.columns:
            r = rushing_df[rushing_df["Player Id"] == pid].copy()
            if not r.empty:
                if "Rushing Yards" in r.columns:
                    r["Rushing Yards"] = (
                        r["Rushing Yards"].astype(str).str.replace(",", "", regex=False)
                    )
                    feats["RushYds"] = pd.to_numeric(
                        r["Rushing Yards"], errors="coerce"
                    ).fillna(0).sum()

                if "Rushing TDs" in r.columns:
                    feats["RushTD"] = pd.to_numeric(
                        r["Rushing TDs"], errors="coerce"
                    ).fillna(0).sum()

        # ----- Receiving -----
        if receiving_df is not None and "Player Id" in receiving_df.columns:
            rec = receiving_df[receiving_df["Player Id"] == pid].copy()
            if not rec.empty:
                if "Receiving Yards" in rec.columns:
                    rec["Receiving Yards"] = (
                        rec["Receiving Yards"].astype(str).str.replace(",", "", regex=False)
                    )
                    feats["RecYds"] = pd.to_numeric(
                        rec["Receiving Yards"], errors="coerce"
                    ).fillna(0).sum()

                if "Receiving TDs" in rec.columns:
                    feats["RecTD"] = pd.to_numeric(
                        rec["Receiving TDs"], errors="coerce"
                    ).fillna(0).sum()

        # ----- Defense -----
        if defensive_df is not None and "Player Id" in defensive_df.columns:
            d = defensive_df[defensive_df["Player Id"] == pid].copy()
            if not d.empty:
                if "Total Tackles" in d.columns:
                    feats["Tackles"] = pd.to_numeric(
                        d["Total Tackles"], errors="coerce"
                    ).fillna(0).sum()

                if "Sacks" in d.columns:
                    feats["Sacks"] = pd.to_numeric(
                        d["Sacks"], errors="coerce"
                    ).fillna(0).sum()

                if "Ints" in d.columns:
                    feats["DefInts"] = pd.to_numeric(
                        d["Ints"], errors="coerce"
                    ).fillna(0).sum()

        # ----- Kick Return -----
        if kick_return_df is not None and "Player Id" in kick_return_df.columns:
            kr = kick_return_df[kick_return_df["Player Id"] == pid].copy()
            if not kr.empty and "Yards Returned" in kr.columns:
                kr["Yards Returned"] = (
                    kr["Yards Returned"].astype(str).str.replace(",", "", regex=False)
                )
                feats["KR_Yds"] = pd.to_numeric(
                    kr["Yards Returned"], errors="coerce"
                ).fillna(0).sum()

        # ----- Punt Return -----
        if punt_return_df is not None and "Player Id" in punt_return_df.columns:
            pr = punt_return_df[punt_return_df["Player Id"] == pid].copy()
            if not pr.empty and "Yards Returned" in pr.columns:
                pr["Yards Returned"] = (
                    pr["Yards Returned"].astype(str).str.replace(",", "", regex=False)
                )
                feats["PR_Yds"] = pd.to_numeric(
                    pr["Yards Returned"], errors="coerce"
                ).fillna(0).sum()

        # ----- Kicking -----
        if fg_df is not None and "Player Id" in fg_df.columns:
            k = fg_df[fg_df["Player Id"] == pid].copy()
            if not k.empty:
                if "FGs Made" in k.columns:
                    feats["FGMade"] = pd.to_numeric(
                        k["FGs Made"], errors="coerce"
                    ).fillna(0).sum()

                if "Extra Points Made" in k.columns:
                    feats["XPMade"] = pd.to_numeric(
                        k["Extra Points Made"], errors="coerce"
                    ).fillna(0).sum()

        # ----- Punting -----
        if punting_df is not None and "Player Id" in punting_df.columns:
            pnt = punting_df[punting_df["Player Id"] == pid].copy()
            if not pnt.empty:
                if "Punts" in pnt.columns:
                    feats["Punts"] = pd.to_numeric(
                        pnt["Punts"], errors="coerce"
                    ).fillna(0).sum()

                if "Gross Punting Yards" in pnt.columns:
                    pnt["Gross Punting Yards"] = (
                        pnt["Gross Punting Yards"].astype(str).str.replace(",", "", regex=False)
                    )
                    feats["PuntYds"] = pd.to_numeric(
                        pnt["Gross Punting Yards"], errors="coerce"
                    ).fillna(0).sum()

        rows.append(feats)

    feature_df = pd.DataFrame(rows)

    # Ensure numeric types
    feature_cols = [
        "PassYds", "PassTD", "IntThrown",
        "RushYds", "RushTD",
        "RecYds", "RecTD",
        "Tackles", "Sacks", "DefInts",
        "FGMade", "XPMade",
        "Punts", "PuntYds",
        "KR_Yds", "PR_Yds",
    ]

    for c in feature_cols:
        if c in feature_df.columns:
            feature_df[c] = pd.to_numeric(feature_df[c], errors="coerce").fillna(0.0)
        else:
            feature_df[c] = 0.0

    return feature_df, feature_cols

def pretty_feature_name(feat: str) -> str:
    """Convert short feature codes into nicer labels."""
    mapping = {
        "PassYds": "Passing Yards",
        "PassTD": "Passing TDs",
        "IntThrown": "Interceptions Thrown",
        "RushYds": "Rushing Yards",
        "RushTD": "Rushing TDs",
        "RecYds": "Receiving Yards",
        "RecTD": "Receiving TDs",
        "Tackles": "Total Tackles",
        "Sacks": "Sacks",
        "DefInts": "Defensive INTs",
        "FGMade": "FGs Made",
        "XPMade": "Extra Points Made",
        "Punts": "Punts",
        "PuntYds": "Punt Yards",
        "KR_Yds": "Kick Return Yards",
        "PR_Yds": "Punt Return Yards",
    }
    return mapping.get(feat, feat)


def cluster_archetype_prefix(feature_group: str) -> str:
    """Short prefix based on which feature group we're clustering on."""
    if feature_group == "Offense":
        return "Offensive archetype"
    elif feature_group == "Defense":
        return "Defensive archetype"
    elif feature_group == "Special Teams":
        return "Special teams archetype"
    else:
        return "Player archetype"


def build_cluster_label_map(centers, feature_cols, feature_group: str):
    """
    Given KMeans centers (in scaled space) + feature names, return
    a dict: cluster_index -> human-readable archetype label.
    We look at the largest-magnitude features for each cluster.
    """
    labels = {}
    prefix = cluster_archetype_prefix(feature_group)

    centers = np.asarray(centers)

    for idx, center in enumerate(centers):
        # Sort features by absolute importance in this cluster center
        center = np.asarray(center)
        order = np.argsort(-np.abs(center))
        # Take top 2–3 features
        top_feats = [feature_cols[i] for i in order[:3] if i < len(feature_cols)]

        if not top_feats:
            labels[idx] = f"{prefix} {idx}"
            continue

        pretty_feats = [pretty_feature_name(f) for f in top_feats]
        # Simple label like: "Offensive archetype: Passing Yards & Passing TDs"
        if len(pretty_feats) == 1:
            core = pretty_feats[0]
        else:
            core = " & ".join(pretty_feats[:2])

        labels[idx] = f"{prefix}: {core}"

    return labels

def normalize_position(pos: str) -> str:
    pos = pos.upper()
    if pos.startswith("QB"): return "QB"
    if pos in ("RB","HB","FB"): return "RB"
    if pos.startswith(("WR","TE")): return "WR"
    if pos.startswith(("DE","DL","DT","NT")): return "DL"
    if pos.startswith(("ILB","LB","MLB","OLB")): return "LB"
    if pos.startswith(("CB","DB")): return "CB"
    if pos.startswith(("FS","SS","SAF")): return "S"
    return "OTHER"

def _get_feat(fdict, key):
    return float(fdict.get(key, 0.0) or 0.0)

def _share(part, total):
    total = float(total)
    if total <= 0:
        return 0.0
    return float(part) / total

def _soft_norm(x, scale):
    # squashes x to roughly 0–1 range based on a rough scale
    return float(x) / (float(x) + scale)


# ----------------------------------------------------------
# Player Archetype Scoring
# ----------------------------------------------------------

# Quarterback Archetypes
def classify_qb_archetype(center):
    scores = {
        "Pocket Passer": 0,
        "Gunslinger": 0,
        "Game Manager": 0,
        "Dual Threat": 0,
        "Rushing QB": 0,
        "Play Action Specialist": 0,
    }

    att_pg = center.get("PassAttemptsPerGame", 0)
    ypa = center.get("PassingYardsPerAttempt", 0)
    rush_yards = center.get("RushYds", 0)
    rush_td = center.get("RushTD", 0)
    int_rate = center.get("IntRate", 0)
    sack_rate = center.get("SackRate", 0)
    comp_pct = center.get("CompletionPercentage", 0)

    # --- Pocket Passer ---
    if att_pg > 30: scores["Pocket Passer"] += 2
    if comp_pct > 65: scores["Pocket Passer"] += 1
    if rush_yards < 150: scores["Pocket Passer"] += 1

    # --- Gunslinger ---
    if ypa > 7.8: scores["Gunslinger"] += 2
    if int_rate > 2.5: scores["Gunslinger"] += 2
    if rush_yards < 200: scores["Gunslinger"] += 1

    # --- Game Manager ---
    if att_pg < 25: scores["Game Manager"] += 2
    if int_rate < 1.5: scores["Game Manager"] += 2
    if comp_pct >= 62: scores["Game Manager"] += 1

    # --- Dual Threat ---
    if rush_yards > 400: scores["Dual Threat"] += 3
    if rush_td >= 4: scores["Dual Threat"] += 2
    if sack_rate < 5: scores["Dual Threat"] += 1

    # --- Rushing QB ---
    if rush_yards > 650: scores["Rushing QB"] += 4
    if rush_td >= 8: scores["Rushing QB"] += 3

    # --- Play Action Specialist ---
    if ypa > 8.5 and att_pg < 28:
        scores["Play Action Specialist"] += 3

    return scores

# Running Back Archetypes
def classify_rb_archetype(center):
    scores = {
        "Bell Cow": 0,
        "Power Back": 0,
        "Speed Back": 0,
        "Receiving Back": 0,
        "Change of Pace": 0,
        "Workhorse Runner": 0,
    }

    carries_pg = center.get("RushingAttemptsPerGame", 0)
    rush_yards = center.get("RushYds", 0)
    ypc = center.get("YardsPerCarry", 0)
    rec_yards = center.get("RecYds", 0)
    receptions = center.get("Receptions", 0)
    long_runs20 = center.get("RushingMoreThan20", 0)

    # --- Bell Cow ---
    if carries_pg >= 15: scores["Bell Cow"] += 3
    if rush_yards > 1000: scores["Bell Cow"] += 2

    # --- Power Back ---
    if ypc < 4.0: scores["Power Back"] += 2
    if center.get("RushTD", 0) >= 8: scores["Power Back"] += 2

    # --- Speed Back ---
    if ypc > 4.8: scores["Speed Back"] += 2
    if long_runs20 >= 5: scores["Speed Back"] += 2

    # --- Receiving Back ---
    if receptions >= 40: scores["Receiving Back"] += 3
    if rec_yards > 300: scores["Receiving Back"] += 2

    # --- Change of Pace ---
    if carries_pg < 8 and ypc > 4.5:
        scores["Change of Pace"] += 3

    # --- Workhorse Runner ---
    if carries_pg >= 18 and receptions < 20:
        scores["Workhorse Runner"] += 3

    return scores

# Receiver Archetypes
def classify_wr_archetype(center):
    scores = {
        "Deep Threat": 0,
        "Possession Receiver": 0,
        "Slot Receiver": 0,
        "X Receiver": 0,
        "Z Receiver": 0,
        "Gadget Hybrid": 0,
    }

    ypr = center.get("YardsPerReception", 0)
    rec = center.get("Receptions", 0)
    rec_yards = center.get("ReceivingYards", 0)
    rush_attempts = center.get("RushingAttempts", 0)
    td_rate = center.get("ReceivingTDs", 0)

    # --- Deep Threat ---
    if ypr > 15: scores["Deep Threat"] += 3
    if td_rate > 6: scores["Deep Threat"] += 2

    # --- Possession ---
    if ypr < 12: scores["Possession Receiver"] += 2
    if rec >= 70: scores["Possession Receiver"] += 2

    # --- Slot ---
    if rec >= 60 and ypr < 11:
        scores["Slot Receiver"] += 3

    # --- X Receiver ---
    if rec >= 80 or rec_yards >= 1000:
        scores["X Receiver"] += 3

    # --- Z Receiver ---
    if 12 <= ypr <= 15: scores["Z Receiver"] += 2

    # --- Gadget ---
    if rush_attempts >= 5:
        scores["Gadget Hybrid"] += 3

    return scores

# Defensive Line Archetypes
def classify_dl_archetype(center: dict) -> str:
    """
    Uses only: Total Tackles, Sacks
    to distinguish Speed Rusher / Power Rusher / Run Stopper / Nose / 3-Tech-style.
    """
    scores = {
        "Speed Rusher": 0,
        "Power Rusher": 0,
        "Run-Stopper Edge": 0,
        "Nose Tackle": 0,
        "Penetrating 3-Tech": 0,
    }

    sacks = center.get("Sacks", 0) or 0
    tackles = center.get("Total Tackles", 0) or 0

    # Normalize a bit just in case we’re looking at multi-year totals
    # but don't overthink it – it's just relative.
    sacks_per_50_tk = sacks / max(tackles, 1) * 50

    # --- Speed Rusher ---
    if sacks_per_50_tk >= 8:  # lots of sacks relative to tackles
        scores["Speed Rusher"] += 3
    if sacks >= 8:
        scores["Speed Rusher"] += 2
    if tackles < 60:
        scores["Speed Rusher"] += 1

    # --- Power Rusher ---
    if 4 <= sacks <= 10 and tackles >= 50:
        scores["Power Rusher"] += 3
    if sacks >= 6 and tackles >= 70:
        scores["Power Rusher"] += 1

    # --- Run-Stopper Edge ---
    if tackles >= 75 and sacks <= 5:
        scores["Run-Stopper Edge"] += 3

    # --- Nose Tackle ---
    if sacks <= 2 and tackles <= 55:
        scores["Nose Tackle"] += 3

    # --- Penetrating 3-Tech ---
    if sacks >= 6 and 40 <= tackles <= 80:
        scores["Penetrating 3-Tech"] += 3

    return scores

# Linebacker Archetypes
def classify_lb_archetype(center: dict) -> str:
    """
    LB archetypes:
      - Tackling Machine / MIKE
      - Coverage Linebacker
      - Pass-Rush LB / Edge Hybrid
    Uses: Total Tackles, Sacks, Passes Defended, Ints
    """
    scores = {
        "Tackling Machine MIKE": 0,
        "Coverage Linebacker": 0,
        "Pass-Rush LB / Edge": 0,
    }

    tackles = center.get("Total Tackles", 0) or 0
    sacks = center.get("Sacks", 0) or 0
    pdef = center.get("Passes Defended", 0) or 0
    ints = center.get("Ints", 0) or 0

    # --- Tackling Machine ---
    if tackles >= 100:
        scores["Tackling Machine MIKE"] += 3
    if tackles >= 130:
        scores["Tackling Machine MIKE"] += 2
    if sacks < 6 and pdef < 10:
        scores["Tackling Machine MIKE"] += 1

    # --- Coverage Linebacker ---
    if pdef >= 8:
        scores["Coverage Linebacker"] += 3
    if ints >= 3:
        scores["Coverage Linebacker"] += 2
    if tackles <= 110:
        scores["Coverage Linebacker"] += 1

    # --- Pass-Rush LB / Edge Hybrid ---
    if sacks >= 7:
        scores["Pass-Rush LB / Edge"] += 3
    if sacks >= 10:
        scores["Pass-Rush LB / Edge"] += 2
    if pdef <= 6 and ints <= 2:
        scores["Pass-Rush LB / Edge"] += 1

    return scores

# Corner Archetypes
def classify_cb_archetype(center: dict) -> str:
    """
    CB archetypes:
      - Ball-Hawk Corner
      - Shutdown / Low-Target Corner
      - Nickel Corner
      - Zone Corner
    Uses: Total Tackles, Passes Defended, Ints, Int Yards
    """
    scores = {
        "Ball-Hawk Corner": 0,
        "Shutdown Corner": 0,
        "Nickel Corner": 0,
        "Zone Corner": 0,
    }

    tackles = center.get("Total Tackles", 0) or 0
    pdef = center.get("Passes Defended", 0) or 0
    ints = center.get("Ints", 0) or 0
    int_yards = center.get("Int Yards", 0) or 0

    # --- Ball-Hawk ---
    if ints >= 4:
        scores["Ball-Hawk Corner"] += 3
    if pdef >= 10:
        scores["Ball-Hawk Corner"] += 2
    if int_yards >= 80:
        scores["Ball-Hawk Corner"] += 1

    # --- Shutdown (low target) ---
    if tackles <= 45 and pdef <= 8 and ints <= 3:
        scores["Shutdown Corner"] += 3

    # --- Nickel Corner ---
    if tackles >= 60:
        scores["Nickel Corner"] += 3
    if pdef >= 6 and ints <= 3:
        scores["Nickel Corner"] += 1

    # --- Zone Corner ---
    # More help tackling, more PD than INTs
    if tackles >= 50 and pdef >= 8 and ints <= 4:
        scores["Zone Corner"] += 3

    return scores

# Safety Archhetypes

def classify_s_archetype(center: dict) -> str:
    """
    Safety archetypes:
      - Free Safety / Centerfielder
      - Box Safety
      - Hybrid SS/LB
      - Blitz Safety
    Uses: Total Tackles, Sacks, Passes Defended, Ints, Safties
    """
    scores = {
        "Free Safety / Centerfielder": 0,
        "Box Safety": 0,
        "Hybrid SS/LB": 0,
        "Blitz Safety": 0,
    }

    tackles = center.get("Total Tackles", 0) or 0
    sacks = center.get("Sacks", 0) or 0
    pdef = center.get("Passes Defended", 0) or 0
    ints = center.get("Ints", 0) or 0
    safeties = center.get("Safties", 0) or 0  # column spelled this way

    # --- Free Safety / Centerfielder ---
    if ints >= 4:
        scores["Free Safety / Centerfielder"] += 3
    if pdef >= 9:
        scores["Free Safety / Centerfielder"] += 2
    if tackles <= 80:
        scores["Free Safety / Centerfielder"] += 1

    # --- Box Safety ---
    if tackles >= 90:
        scores["Box Safety"] += 3
    if sacks >= 2:
        scores["Box Safety"] += 1
    if ints <= 3:
        scores["Box Safety"] += 1

    # --- Hybrid SS/LB ---
    if tackles >= 100:
        scores["Hybrid SS/LB"] += 2
    if sacks >= 2 and pdef >= 6:
        scores["Hybrid SS/LB"] += 2

    # --- Blitz Safety ---
    if sacks >= 3:
        scores["Blitz Safety"] += 3
    if safeties >= 1:
        scores["Blitz Safety"] += 2

    return scores


def classify_archetype(center, position_group):
    if position_group == "QB":
        return classify_qb_archetype(center)
    if position_group == "RB":
        return classify_rb_archetype(center)
    if position_group == "WR":
        return classify_wr_archetype(center)
    if position_group == "DL":
        return classify_dl_archetype(center)
    if position_group == "LB":
        return classify_lb_archetype(center)
    if position_group == "CB":
        return classify_cb_archetype(center)
    if position_group == "S":
        return classify_s_archetype(center)
    return "Unknown Archetype"

def get_player_archetype_debug(player_row):
    """
    Return (label, scores_dict) for a single player based on the same
    feature construction used in clustering.
    """
    if "Player Id" not in player_row:
        return "Unknown", {}

    pid = player_row["Player Id"]

    # Reuse your cluster feature builder
    cluster_features_df, all_feature_cols = build_cluster_feature_df()
    if cluster_features_df.empty:
        return "Unknown", {}

    row = cluster_features_df[cluster_features_df["Player Id"] == pid]
    if row.empty:
        return "Unknown", {}

    row = row.iloc[0]

    # Build feature dict
    feat_dict = {col: float(row[col]) for col in all_feature_cols if col in row}

    pos_group = normalize_position(player_row["Position"])
    label, scores = classify_archetype(feat_dict, pos_group, return_scores=True)

    return label, scores

# ----------------------------------------------------------
# PLAYER SIMILARITY MODEL
# ----------------------------------------------------------
@st.cache(allow_output_mutation=True)
def build_player_similarity_models():
    """
    Build:
      - player_meta: basic info per player (index = Player Id)
      - features: numeric feature matrix (index = Player Id)
      - scaler: StandardScaler fitted on features
      - knn: NearestNeighbors model fitted on scaled features
    """
    # Basic player metadata
    player_meta = (
        df[["Player Id", "Name", "Position", "Current Team"]]
        .dropna(subset=["Player Id"])
        .drop_duplicates(subset=["Player Id"])
        .set_index("Player Id")
    )

    # Add position group for smarter similarity
    player_meta["PosGroup"] = player_meta["Position"].astype(str).apply(position_group)

    # Start features as just metadata index, no numeric cols yet
    feats = player_meta.copy()

    # Helper: add aggregated stats from a stats DF
    def add_stats(stat_df, prefix=""):
        nonlocal feats
        if stat_df is None or "Player Id" not in stat_df.columns:
            return

        temp = stat_df.copy()

        # Clean numeric-like text
        for col in temp.columns:
            if col not in ["Player Id", "Name", "Position", "Team", "Year"]:
                if temp[col].dtype == object:
                    temp[col] = (
                        temp[col]
                        .astype(str)
                        .str.replace(",", "", regex=False)
                        .str.replace("T", "", regex=False)
                    )
                temp[col] = pd.to_numeric(temp[col], errors="coerce")

        # Group by player across all seasons
        numeric_cols = [
            c
            for c in temp.columns
            if c not in ["Player Id", "Name", "Position", "Team", "Year"]
        ]
        if not numeric_cols:
            return

        agg = (
            temp[["Player Id"] + numeric_cols]
            .groupby("Player Id", as_index=True)
            .sum()
        )

        # Prefix numeric columns
        agg = agg.add_prefix(prefix)

        # Align indices to feats (all players)
        agg = agg.reindex(feats.index, fill_value=0)

        # Add / overwrite columns in feats
        for col in agg.columns:
            feats[col] = agg[col]

    # Add features from every stat file
    add_stats(passing_df, prefix="pass_")
    add_stats(rushing_df, prefix="rush_")
    add_stats(receiving_df, prefix="recv_")
    add_stats(defensive_df, prefix="def_")
    add_stats(fg_df, prefix="fg_")
    add_stats(kick_return_df, prefix="kr_")
    add_stats(punt_return_df, prefix="pr_")
    add_stats(punting_df, prefix="punt_")
    add_stats(kickoff_df, prefix="ko_")

    # Extract numeric feature matrix ONLY (drop name/position/etc)
    features = feats.select_dtypes(include=["number"])

    # If no numeric features, bail gracefully
    if features.empty or features.shape[1] == 0:
        return player_meta, features, None, None

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)

    # Use more neighbors internally so we can filter by position group
    n_samples = features.shape[0]
    n_neighbors = min(max(10, 30), n_samples)  # between 10 and 30, but ≤ n_samples

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(X_scaled)

    return player_meta, features, scaler, knn


# Build similarity structures once
player_meta, player_features, similarity_scaler, similarity_knn = build_player_similarity_models()


def find_similar_players(player_id, top_k=5):
    """
    Given a Player Id, return up to top_k most similar players.
    Similarity is:
      - computed in numeric feature space
      - filtered to same position group when possible
    """
    # If model didn't build (no numeric features), just bail
    if similarity_knn is None or similarity_scaler is None:
        return None

    if player_id not in player_features.index:
        return None

    # Get pos group for this player
    pos_group = None
    if player_id in player_meta.index and "PosGroup" in player_meta.columns:
        pos_group = player_meta.loc[player_id, "PosGroup"]

    # Feature vector for this player
    vec = player_features.loc[player_id].values.reshape(1, -1)
    vec_scaled = similarity_scaler.transform(vec)

    # Query nearest neighbors
    distances, indices = similarity_knn.kneighbors(vec_scaled)

    distances = distances[0]
    indices = indices[0]

    id_list = player_features.index.to_list()

    similar = []

    # First pass: same position group only (if pos_group known)
    for dist, idx in zip(distances, indices):
        if idx < 0 or idx >= len(id_list):
            continue

        pid = id_list[idx]
        if pid == player_id:
            continue

        if pos_group is not None:
            # Skip different groups in first pass
            if pid not in player_meta.index:
                continue
            if player_meta.loc[pid, "PosGroup"] != pos_group:
                continue

        meta_row = player_meta.loc[pid]
        similar.append(
            {
                "Player Id": pid,
                "Name": meta_row["Name"],
                "Position": meta_row["Position"],
                "Team": meta_row["Current Team"],
                "PosGroup": meta_row["PosGroup"],
                "Distance": float(dist),
            }
        )

        if len(similar) >= top_k:
            break

    # If we didn't get enough from same group, fill with closest others
    if len(similar) < top_k:
        for dist, idx in zip(distances, indices):
            if idx < 0 or idx >= len(id_list):
                continue

            pid = id_list[idx]
            if pid == player_id:
                continue

            # Skip if already added
            if any(s["Player Id"] == pid for s in similar):
                continue

            if pid not in player_meta.index:
                continue

            meta_row = player_meta.loc[pid]
            similar.append(
                {
                    "Player Id": pid,
                    "Name": meta_row["Name"],
                    "Position": meta_row["Position"],
                    "Team": meta_row["Current Team"],
                    "PosGroup": meta_row["PosGroup"],
                    "Distance": float(dist),
                }
            )

            if len(similar) >= top_k:
                break

    if not similar:
        return None

    return similar

# ----------------------------------------------------------
# STAT PROFILE COMPARISON (radar-style line chart)
# ----------------------------------------------------------
def get_profile_features_for_group(pos_group: str):
    """
    Decide which aggregated features to use for profile plots,
    based on our prefixed feature columns in player_features.
    """
    pos_group = (pos_group or "OTHER").upper()

    if pos_group == "QB":
        return [
            "pass_Passing Yards",
            "pass_TD Passes",
            "pass_Ints",
            "rush_Rushing Yards",
        ]
    if pos_group == "RB":
        return [
            "rush_Rushing Yards",
            "rush_Rushing TDs",
            "recv_Receiving Yards",
        ]
    if pos_group == "WRTE":
        return [
            "recv_Receiving Yards",
            "recv_Receiving TDs",
            "rush_Rushing Yards",
        ]
    if pos_group in {"DB", "LB"}:
        return [
            "def_Total Tackles",
            "def_Sacks",
            "def_Ints",
        ]
    if pos_group == "DL":
        return [
            "def_Total Tackles",
            "def_Sacks",
        ]
    if pos_group == "K":
        return [
            "fg_FGs Made",
            "fg_Extra Points Made",
        ]
    if pos_group == "P":
        return [
            "punt_Punts",
            "punt_Gross Punting Yards",
        ]
    if pos_group == "RET":
        return [
            "kr_Yards Returned",
            "kr_Returns for TDs",
            "pr_Yards Returned",
            "pr_Returns for TDs",
        ]

    # Fallback: just pick a few of the largest total columns if they exist
    if not player_features.empty:
        return list(player_features.columns[:5])

    return []


def make_stat_profile_dataframe(player_id_1, player_id_2):
    """
    Build a long-form dataframe with normalized stats for two players:
    columns: Metric, Player, Value (0..1)
    """
    if player_features is None or player_features.empty:
        return None

    if player_id_1 not in player_meta.index:
        return None
    if player_id_2 not in player_meta.index:
        return None

    pg1 = player_meta.loc[player_id_1, "PosGroup"]
    pg2 = player_meta.loc[player_id_2, "PosGroup"]

    # Use the first player's group as the reference
    pos_group = pg1 or pg2
    feature_cols = get_profile_features_for_group(pos_group)

    feature_cols = [c for c in feature_cols if c in player_features.columns]
    if not feature_cols:
        return None

    v1 = player_features.loc[player_id_1, feature_cols].astype(float)
    v2 = player_features.loc[player_id_2, feature_cols].astype(float)

    # Combine to normalize per-feature
    mat = pd.DataFrame(
        {
            "Player1": v1,
            "Player2": v2,
        },
        index=feature_cols,
    )

    # Normalize each feature to 0..1 across the two players
    mins = mat.min(axis=1)
    maxs = mat.max(axis=1)
    ranges = maxs - mins

    # Avoid divide-by-zero: if range == 0 → all zeros
    norm = pd.DataFrame(index=feature_cols, columns=["Player1", "Player2"], dtype=float)
    for f in feature_cols:
        if ranges[f] == 0:
            norm.loc[f, "Player1"] = 0.0
            norm.loc[f, "Player2"] = 0.0
        else:
            norm.loc[f, "Player1"] = (mat.loc[f, "Player1"] - mins[f]) / ranges[f]
            norm.loc[f, "Player2"] = (mat.loc[f, "Player2"] - mins[f]) / ranges[f]

    # Long form
    name1 = player_meta.loc[player_id_1, "Name"]
    name2 = player_meta.loc[player_id_2, "Name"]

    df_long = pd.DataFrame(
        {
            "Metric": list(feature_cols) * 2,
            "Player": [name1] * len(feature_cols) + [name2] * len(feature_cols),
            "Value": pd.concat(
                [norm["Player1"], norm["Player2"]], ignore_index=True
            ),
        }
    )

    # Clean metric labels (strip prefixes like "pass_", "rush_")
    df_long["Metric"] = df_long["Metric"].apply(
        lambda s: s.split("_", 1)[1] if "_" in s else s
    )

    return df_long


def show_stat_profile_chart(player_id_1, player_id_2, key_prefix="stat_profile"):
    df_long = make_stat_profile_dataframe(player_id_1, player_id_2)
    if df_long is None or df_long.empty:
        st.info("Not enough data to build a stat profile comparison.")
        return

    chart = (
        alt.Chart(df_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("Metric:N", title="Metric"),
            y=alt.Y(
                "Value:Q",
                title="Normalized Value (0–1)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color("Player:N", title="Player"),
            tooltip=["Player", "Metric", "Value"],
        )
        .properties(height=350, title="Stat Profile Comparison")
    )

    st.altair_chart(chart, use_container_width=True)


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
tab_pos, tab_team, tab_dist, tab_leaders, tab_clusters = st.tabs(
    ["Positions", "Teams", "Distributions", "League Leaders", "tab_clusters"]
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


# TEAMS (TEAM DASHBOARDS)
with tab_team:
    st.markdown("### Team Dashboard")

    # Use all teams from the basic stats (not filtered),
    # but you could use df_filtered if you want it tied to sidebar filters.
    all_teams = (
        df["Current Team"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    if not all_teams:
        st.info("No team information available.")
    else:
        selected_team = st.selectbox(
            "Select a Team",
            all_teams,
            key="team_dashboard_select",
        )

        team_basic = df[df["Current Team"] == selected_team].copy()

        if team_basic.empty:
            st.info("No players found for this team.")
        else:
            # ---------- SUMMARY METRICS ----------
            st.markdown(f"#### {selected_team} Overview")

            # Player count
            num_players = len(team_basic)

            # Avg age / height / weight
            age_series = pd.to_numeric(team_basic.get("Age"), errors="coerce")
            h_series = pd.to_numeric(team_basic.get("Height (inches)"), errors="coerce")
            w_series = pd.to_numeric(team_basic.get("Weight (lbs)"), errors="coerce")

            col_a, col_b, col_c, col_d = st.columns(4)

            col_a.metric("Players", num_players)

            if age_series.notna().any():
                col_b.metric("Avg Age", f"{age_series.mean():.1f}")
            else:
                col_b.metric("Avg Age", "N/A")

            if h_series.notna().any():
                col_c.metric("Avg Height (in)", f"{h_series.mean():.1f}")
            else:
                col_c.metric("Avg Height (in)", "N/A")

            if w_series.notna().any():
                col_d.metric("Avg Weight (lbs)", f"{w_series.mean():.1f}")
            else:
                col_d.metric("Avg Weight (lbs)", "N/A")

            st.write("---")

            # ---------- ROSTER TABLE ----------
            st.markdown("#### Roster")

            roster_cols = [
                "Name",
                "Number",
                "Position",
                "Age",
                "Height (inches)",
                "Weight (lbs)",
                "Experience",
                "College",
            ]
            roster_cols = [c for c in roster_cols if c in team_basic.columns]

            st.dataframe(team_basic[roster_cols].sort_values(by="Position"))

            st.write("---")

            # Set of player IDs on this team (for career stats aggregation)
            team_player_ids = set(team_basic["Player Id"].dropna()) if "Player Id" in team_basic.columns else set()

            # ---------- TEAM AGGREGATE STATS ----------
            st.markdown("#### Team Career Totals (All Players on Current Roster)")

            col_off1, col_off2 = st.columns(2)

            with col_off1:
                # Passing
                if passing_df is not None:
                    total_pass_yds, top_passers = compute_team_stat_totals(
                        passing_df,
                        team_player_ids,
                        value_col="Passing Yards",
                        top_n=5,
                        label="Passing Yards",
                    )
                    st.metric("Total Passing Yards", f"{int(total_pass_yds):,}")

                    if top_passers is not None and not top_passers.empty:
                        st.markdown("**Top Passers**")
                        st.dataframe(top_passers[["Name", "Position", "Passing Yards"]])

                # Rushing
                if rushing_df is not None:
                    total_rush_yds, top_rushers = compute_team_stat_totals(
                        rushing_df,
                        team_player_ids,
                        value_col="Rushing Yards",
                        top_n=5,
                        label="Rushing Yards",
                    )
                    st.metric("Total Rushing Yards", f"{int(total_rush_yds):,}")

                    if top_rushers is not None and not top_rushers.empty:
                        st.markdown("**Top Rushers**")
                        st.dataframe(top_rushers[["Name", "Position", "Rushing Yards"]])

            with col_off2:
                # Receiving
                if receiving_df is not None:
                    total_recv_yds, top_receivers = compute_team_stat_totals(
                        receiving_df,
                        team_player_ids,
                        value_col="Receiving Yards",
                        top_n=5,
                        label="Receiving Yards",
                    )
                    st.metric("Total Receiving Yards", f"{int(total_recv_yds):,}")

                    if top_receivers is not None and not top_receivers.empty:
                        st.markdown("**Top Receivers**")
                        st.dataframe(top_receivers[["Name", "Position", "Receiving Yards"]])

                # Defensive tackles
                if defensive_df is not None:
                    total_tackles, top_defenders = compute_team_stat_totals(
                        defensive_df,
                        team_player_ids,
                        value_col="Total Tackles",
                        top_n=5,
                        label="Total Tackles",
                    )
                    st.metric("Total Tackles", f"{int(total_tackles):,}")

                    if top_defenders is not None and not top_defenders.empty:
                        st.markdown("**Top Tacklers**")
                        st.dataframe(top_defenders[["Name", "Position", "Total Tackles"]])

            st.write("---")

            # ---------- SPECIAL TEAMS SUMMARY ----------
            st.markdown("#### Special Teams Summary")

            col_st1, col_st2 = st.columns(2)

            with col_st1:
                if fg_df is not None:
                    total_fgs, top_kickers = compute_team_stat_totals(
                        fg_df,
                        team_player_ids,
                        value_col="FGs Made",
                        top_n=3,
                        label="FGs Made",
                    )
                    st.metric("Field Goals Made", f"{int(total_fgs):,}")

                    if top_kickers is not None and not top_kickers.empty:
                        st.markdown("**Top Kickers**")
                        st.dataframe(top_kickers[["Name", "Position", "FGs Made"]])

            with col_st2:
                if punting_df is not None:
                    total_punts, top_punters = compute_team_stat_totals(
                        punting_df,
                        team_player_ids,
                        value_col="Punts",
                        top_n=3,
                        label="Punts",
                    )
                    st.metric("Total Punts", f"{int(total_punts):,}")

                    if top_punters is not None and not top_punters.empty:
                        st.markdown("**Top Punters**")
                        st.dataframe(top_punters[["Name", "Position", "Punts"]])




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

# ----------------------------------------------------------
# LEAGUE LEADERS TAB
# ----------------------------------------------------------
with tab_leaders:
    st.markdown(
        "League leaders based on **career totals** or **single-season** stats "
        "within the current filters."
    )

    # Career vs Single Season
    mode = st.radio(
        "Leaderboard Type",
        ["Career Totals", "Single Season"],
        index=0,
        horizontal=True,
        key="leaders_mode",
    )

    selected_year = None

    # If Single Season, pick a season (Year)
    if mode == "Single Season":
        year_set = set()

        for df_ in [passing_df, rushing_df, receiving_df, defensive_df, fg_df, punting_df]:
            if df_ is not None and "Year" in df_.columns:
                yrs = pd.to_numeric(df_["Year"], errors="coerce").dropna().unique()
                year_set.update(int(y) for y in yrs)

        if year_set:
            years_list = sorted(year_set)
            year_strs = [str(y) for y in years_list]
            year_choice = st.selectbox(
                "Season (Year)",
                year_strs,
                index=len(year_strs) - 1,
                key="leaders_year",
            )
            selected_year = int(year_choice)
            st.markdown(f"_Showing leaders for the {selected_year} season._")
        else:
            st.info("No 'Year' information available to compute single-season leaders.")
            mode = "Career Totals"  # fallback to career

    # top N
    top_n = 10

    col1, col2 = st.columns(2)

    # Helper to pass year when in Single Season mode
    year_arg = selected_year if mode == "Single Season" else None

    # ---------- PASSING LEADERS ----------
    with col1:
        if passing_df is not None:
            leaders_pass_yds = compute_leaders(
                passing_df,
                value_col="Passing Yards",
                top_n=top_n,
                label="Passing Yards",
                year=year_arg,
            )
            show_leaderboard(
                "Passing Yards Leaders",
                leaders_pass_yds,
                "Passing Yards",
            )

        if passing_df is not None:
            leaders_pass_tds = compute_leaders(
                passing_df,
                value_col="TD Passes",
                top_n=top_n,
                label="TD Passes",
                year=year_arg,
            )
            show_leaderboard(
                "Passing TD Leaders",
                leaders_pass_tds,
                "TD Passes",
            )

    # ---------- RUSHING / RECEIVING LEADERS ----------
    with col2:
        if rushing_df is not None:
            leaders_rush_yds = compute_leaders(
                rushing_df,
                value_col="Rushing Yards",
                top_n=top_n,
                label="Rushing Yards",
                year=year_arg,
            )
            show_leaderboard(
                "Rushing Yards Leaders",
                leaders_rush_yds,
                "Rushing Yards",
            )

        if receiving_df is not None:
            leaders_recv_yds = compute_leaders(
                receiving_df,
                value_col="Receiving Yards",
                top_n=top_n,
                label="Receiving Yards",
                year=year_arg,
            )
            show_leaderboard(
                "Receiving Yards Leaders",
                leaders_recv_yds,
                "Receiving Yards",
            )

    st.write("---")

    col3, col4 = st.columns(2)

    # ---------- RECEIVING TDs + RUSH TDs ----------
    with col3:
        if receiving_df is not None:
            leaders_recv_tds = compute_leaders(
                receiving_df,
                value_col="Receiving TDs",
                top_n=top_n,
                label="Receiving TDs",
                year=year_arg,
            )
            show_leaderboard(
                "Receiving TD Leaders",
                leaders_recv_tds,
                "Receiving TDs",
            )

        if rushing_df is not None:
            leaders_rush_tds = compute_leaders(
                rushing_df,
                value_col="Rushing TDs",
                top_n=top_n,
                label="Rushing TDs",
                year=year_arg,
            )
            show_leaderboard(
                "Rushing TD Leaders",
                leaders_rush_tds,
                "Rushing TDs",
            )

    # ---------- DEFENSE / SPECIAL TEAMS ----------
    with col4:
        if defensive_df is not None:
            leaders_tackles = compute_leaders(
                defensive_df,
                value_col="Total Tackles",
                top_n=top_n,
                label="Total Tackles",
                year=year_arg,
            )
            show_leaderboard(
                "Total Tackles Leaders",
                leaders_tackles,
                "Total Tackles",
            )

        if fg_df is not None:
            leaders_fgs = compute_leaders(
                fg_df,
                value_col="FGs Made",
                top_n=top_n,
                label="FGs Made",
                year=year_arg,
            )
            show_leaderboard(
                "Field Goals Made Leaders",
                leaders_fgs,
                "FGs Made",
            )

        if punting_df is not None:
            leaders_punts = compute_leaders(
                punting_df,
                value_col="Punts",
                top_n=top_n,
                label="Punts",
                year=year_arg,
            )
            show_leaderboard(
                "Punting Leaders (Punts)",
                leaders_punts,
                "Punts",
            )

# ----------------------------------------------------------
# PLAYER CLUSTERS TAB
# ----------------------------------------------------------
# with tab_clusters:
#     st.markdown("### Player Clusters (K-Means)")

#     # Build base feature matrix (same as for similarity)
#     cluster_features_df, all_feature_cols = build_cluster_feature_df()

#     if cluster_features_df.empty:
#         st.info("No feature data available for clustering.")
#     else:
#         # Respect current sidebar filters (team + position)
#         if "Player Id" in df_filtered.columns:
#             cluster_df = cluster_features_df[
#                 cluster_features_df["Player Id"].isin(df_filtered["Player Id"])
#             ].copy()
#         else:
#             cluster_df = cluster_features_df.copy()

#         if cluster_df.empty:
#             st.info("No players available under current filters to cluster.")
#         else:
#             # Position filter inside cluster tab
#             pos_options = (
#                 cluster_df["Position"]
#                 .fillna("Unknown")
#                 .astype(str)
#                 .sort_values()
#                 .unique()
#                 .tolist()
#             )

#             pos_choice = st.selectbox(
#                 "Position group to cluster",
#                 ["All Positions"] + pos_options,
#                 index=0,
#                 key="cluster_pos_choice",
#             )

#             if pos_choice != "All Positions":
#                 cluster_df = cluster_df[cluster_df["Position"] == pos_choice].copy()

#             if cluster_df.empty:
#                 st.info("No players match this position filter.")
#             else:
#                 # Choose which feature group to use
#                 feature_group = st.radio(
#                     "Feature group",
#                     ["All", "Offense", "Defense", "Special Teams"],
#                     index=0,
#                     horizontal=True,
#                     key="cluster_feature_group",
#                 )

#                 if feature_group == "Offense":
#                     base_feature_cols = [
#                         "PassYds", "PassTD", "IntThrown",
#                         "RushYds", "RushTD",
#                         "RecYds", "RecTD",
#                     ]
#                 elif feature_group == "Defense":
#                     base_feature_cols = ["Tackles", "Sacks", "DefInts"]
#                 elif feature_group == "Special Teams":
#                     base_feature_cols = [
#                         "FGMade", "XPMade",
#                         "Punts", "PuntYds",
#                         "KR_Yds", "PR_Yds",
#                     ]
#                 else:  # "All"
#                     base_feature_cols = list(all_feature_cols)

#                 # Only keep features that actually exist in this df
#                 feature_cols = [c for c in base_feature_cols if c in cluster_df.columns]

#                 if not feature_cols:
#                     st.info("No numeric features available for the selected group.")
#                 else:
#                     # Filter out players with all-zero features (optional)
#                     mat = cluster_df[feature_cols].to_numpy()
#                     non_zero_mask = (mat.sum(axis=1) != 0)
#                     cluster_df = cluster_df[non_zero_mask].copy()
#                     mat = mat[non_zero_mask]

#                     if cluster_df.shape[0] < 2:
#                         st.info("Not enough players with non-zero stats to form clusters.")
#                     else:
#                         # Number of clusters
#                         max_k = min(10, cluster_df.shape[0])
#                         k = st.slider(
#                             "Number of clusters (K)",
#                             min_value=2,
#                             max_value=max_k,
#                             value=min(5, max_k),
#                             step=1,
#                             key="cluster_k",
#                         )

#                         # Scale + cluster
#                         scaler = StandardScaler()
#                         X_scaled = scaler.fit_transform(mat)

#                         kmeans = KMeans(
#                             n_clusters=k,
#                             random_state=42,
#                             n_init=10,
#                         )
#                         labels = kmeans.fit_predict(X_scaled)

#                         cluster_df = cluster_df.copy()
#                         cluster_df["Cluster"] = labels

#                         # ----- Archetype labels per cluster -----
#                         cluster_label_map = {}
#                         for i in range(k):
#                             # center in original (unscaled) feature units
#                             center_scaled = kmeans.cluster_centers_[i].reshape(1, -1)
#                             center_raw = scaler.inverse_transform(center_scaled)[0]
#                             center_dict = dict(zip(feature_cols, center_raw))

#                             # dominant position in this cluster
#                             cluster_positions = (
#                                 cluster_df.loc[cluster_df["Cluster"] == i, "Position"]
#                                 .dropna()
#                                 .astype(str)
#                                 .tolist()
#                             )
#                             if cluster_positions:
#                                 pos_counts = {}
#                                 for p in cluster_positions:
#                                     pos_counts[p] = pos_counts.get(p, 0) + 1
#                                 dominant_pos = max(pos_counts, key=pos_counts.get)
#                             else:
#                                 dominant_pos = "OTHER"

#                             pos_group = normalize_position(dominant_pos)
#                             label = classify_archetype(center_dict, pos_group)
#                             cluster_label_map[i] = label

#                         cluster_df["Archetype"] = cluster_df["Cluster"].map(cluster_label_map)

#                         st.markdown("#### Cluster Assignments (Filtered Players)")
#                         cols_to_show = ["Name", "Current Team", "Position", "Cluster", "Archetype"] + feature_cols
#                         cols_to_show = [c for c in cols_to_show if c in cluster_df.columns]

#                         st.dataframe(
#                             cluster_df[cols_to_show].sort_values(by=["Cluster", "Name"])
#                         )

#                         # 2D scatter
#                         if len(feature_cols) >= 2:
#                             x_feat = st.selectbox(
#                                 "X-axis feature",
#                                 feature_cols,
#                                 index=0,
#                                 key="cluster_x_feat",
#                             )
#                             y_feat = st.selectbox(
#                                 "Y-axis feature",
#                                 feature_cols,
#                                 index=1,
#                                 key="cluster_y_feat",
#                             )

#                             scatter_df = cluster_df[
#                                 ["Name", "Current Team", "Position", "Cluster", "Archetype", x_feat, y_feat]
#                             ].copy()

#                             chart = (
#                                 alt.Chart(scatter_df)
#                                 .mark_circle(size=80)
#                                 .encode(
#                                     x=alt.X(f"{x_feat}:Q", title=x_feat),
#                                     y=alt.Y(f"{y_feat}:Q", title=y_feat),
#                                     color=alt.Color("Archetype:N", title="Archetype"),
#                                     tooltip=[
#                                         "Name",
#                                         "Position",
#                                         "Current Team",
#                                         "Cluster",
#                                         "Archetype",
#                                         x_feat,
#                                         y_feat,
#                                     ],
#                                 )
#                                 .properties(height=400, title="Player Clusters by Archetype")
#                             )

#                             st.markdown("#### Cluster Visualization")
#                             st.altair_chart(chart, use_container_width=True)
#                         else:
#                             st.info("Not enough features to make a 2D scatter plot.")


# ----------------------------------------------------------
# PLAYER LOOKUP
# ----------------------------------------------------------
st.subheader("Player Lookup")

player_list = sorted(df_filtered["Name"].unique())
player_options = ["-- Select a player --"] + player_list

# Determine default index based on session_state (for jump-from-leaders)
default_index = 0
if "player_select" in st.session_state:
    try:
        default_index = player_options.index(st.session_state["player_select"])
    except ValueError:
        default_index = 0

selected_player = st.selectbox(
    "Select a Player",
    player_options,
    index=default_index,
    key="player_select",
)

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
# ARCHETYPE DEBUG SECTION
# ----------------------------------------------------------
archetype_label, archetype_scores = get_player_archetype_debug(player_data)

st.subheader("Experimental Archetype (Debug)")

if not archetype_scores:
    st.info("No archetype data available for this player.")
else:
    st.markdown(f"**Guessed Archetype:** {archetype_label}")

    # Show a small table of scores
    scores_df = (
        pd.DataFrame(
            [
                {"Archetype": name, "Score": float(score)}
                for name, score in archetype_scores.items()
            ]
        )
        .sort_values(by="Score", ascending=False)
        .reset_index(drop=True)
    )

    st.markdown("**Archetype Scores (higher = stronger match)**")
    st.dataframe(scores_df)


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
# GENERIC GAME LOGS SECTION (season + context + rolling + heatmap)
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
      - optional Season (Year) filter
      - optional context filters: Home/Away, Opponent, Result/Outcome
      - cleans '--', '-', '' to NaN then 0 for numeric stats
      - sorts games (Year+Week or Game Date if available)
      - adds a 'Game #' index
      - shows full game log table
      - lets user pick a numeric metric to plot vs Game #
      - optional rolling averages (3-game, 5-game)
      - optional heatmap: metric vs Opponent & Year
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

    # ------------------------------------------------------
    # SEASON FILTER (Year)
    # ------------------------------------------------------
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

    # ------------------------------------------------------
    # CONTEXT FILTERS: LOCATION, OPPONENT, RESULT
    # ------------------------------------------------------
    # Detect opponent column
    opponent_col = None
    for cand in ["Opponent", "Opp"]:
        if cand in logs.columns:
            opponent_col = cand
            break

    # Detect home/away-like column
    home_col = None
    for cand in ["Home or Away", "Location"]:
        if cand in logs.columns:
            home_col = cand
            break

    # Detect result/outcome column
    result_col = None
    for cand in ["Result", "Outcome"]:
        if cand in logs.columns:
            result_col = cand
            break

    # Home/Away filter
    if home_col is not None:
        loc_values = (
            logs[home_col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        loc_values_sorted = sorted(loc_values)
        loc_options = ["All Locations"] + loc_values_sorted

        selected_loc = st.selectbox(
            "Location (Home/Away)",
            loc_options,
            index=0,
            key=f"{key_prefix}_gamelog_location",
        )

        if selected_loc != "All Locations":
            logs = logs[logs[home_col].astype(str) == selected_loc].copy()
            if logs.empty:
                st.info(f"No games found for location filter: {selected_loc}.")
                return

    # Opponent filter
    if opponent_col is not None:
        opp_values = (
            logs[opponent_col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        opp_values_sorted = sorted(opp_values)
        opp_options = ["All Opponents"] + opp_values_sorted

        selected_opp = st.selectbox(
            "Opponent",
            opp_options,
            index=0,
            key=f"{key_prefix}_gamelog_opp",
        )

        if selected_opp != "All Opponents":
            logs = logs[logs[opponent_col].astype(str) == selected_opp].copy()
            if logs.empty:
                st.info(f"No games found against opponent: {selected_opp}.")
                return

    # Result filter
    if result_col is not None:
        res_values = (
            logs[result_col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        res_values_sorted = sorted(res_values)
        res_options = ["All Results"] + res_values_sorted

        selected_res = st.selectbox(
            "Result",
            res_options,
            index=0,
            key=f"{key_prefix}_gamelog_result",
        )

        if selected_res != "All Results":
            logs = logs[logs[result_col].astype(str) == selected_res].copy()
            if logs.empty:
                st.info(f"No games found with result: {selected_res}.")
                return

    # ------------------------------------------------------
    # SORT GAMES
    # ------------------------------------------------------
    if has_year and has_week:
        logs = logs.sort_values(by=["Year", "Week"])
    elif has_year:
        logs = logs.sort_values(by="Year")
    elif "Game Date" in logs.columns:
        logs = logs.sort_values(by="Game Date")

    # Add a simple game index for plotting
    logs.insert(0, "Game #", range(1, len(logs) + 1))

    # ------------------------------------------------------
    # NUMERIC CONVERSION & TABLE DISPLAY
    # ------------------------------------------------------
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

    # ------------------------------------------------------
    # METRIC SELECTION + ROLLING AVERAGE LINE CHART
    # ------------------------------------------------------
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

    plot_df = logs[["Game #", metric]].dropna()
    if plot_df.empty:
        st.info("No valid values to plot for this metric.")
        return

    # Rolling average selector
    rolling_choice = st.selectbox(
        "Rolling average (games)",
        ["None", "3-game", "5-game"],
        index=0,
        key=f"{key_prefix}_gamelog_roll",
    )

    plot_df = plot_df.set_index("Game #")

    if rolling_choice != "None":
        window = 3 if rolling_choice.startswith("3") else 5
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
        key_prefix="passing_main",
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
        key_prefix="rushing_main",
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
        key_prefix="receiving_main",
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
        key_prefix="defensive_main",
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
        key_prefix="defensive_off",
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
        key_prefix="passing_off",
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
        key_prefix="rushing_off",
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
        key_prefix="receiving_off",
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
# SIMILAR PLAYERS
# ----------------------------------------------------------
st.subheader("Similar Players")

similar_list = find_similar_players(player_data["Player Id"], top_k=5)

if not similar_list:
    st.info("No similar player data available.")
else:
    sim_df = pd.DataFrame(similar_list)

    # Show table including position group
    cols_to_show = [c for c in ["Name", "Position", "Team", "PosGroup", "Distance"] if c in sim_df.columns]
    st.dataframe(sim_df[cols_to_show])

    # Distance bar chart
    chart = (
        alt.Chart(sim_df)
        .mark_bar()
        .encode(
            x=alt.X("Distance:Q", title="Similarity Distance (lower = more similar)"),
            y=alt.Y("Name:N", sort="-x", title="Player"),
            color="Distance:Q",
            tooltip=["Name", "Position", "Team", "PosGroup", "Distance"],
        )
        .properties(height=250, title="Closest Players in Feature Space")
    )
    st.altair_chart(chart, use_container_width=True)

    # -------- Stat Profile Comparison with a chosen similar player --------
    st.markdown("#### Stat Profile vs Similar Player")

    # Default to the most similar one (first row)
    default_sim_id = sim_df.iloc[0]["Player Id"]

    # Select which similar player to compare
    sim_name_options = sim_df["Name"].tolist()
    sim_name_default = sim_df.iloc[0]["Name"]

    chosen_name = st.selectbox(
        "Choose a similar player to compare stat profile with",
        sim_name_options,
        index=sim_name_options.index(sim_name_default),
        key="similar_player_profile_select",
    )

    chosen_row = sim_df[sim_df["Name"] == chosen_name].iloc[0]
    chosen_id = chosen_row["Player Id"]

    show_stat_profile_chart(player_data["Player Id"], chosen_id, key_prefix="stat_profile")

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
