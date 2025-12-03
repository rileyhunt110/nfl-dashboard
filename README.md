🏈 NFL Player Analytics Dashboard

An interactive Streamlit dashboard for exploring NFL player statistics, league leaders, team summaries, and player similarity across multiple positions and seasons.

This project showcases skills in data analysis, feature engineering, interactive dashboard creation, and analytical storytelling. It also demonstrates how to structure multi-table sports data for real-world analytics workflows.

🔍 Overview

The dashboard provides a rich environment for exploring NFL player data, including:

✔ Global Player Filters

Filter players by:

* Team

* Position

All views update dynamically based on the filters selected.

✔ Summary Metrics

At a glance:

Number of players shown

Average age

Average height

Average weight

✔ Multiple Analytical Views
1. Positions Overview

Bar chart showing number of players per position

Great for roster composition and league-wide positional analysis

2. Team Dashboard

Includes:

Team roster table

Team average age/height/weight

Aggregated career totals for players on the selected team

Leaderboards for the team’s strongest performers

3. Player Distributions

Explore league-wide or filtered histograms for:

Age

Height

Weight

Helps visualize physical trends across teams and positions.

4. League Leaders

Top performers in:

Passing

Rushing

Receiving

Defensive statistics

Supports:

Career totals

Single-season leaders

5. Player Lookup & Similarity

Search any player and view:

Detailed bio

Career and season stats

Similar player comparisons using:

StandardScaler

NearestNeighbors

Multi-metric similarity across several performance groups

🧱 Data Model (Conceptual)

Even though the app reads CSV files from the data/ folder, the underlying structure is equivalent to this relational model:

Players
|Column	|Description|
|---|---|
|player_id	|Unique player ID |
|name	|Player name|
|position	|Position abbreviation|
|team	|Current team|
|height_in	|Height (in inches)|
|weight_lb	|Weight (in pounds)|
|age	|Age|
|college	|College attended|

SeasonStats
Per-player, per-season statistics:
|Column	|Description|
|---|---|
|player_id	|Foreign key to Players|
|season	|Season year|
|team	|Player's team that year|
|passing_yards	|Total passing yards|
|rushing_yards	|Total rushing yards|
|receiving_yards	|Receiving yards|
|tackles	|Total tackles|
|sacks	|Total sacks|
|interceptions	|Interceptions made|
|...	|(other stat columns)|

Career Aggregations

Computed within the app by grouping season stats.

Similarity Model Features

Numeric stat features (passing, rushing, receiving, defense) are:

Selected by feature group

Standardized

Used to compute nearest neighbors

This demonstrates structured thinking in schema design and analytical modeling.

⚙️ Tech Stack
|Technology	|Purpose|
|---|---|
|Python	|Core programming|
|Streamlit	|Web application framework|
|Pandas	|Data loading & manipulation|
|NumPy	|Numeric operations|
|Altair	|Visualization|
|scikit-learn	|Similarity modeling (scaling + nearest neighbor search)|

📁 Project Structure
```bash
project/
│
├── data/
│   ├── Basic_Stats.csv
│   ├── Passing_Stats.csv
│   ├── Rushing_Receiving_Stats.csv
│   ├── Defense_Stats.csv
│   └── (other CSVs)
│
├── app.py
├── requirements.txt
└── README.md
```

🚀 Getting Started
1. Clone the repository
git clone https://github.com/<your-username>/nfl-dashboard.git
cd nfl-dashboard

2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

4. Add the dataset

Place all NFL CSV files inside the data/ folder.
Filenames should match what the code expects (Basic_Stats, Passing_Stats, etc.).

5. Run the dashboard
streamlit run app.py


Open the URL that appears in the terminal (usually http://localhost:8501).
