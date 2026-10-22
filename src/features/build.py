"""Assemble per-match features using only pre-match information."""
from collections import defaultdict

import pandas as pd

from .elo import BASE_ELO, update_elo
from .form import FormTracker

FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_form_ppg",
    "away_form_ppg",
    "form_diff",
    "home_form_gd",
    "away_form_gd",
    "gd_diff",
    "home_home_ppg",
    "away_away_ppg",
    "opp_strength_home",
    "opp_strength_away",
    "neutral",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Single chronological pass: features come from state *before* each match."""
    elo = defaultdict(lambda: BASE_ELO)
    form = FormTracker()
    rows = []

    for r in df.itertuples(index=False):
        home, away = r.home_team, r.away_team
        home_stats = form.features(home)
        away_stats = form.features(away)
        home_elo, away_elo = elo[home], elo[away]

        rows.append({
            "date": r.date,
            "home_team": home,
            "away_team": away,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "home_form_ppg": home_stats["form_ppg"],
            "away_form_ppg": away_stats["form_ppg"],
            "form_diff": home_stats["form_ppg"] - away_stats["form_ppg"],
            "home_form_gd": home_stats["form_gd"],
            "away_form_gd": away_stats["form_gd"],
            "gd_diff": home_stats["form_gd"] - away_stats["form_gd"],
            "home_home_ppg": form.home_ppg(home),
            "away_away_ppg": form.away_ppg(away),
            "opp_strength_home": away_elo,
            "opp_strength_away": home_elo,
            "neutral": int(r.neutral),
            "result": r.result,
            "home_score": r.home_score,
            "away_score": r.away_score,
        })

        new_home, new_away = update_elo(
            home_elo, away_elo, r.home_score, r.away_score, r.tournament, r.neutral
        )
        elo[home], elo[away] = new_home, new_away
        form.update(home, away, r.home_score, r.away_score)

    return pd.DataFrame(rows)


def current_team_state(df: pd.DataFrame) -> dict:
    """Final Elo + form per team, used to build a prediction row."""
    elo = defaultdict(lambda: BASE_ELO)
    form = FormTracker()

    for r in df.itertuples(index=False):
        new_home, new_away = update_elo(
            elo[r.home_team], elo[r.away_team], r.home_score, r.away_score,
            r.tournament, r.neutral,
        )
        elo[r.home_team], elo[r.away_team] = new_home, new_away
        form.update(r.home_team, r.away_team, r.home_score, r.away_score)

    state = {}
    for team in elo:
        stats = form.features(team)
        state[team] = {
            "elo": round(float(elo[team]), 2),
            "form_ppg": round(float(stats["form_ppg"]), 3),
            "form_gd": round(float(stats["form_gd"]), 3),
            "home_ppg": round(float(form.home_ppg(team)), 3),
            "away_ppg": round(float(form.away_ppg(team)), 3),
        }
    return state
