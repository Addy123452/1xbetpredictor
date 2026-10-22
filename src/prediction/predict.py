"""Predict 1X2 probabilities for a match from the saved model + team state.

Usage:
    python -m src.prediction.predict --home "Brazil" --away "Argentina"
    python -m src.prediction.predict --home "France" --away "England" --neutral
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"


def load_artifacts():
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    state = json.loads((MODELS_DIR / "team_state.json").read_text(encoding="utf-8"))
    columns = json.loads((MODELS_DIR / "feature_columns.json").read_text(encoding="utf-8"))
    return model, state, columns


def build_row(home: str, away: str, state: dict, neutral: bool) -> dict:
    hs, as_ = state[home], state[away]
    return {
        "home_elo": hs["elo"],
        "away_elo": as_["elo"],
        "elo_diff": hs["elo"] - as_["elo"],
        "home_form_ppg": hs["form_ppg"],
        "away_form_ppg": as_["form_ppg"],
        "form_diff": hs["form_ppg"] - as_["form_ppg"],
        "home_form_gd": hs["form_gd"],
        "away_form_gd": as_["form_gd"],
        "gd_diff": hs["form_gd"] - as_["form_gd"],
        "home_home_ppg": hs["home_ppg"],
        "away_away_ppg": as_["away_ppg"],
        "opp_strength_home": as_["elo"],
        "opp_strength_away": hs["elo"],
        "neutral": int(neutral),
    }


def predict(home: str, away: str, neutral: bool = False) -> dict:
    model, state, columns = load_artifacts()
    for team in (home, away):
        if team not in state:
            raise SystemExit(f"Unknown team: {team!r} (try a name from models/team_state.json)")
    row = build_row(home, away, state, neutral)
    X = np.array([[row[c] for c in columns]], dtype=float)
    proba = model.predict_proba(X)[0]
    return {label: float(p) for label, p in zip(model.classes_, proba)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a football match outcome.")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--neutral", action="store_true", help="Neutral venue (no home advantage)")
    args = parser.parse_args()

    probs = predict(args.home, args.away, args.neutral)
    venue = " (neutral)" if args.neutral else ""
    print(f"\n{args.home} vs {args.away}{venue}")
    for label, name in (("H", "Home win"), ("D", "Draw"), ("A", "Away win")):
        print(f"  {name:9s} {probs.get(label, 0.0) * 100:5.1f}%")
    print("\nEstimates only — not guarantees, not betting advice.")


if __name__ == "__main__":
    main()
