"""Load and clean the historical football results dataset.

The dataset is chronological by construction; we sort by date and map former
team names to their current names so ratings stay attached to one team.
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _load_aliases() -> dict:
    path = RAW_DIR / "former_names.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    aliases = {}
    for _, row in df.iterrows():
        former, current = row.get("former"), row.get("current")
        if pd.notna(former) and pd.notna(current):
            aliases[str(former).strip()] = str(current).strip()
    return aliases


def _apply_aliases(series: pd.Series, aliases: dict) -> pd.Series:
    return series.map(lambda name: aliases.get(str(name).strip(), str(name).strip()))


def load_results() -> pd.DataFrame:
    """Return cleaned matches sorted chronologically with a ``result`` column."""
    df = pd.read_csv(RAW_DIR / "results.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    aliases = _load_aliases()
    df["home_team"] = _apply_aliases(df["home_team"], aliases)
    df["away_team"] = _apply_aliases(df["away_team"], aliases)

    neutral = df["neutral"].astype(str).str.upper()
    df["neutral"] = neutral.map({"TRUE": True, "FALSE": False}).fillna(False)

    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    df["result"] = df.apply(
        lambda r: "H" if r.home_score > r.away_score else ("A" if r.home_score < r.away_score else "D"),
        axis=1,
    )
    return df[["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral", "result"]]
