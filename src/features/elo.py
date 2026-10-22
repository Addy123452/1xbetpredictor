"""Chronological Elo ratings.

Ratings are updated match-by-match in date order, so the Elo a team carries into
a match reflects only results *before* that match (no future leakage).
"""

BASE_ELO = 1500.0
HOME_ADVANTAGE = 100.0
FRIENDLY_K = 10.0
DEFAULT_K = 20.0
CONTINENTAL_K = 30.0
WORLD_CUP_K = 40.0


def expected_score(elo_a: float, elo_b: float) -> float:
    """Expected score of A against B (0..1)."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def k_factor(tournament: str) -> float:
    t = str(tournament).lower()
    if "world cup" in t and "qualif" not in t:
        return WORLD_CUP_K
    if "qualif" in t or any(x in t for x in ("euro", "copa", "africa", "asian", "concacaf", "oceania")):
        return CONTINENTAL_K
    if "friendly" in t:
        return FRIENDLY_K
    return DEFAULT_K


def goal_multiplier(goal_diff: int) -> float:
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return 1.75 + (gd - 3) / 8.0


def update_elo(home: float, away: float, home_score: int, away_score: int,
               tournament: str, neutral: bool) -> tuple[float, float]:
    """Return ``(new_home, new_away)`` after a match. Rating change is zero-sum."""
    advantage = 0.0 if neutral else HOME_ADVANTAGE
    expected_home = expected_score(home + advantage, away)
    if home_score > away_score:
        actual = 1.0
    elif home_score < away_score:
        actual = 0.0
    else:
        actual = 0.5

    k = k_factor(tournament) * goal_multiplier(home_score - away_score)
    delta = k * (actual - expected_home)
    return home + delta, away - delta
