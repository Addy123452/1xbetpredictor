"""Rolling team-form tracker updated only from prior matches."""
from collections import defaultdict, deque

DEFAULT_WINDOW = 5
DECAY = 0.75  # older matches count less


def _weighted_avg(values, decay: float = DECAY):
    """Exponentially weight recent values more heavily (newest = weight 1)."""
    n = len(values)
    if n == 0:
        return None
    weights = [decay ** (n - 1 - i) for i in range(n)]
    total = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / total


class FormTracker:
    """Tracks recent points, goal difference, and home/away points per team."""

    def __init__(self, window: int = DEFAULT_WINDOW):
        self.window = window
        self._recent = defaultdict(lambda: deque(maxlen=self.window))
        self._recent_gd = defaultdict(lambda: deque(maxlen=self.window))
        self._home_pts = defaultdict(lambda: deque(maxlen=self.window))
        self._away_pts = defaultdict(lambda: deque(maxlen=self.window))

    def features(self, team: str) -> dict:
        pts = list(self._recent[team])
        gd = list(self._recent_gd[team])
        return {
            "form_ppg": _weighted_avg(pts) if pts else 1.0,
            "form_gd": _weighted_avg(gd) if gd else 0.0,
            "form_samples": len(pts),
        }

    def home_ppg(self, team: str) -> float:
        pts = list(self._home_pts[team])
        return _weighted_avg(pts) if pts else 1.0

    def away_ppg(self, team: str) -> float:
        pts = list(self._away_pts[team])
        return _weighted_avg(pts) if pts else 1.0

    def update(self, home: str, away: str, home_score: int, away_score: int) -> None:
        if home_score > away_score:
            home_points, away_points = 3, 0
        elif home_score < away_score:
            home_points, away_points = 0, 3
        else:
            home_points, away_points = 1, 1

        self._recent[home].append(home_points)
        self._recent[away].append(away_points)
        self._recent_gd[home].append(home_score - away_score)
        self._recent_gd[away].append(away_score - home_score)
        self._home_pts[home].append(home_points)
        self._away_pts[away].append(away_points)
