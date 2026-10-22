"""Focused tests for the Elo and form feature logic (no future leakage)."""
from src.features.elo import expected_score, update_elo
from src.features.form import FormTracker


def test_expected_score_equal_teams_is_half():
    assert abs(expected_score(1500, 1500) - 0.5) < 1e-12


def test_elo_change_is_zero_sum():
    home, away = update_elo(1500, 1500, 2, 1, "Friendly", False)
    assert abs((home + away) - 3000) < 1e-9


def test_winner_gains_ratings():
    home, away = update_elo(1500, 1500, 3, 0, "Friendly", False)
    assert home > 1500 > away


def test_form_starts_empty():
    tracker = FormTracker()
    stats = tracker.features("Never Seen")
    assert stats["form_samples"] == 0
    assert stats["form_ppg"] == 1.0


def test_form_uses_only_prior_matches():
    tracker = FormTracker()
    tracker.update("A", "B", 2, 0)
    assert tracker.features("A")["form_ppg"] == 3.0
    assert tracker.features("B")["form_ppg"] == 0.0
    assert tracker.features("A")["form_samples"] == 1


def test_form_window_is_bounded():
    tracker = FormTracker(window=5)
    for _ in range(10):
        tracker.update("A", "B", 1, 0)
    assert tracker.features("A")["form_samples"] == 5


def test_recent_form_is_weighted_more():
    tracker = FormTracker()
    tracker.update("A", "B", 1, 0)  # win -> 3 pts
    tracker.update("A", "C", 0, 1)  # loss -> 0 pts (most recent)
    # Weighted PPG must sit below the simple average (1.5).
    assert tracker.features("A")["form_ppg"] < 1.5


def test_neutral_venue_removes_home_advantage():
    home, away = update_elo(1500, 1500, 0, 0, "Friendly", True)
    assert abs(home - away) < 1e-9


def _run_all():
    count = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            count += 1
            print(f"  PASS {name}")
    print(f"All {count} tests passed.")


if __name__ == "__main__":
    _run_all()
