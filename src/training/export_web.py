"""Export the trained linear model + team state to JSON for the static web page.

The static page re-implements multinomial logistic regression in JavaScript, so
it reproduces the trained model's probabilities exactly without a backend.
"""
import json
from pathlib import Path

import joblib

from src.features.build import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"


def main() -> None:
    model = joblib.load(MODELS_DIR / "best_model.joblib")

    if hasattr(model, "named_steps"):
        scaler = model.named_steps.get("standardscaler")
        estimator = model.named_steps.get("logisticregression")
    else:
        scaler = None
        estimator = model

    if estimator is None or not hasattr(estimator, "coef_"):
        raise SystemExit(
            "The saved best model is not a linear (logistic regression) model; "
            "the static page only supports exporting linear coefficients."
        )

    mean = scaler.mean_.tolist() if scaler is not None else [0.0] * len(FEATURE_COLUMNS)
    scale = scaler.scale_.tolist() if scaler is not None else [1.0] * len(FEATURE_COLUMNS)
    if not all(scale):
        scale = [s if s else 1.0 for s in scale]

    state = json.loads((MODELS_DIR / "team_state.json").read_text(encoding="utf-8"))

    payload = {
        "model": {
            "type": "multinomial_logistic_regression",
            "classes": list(estimator.classes_),
            "coef": estimator.coef_.tolist(),
            "intercept": estimator.intercept_.tolist(),
            "scaler_mean": mean,
            "scaler_scale": scale,
            "feature_columns": FEATURE_COLUMNS,
        },
        "teams": state,
    }

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    target = DOCS_DIR / "predictor_data.json"
    target.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {target} ({len(state)} teams, {len(estimator.coef_)} classes)")


if __name__ == "__main__":
    main()
