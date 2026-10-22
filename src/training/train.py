"""Train and evaluate the 1X2 models on a chronological train/test split."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.load import load_results
from src.features.build import FEATURE_COLUMNS, build_features, current_team_state

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"


def chronological_split(features, ratio: float = 0.8):
    cut = int(len(features) * ratio)
    return features.iloc[:cut].copy(), features.iloc[cut:].copy()


def multiclass_brier(y, proba, classes) -> float:
    index = {c: i for i, c in enumerate(classes)}
    onehot = np.zeros_like(proba)
    for row, label in enumerate(y):
        onehot[row, index[label]] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def evaluate(model, X, y) -> dict:
    classes = list(model.classes_)
    proba = model.predict_proba(X)
    pred = model.predict(X)
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "log_loss": round(float(log_loss(y, proba, labels=classes)), 4),
        "brier": round(multiclass_brier(y, proba, classes), 4),
        "classes": classes,
        "n": int(len(y)),
        "report": classification_report(y, pred, labels=classes, output_dict=True, zero_division=0),
    }


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results ...")
    df = load_results()
    print(f"  matches: {len(df)}  ({df['date'].min().date()} -> {df['date'].max().date()})")

    print("Building features (Elo + form, chronological) ...")
    features = build_features(df)
    features.to_csv(PROCESSED_DIR / "features.csv", index=False)

    train, test = chronological_split(features)
    print(f"  train: {len(train)}  test: {len(test)}  (chronological 80/20)")
    print(f"  test range: {test['date'].min().date()} -> {test['date'].max().date()}")

    X_train = train[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train["result"].to_numpy()
    X_test = test[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test = test["result"].to_numpy()

    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0),
        ),
        "logistic_regression_calibrated": CalibratedClassifierCV(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
            method="sigmoid",
            cv=5,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
    }

    results, fitted = {}, {}
    for name, model in candidates.items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[name] = metrics
        fitted[name] = model
        print(f"  accuracy={metrics['accuracy']}  log_loss={metrics['log_loss']}  brier={metrics['brier']}")

    # Probabilistic quality (log loss) matters more than raw accuracy for a
    # predictions product, so the best model is chosen by lowest log loss.
    best_name = min(results, key=lambda n: results[n]["log_loss"])
    print(f"Best model (lowest log loss): {best_name}")

    joblib.dump(fitted[best_name], MODELS_DIR / "best_model.joblib")
    (MODELS_DIR / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2))
    (MODELS_DIR / "team_state.json").write_text(json.dumps(current_team_state(df), indent=2))

    summary = {
        "dataset": {
            "matches": int(len(df)),
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
        },
        "split": {
            "strategy": "chronological 80/20",
            "train_matches": int(len(train)),
            "test_matches": int(len(test)),
            "test_date_min": str(test["date"].min().date()),
            "test_date_max": str(test["date"].max().date()),
        },
        "features": FEATURE_COLUMNS,
        "models": {name: {k: v for k, v in m.items() if k != "report"} for name, m in results.items()},
        "best_model": best_name,
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Results (chronological test set) ===")
    for name, metrics in results.items():
        print(f"{name:24s} acc={metrics['accuracy']:.4f}  log_loss={metrics['log_loss']:.4f}  brier={metrics['brier']:.4f}")
    print(f"\nSaved: models/best_model.joblib ({best_name}), metrics.json, team_state.json")


if __name__ == "__main__":
    main()
