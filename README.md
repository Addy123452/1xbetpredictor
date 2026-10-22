# 🏆 Sports Outcome AI

A machine learning project designed to predict the possible outcome of a
sports match using historical match data and sport-specific features.

> **Current version: v0.2 --- Elo + Form Predictor**

## 📌 Project Goal

The long-term goal of this project is to build a universal sports
prediction system that can estimate the probabilities of possible
outcomes for an ongoing match.

Examples:

-   Football → Home Win / Draw / Away Win
-   Cricket → Team A Win / Team B Win
-   Basketball → Team A Win / Team B Win
-   Tennis → Player A Win / Player B Win

The system will eventually use live match information and sport-specific
machine learning models.

------------------------------------------------------------------------

## 🚀 Current Version

The current version focuses on **football pre-match prediction** using
historical results.

### Current pipeline

``` text
Historical Match Data
        ↓
Data Cleaning (name aliases, chronology)
        ↓
Feature Engineering (Elo ratings + time-weighted form)
        ↓
Chronological Train/Test Split (80/20)
        ↓
Model Comparison (logistic regression, calibrated LR, gradient boosting, random forest)
        ↓
Match Outcome Prediction
        ↓
Model Evaluation (accuracy, log loss, Brier, classification report)
```

------------------------------------------------------------------------

## 📊 Dataset

The current dataset contains historical football match results.

Available information includes:

-   Date
-   Home team
-   Away team
-   Home score
-   Away score
-   Tournament
-   City
-   Country
-   Neutral venue

The dataset currently contains approximately **49,520 matches**.

### Important limitation

The current dataset does **not** contain live match statistics such as:

-   Current minute
-   Current score during the match
-   Possession
-   Shots
-   Shots on target
-   Corners
-   Cards
-   Expected goals (xG)

Therefore, the current model is a **pre-match baseline**, not yet a live
prediction system.

------------------------------------------------------------------------

## 🧠 Features

The current model (v0.2) uses pre-match information only:

-   Elo rating for the home and away team (computed chronologically)
-   Difference in Elo ratings
-   Time-weighted recent form (last 5 matches, exponentially decayed)
-   Recent goal difference (time-weighted)
-   Home- vs away-specific points per game
-   Opponent strength (opponent's pre-match Elo)
-   Neutral venue flag

The model is designed to avoid using the current match's final result
when creating predictive features.

------------------------------------------------------------------------

## 🤖 Machine Learning Model

### Baseline Model

**Algorithm:** Logistic Regression

**Target:**

``` text
H = Home Win
D = Draw
A = Away Win
```

### Train/Test Strategy

Because sports data is time-dependent, the dataset is split
chronologically:

``` text
Older 80% → Training
Newer 20% → Testing
```

This avoids randomly mixing past and future matches and provides a more
realistic evaluation.

------------------------------------------------------------------------

## 📈 v0.2 Results

Evaluated on the **chronological 20% hold-out** (last 9,904 matches,
2016-03-29 → 2026-07-15). Features use only information available before each
match, and the split never mixes future matches into training.

| Model | Accuracy | Log loss | Brier |
| --- | --- | --- | --- |
| Logistic regression | **0.5963** | **0.8802** | **0.5175** |
| Logistic regression (sigmoid-calibrated) | 0.5958 | 0.8912 | 0.5231 |
| Histogram gradient boosting | 0.5958 | 0.8851 | 0.5201 |
| Random forest | 0.5937 | 0.8930 | 0.5248 |

The selected model is the **uncalibrated logistic regression** (lowest log
loss). Explicit sigmoid calibration was tested but did **not** improve log loss
on this split, so it was not adopted — recorded here rather than silently
omitted.

Draws remain the hardest class to predict (the model favours home/away
outcomes), which matches the known difficulty of football draws.

## 🚀 How to run

```powershell
python -m pip install -r requirements.txt

# Train, evaluate, and write models/ (model, metrics, team state)
python -m src.training.train

# Predict a match
python -m src.prediction.predict --home "Brazil" --away "Argentina"
python -m src.prediction.predict --home "France" --away "England" --neutral

# Run the feature tests
python -m tests.test_features
```

------------------------------------------------------------------------

## 🗂️ Project Structure

``` text
sports-outcome-ai/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── training/
│   └── prediction/
│
├── models/
│
├── tests/
│
├── README.md
└── .gitignore
```

### Folder purposes

**`data/raw/`**\
Original datasets.

**`data/processed/`**\
Cleaned and feature-engineered datasets.

**`notebooks/`**\
Data exploration and experiments.

**`src/`**\
Main project source code.

**`models/`**\
Saved machine learning models.

**`tests/`**\
Automated tests.

------------------------------------------------------------------------

## 🛠️ Technology Stack

Current:

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Jupyter Notebook
-   Git
-   GitHub

Planned:

-   XGBoost / LightGBM
-   FastAPI
-   PostgreSQL
-   React
-   Docker
-   MLflow
-   Live sports API

------------------------------------------------------------------------

## 🗺️ Roadmap

### Version 0.1 --- Completed ✅

-   [x] Project setup
-   [x] Historical football dataset
-   [x] Data exploration
-   [x] Match outcome target
-   [x] Recent-form features
-   [x] Historical team-strength features
-   [x] Chronological train/test split
-   [x] Logistic Regression baseline
-   [x] Model evaluation

### Version 0.2 --- Completed ✅

-   [x] Elo ratings
-   [x] Goal-difference features
-   [x] Home/away performance
-   [x] Opponent strength
-   [x] Time-weighted form
-   [x] Better probability calibration (tested; not adopted — see Results)
-   [x] Compare multiple ML algorithms

### Version 0.3 --- Live Football Prediction

-   [ ] Find suitable live football data source
-   [ ] Live match-state features
-   [ ] Current score
-   [ ] Match minute
-   [ ] Shots
-   [ ] Possession
-   [ ] Corners
-   [ ] Cards
-   [ ] Live probability updates

### Version 0.4 --- Prediction API

-   [ ] FastAPI backend
-   [ ] `/predict` endpoint
-   [ ] `/matches` endpoint
-   [ ] Model loading service
-   [ ] API testing

### Version 0.5 --- Web Dashboard

-   [ ] React frontend
-   [ ] Live match dashboard
-   [ ] Win probability graph
-   [ ] Match timeline
-   [ ] Prediction explanations

### Version 1.0 --- Multi-Sport AI

-   [ ] Football
-   [ ] Cricket
-   [ ] Basketball
-   [ ] Tennis
-   [ ] Sport-specific feature engineering
-   [ ] Sport-specific models
-   [ ] Universal prediction engine

------------------------------------------------------------------------

## ⚠️ Important Disclaimer

This project is an educational machine learning and sports analytics
project.

Predictions are probabilistic estimates and are not guaranteed outcomes.
The system should not be treated as a reliable method for gambling or
financial decision-making.

------------------------------------------------------------------------

## 👨‍💻 Development Philosophy

The project is being developed incrementally:

``` text
Build
 ↓
Measure
 ↓
Understand
 ↓
Improve
 ↓
Evaluate
 ↓
Deploy
```

The goal is not simply to maximize accuracy, but to build a reliable,
explainable, time-aware sports prediction system.

------------------------------------------------------------------------

## 📄 License

License to be added when the project reaches a stable public release.
