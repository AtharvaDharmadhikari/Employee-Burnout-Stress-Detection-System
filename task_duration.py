"""
Task Duration Prediction Module
Predicts estimated task duration (in hours) given:
  - task_description (text → TF-IDF features)
  - priority         (Low / Medium / High / Critical)
  - days_until_deadline
  - workload         (1-10)

Uses a GradientBoostingRegressor trained on synthetic but realistic data.
"""

from __future__ import annotations
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")

PRIORITY_OPTIONS = ["Low", "Medium", "High", "Critical"]

# ── Synthetic training data ───────────────────────────────────────────────────
# (task_description, priority, days_until_deadline, workload, duration_hours)

TRAINING_DATA = [
    ("Write unit tests for authentication module", "High",     3,  7,  4.0),
    ("Design database schema for new feature",     "High",     5,  6,  6.0),
    ("Review pull request",                        "Medium",   1,  4,  1.5),
    ("Fix critical bug in payment gateway",        "Critical", 1,  9,  8.0),
    ("Update API documentation",                   "Low",      7,  3,  2.0),
    ("Implement user dashboard",                   "High",     7,  6,  12.0),
    ("Refactor legacy code",                       "Medium",   10, 5,  8.0),
    ("Create weekly report",                       "Low",      2,  2,  1.0),
    ("Deploy to production server",                "Critical", 1,  8,  3.0),
    ("Research new ML frameworks",                 "Low",      14, 4,  5.0),
    ("Fix UI alignment issues",                    "Low",      3,  2,  1.5),
    ("Implement OAuth2 login",                     "High",     5,  7,  8.0),
    ("Write technical specification",              "Medium",   7,  5,  4.0),
    ("Conduct code review session",                "Medium",   2,  4,  2.0),
    ("Optimize database queries",                  "High",     4,  6,  6.0),
    ("Setup CI/CD pipeline",                       "High",     5,  7,  10.0),
    ("Respond to customer support tickets",        "Medium",   1,  3,  2.0),
    ("Plan sprint retrospective",                  "Low",      3,  3,  1.0),
    ("Integrate third-party payment API",          "High",     6,  7,  10.0),
    ("Analyze user feedback data",                 "Medium",   5,  5,  4.0),
    ("Build recommendation engine",                "High",     14, 8,  20.0),
    ("Fix memory leak in backend service",         "Critical", 2,  8,  6.0),
    ("Create onboarding tutorial",                 "Medium",   7,  4,  5.0),
    ("Migrate to new cloud provider",              "Critical", 10, 9,  30.0),
    ("Write blog post about new feature",          "Low",      7,  2,  3.0),
    ("Implement push notifications",               "Medium",   5,  5,  6.0),
    ("Perform security audit",                     "High",     7,  7,  8.0),
    ("Update dependencies",                        "Low",      5,  2,  1.0),
    ("Build admin dashboard",                      "High",     10, 7,  15.0),
    ("Create automated test suite",                "High",     7,  6,  12.0),
    ("Review",                                     "Low",      1,  2,  0.5),
    ("Meeting",                                    "Low",      1,  3,  1.0),
    ("Debug production issue",                     "Critical", 1,  9,  5.0),
    ("Write user story",                           "Low",      2,  2,  0.5),
    ("Implement caching layer",                    "Medium",   5,  5,  6.0),
    ("Data analysis and reporting",                "Medium",   5,  5,  5.0),
    ("Design system architecture",                 "High",     7,  7,  10.0),
    ("Train ML model",                             "High",     7,  8,  15.0),
    ("Handle customer escalation",                 "Critical", 1,  9,  3.0),
    ("Create wireframes",                          "Medium",   3,  4,  4.0),
]


# ── Model ─────────────────────────────────────────────────────────────────────

class DurationPredictor:
    def __init__(self):
        self._priority_enc = LabelEncoder()
        self._tfidf = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
        self._model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
        )
        self._train()

    def _build_features(self, descriptions, priorities, deadlines, workloads,
                        fit=False):
        if fit:
            text_feats = self._tfidf.fit_transform(descriptions).toarray()
            priority_enc = self._priority_enc.fit_transform(priorities)
        else:
            text_feats = self._tfidf.transform(descriptions).toarray()
            # Handle unseen priority labels
            safe_priorities = [
                p if p in self._priority_enc.classes_ else "Medium"
                for p in priorities
            ]
            priority_enc = self._priority_enc.transform(safe_priorities)

        numeric = np.column_stack([
            priority_enc,
            np.array(deadlines, dtype=float),
            np.array(workloads, dtype=float),
        ])
        return np.hstack([text_feats, numeric])

    def _train(self):
        descs      = [r[0] for r in TRAINING_DATA]
        priorities = [r[1] for r in TRAINING_DATA]
        deadlines  = [r[2] for r in TRAINING_DATA]
        workloads  = [r[3] for r in TRAINING_DATA]
        durations  = [r[4] for r in TRAINING_DATA]

        X = self._build_features(descs, priorities, deadlines, workloads, fit=True)
        self._model.fit(X, durations)

    def predict(self, task_description: str, priority: str,
                days_until_deadline: int, workload: int = 5) -> dict:
        """
        Predict task duration in hours.
        Returns dict with prediction and a human-readable label.
        """
        priority = priority.capitalize()
        days_until_deadline = max(1, int(days_until_deadline))
        workload = max(1, min(10, int(workload)))

        X = self._build_features(
            [task_description], [priority],
            [days_until_deadline], [workload],
            fit=False
        )
        raw = float(self._model.predict(X)[0])
        hours = max(0.5, round(raw * 2) / 2)  # round to nearest 0.5h

        return {
            "estimated_hours": hours,
            "label": _hours_label(hours),
            "task_description": task_description,
            "priority": priority,
            "days_until_deadline": days_until_deadline,
        }


def _hours_label(hours: float) -> str:
    if hours < 1:
        return f"{int(hours * 60)} minutes"
    elif hours == 1:
        return "1 hour"
    elif hours < 8:
        return f"{hours:.1f} hours"
    elif hours < 16:
        return f"{hours:.0f} hours (~1 day)"
    else:
        days = hours / 8
        return f"{hours:.0f} hours (~{days:.1f} days)"


# Singleton
_predictor: DurationPredictor | None = None


def get_predictor() -> DurationPredictor:
    global _predictor
    if _predictor is None:
        _predictor = DurationPredictor()
    return _predictor


def predict_task_duration(task_description: str, priority: str,
                          days_until_deadline: int, workload: int = 5) -> dict:
    """Convenience wrapper."""
    return get_predictor().predict(task_description, priority,
                                   days_until_deadline, workload)
