"""
Task Recommendation Module
Uses a DecisionTreeClassifier trained on mood + workload to recommend a task type.
The model is trained on a built-in dataset at import time (no external data file needed).
"""

from __future__ import annotations
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ── Training data ─────────────────────────────────────────────────────────────
# Columns: mood, workload (1-10), recommended_task

TRAINING_DATA = [
    # Happy
    ("Happy",    1,  "Creative Work"),
    ("Happy",    2,  "Creative Work"),
    ("Happy",    3,  "Creative Work"),
    ("Happy",    4,  "Collaborative Work"),
    ("Happy",    5,  "Collaborative Work"),
    ("Happy",    6,  "Collaborative Work"),
    ("Happy",    7,  "Strategic Planning"),
    ("Happy",    8,  "Strategic Planning"),
    ("Happy",    9,  "Strategic Planning"),
    ("Happy",    10, "Strategic Planning"),
    # Calm
    ("Calm",     1,  "Deep Work"),
    ("Calm",     2,  "Deep Work"),
    ("Calm",     3,  "Deep Work"),
    ("Calm",     4,  "Deep Work"),
    ("Calm",     5,  "Strategic Planning"),
    ("Calm",     6,  "Strategic Planning"),
    ("Calm",     7,  "Documentation"),
    ("Calm",     8,  "Documentation"),
    ("Calm",     9,  "Documentation"),
    ("Calm",     10, "Simple Tasks"),
    # Neutral
    ("Neutral",  1,  "Deep Work"),
    ("Neutral",  2,  "Deep Work"),
    ("Neutral",  3,  "Collaborative Work"),
    ("Neutral",  4,  "Collaborative Work"),
    ("Neutral",  5,  "Documentation"),
    ("Neutral",  6,  "Documentation"),
    ("Neutral",  7,  "Simple Tasks"),
    ("Neutral",  8,  "Simple Tasks"),
    ("Neutral",  9,  "Simple Tasks"),
    ("Neutral",  10, "Relaxation Break"),
    # Tired
    ("Tired",    1,  "Simple Tasks"),
    ("Tired",    2,  "Simple Tasks"),
    ("Tired",    3,  "Simple Tasks"),
    ("Tired",    4,  "Simple Tasks"),
    ("Tired",    5,  "Simple Tasks"),
    ("Tired",    6,  "Relaxation Break"),
    ("Tired",    7,  "Relaxation Break"),
    ("Tired",    8,  "Relaxation Break"),
    ("Tired",    9,  "Relaxation Break"),
    ("Tired",    10, "Relaxation Break"),
    # Stressed
    ("Stressed", 1,  "Simple Tasks"),
    ("Stressed", 2,  "Simple Tasks"),
    ("Stressed", 3,  "Relaxation Break"),
    ("Stressed", 4,  "Relaxation Break"),
    ("Stressed", 5,  "Relaxation Break"),
    ("Stressed", 6,  "Relaxation Break"),
    ("Stressed", 7,  "Mindfulness Activity"),
    ("Stressed", 8,  "Mindfulness Activity"),
    ("Stressed", 9,  "Mindfulness Activity"),
    ("Stressed", 10, "Mindfulness Activity"),
    # Sad
    ("Sad",      1,  "Simple Tasks"),
    ("Sad",      2,  "Simple Tasks"),
    ("Sad",      3,  "Simple Tasks"),
    ("Sad",      4,  "Relaxation Break"),
    ("Sad",      5,  "Relaxation Break"),
    ("Sad",      6,  "Relaxation Break"),
    ("Sad",      7,  "Mindfulness Activity"),
    ("Sad",      8,  "Mindfulness Activity"),
    ("Sad",      9,  "Mindfulness Activity"),
    ("Sad",      10, "Mindfulness Activity"),
    # Angry
    ("Angry",    1,  "Relaxation Break"),
    ("Angry",    2,  "Relaxation Break"),
    ("Angry",    3,  "Relaxation Break"),
    ("Angry",    4,  "Mindfulness Activity"),
    ("Angry",    5,  "Mindfulness Activity"),
    ("Angry",    6,  "Mindfulness Activity"),
    ("Angry",    7,  "Mindfulness Activity"),
    ("Angry",    8,  "Mindfulness Activity"),
    ("Angry",    9,  "Mindfulness Activity"),
    ("Angry",    10, "Mindfulness Activity"),
    # Surprise / Fear / Disgust
    ("Surprise", 3,  "Collaborative Work"),
    ("Surprise", 6,  "Documentation"),
    ("Surprise", 9,  "Simple Tasks"),
    ("Fear",     3,  "Simple Tasks"),
    ("Fear",     6,  "Relaxation Break"),
    ("Fear",     9,  "Mindfulness Activity"),
    ("Disgust",  3,  "Relaxation Break"),
    ("Disgust",  6,  "Mindfulness Activity"),
    ("Disgust",  9,  "Mindfulness Activity"),
]

# Task descriptions shown to the user
TASK_DESCRIPTIONS: dict[str, str] = {
    "Creative Work":       "Brainstorming, design, writing, or innovation tasks",
    "Collaborative Work":  "Team meetings, pair programming, or group projects",
    "Strategic Planning":  "Goal setting, roadmapping, or high-level analysis",
    "Deep Work":           "Focused coding, research, or complex problem-solving",
    "Documentation":       "Writing docs, updating wikis, or creating reports",
    "Simple Tasks":        "Emails, admin tasks, or low-cognitive-load work",
    "Relaxation Break":    "Short break — walk, stretch, or breathing exercise",
    "Mindfulness Activity":"Guided meditation, journaling, or talking to a mentor",
}


# ── Model ─────────────────────────────────────────────────────────────────────

class TaskRecommender:
    def __init__(self):
        self._mood_enc = LabelEncoder()
        self._task_enc = LabelEncoder()
        self._model = DecisionTreeClassifier(random_state=42)
        self._train()

    def _train(self):
        moods   = [row[0] for row in TRAINING_DATA]
        loads   = [row[1] for row in TRAINING_DATA]
        tasks   = [row[2] for row in TRAINING_DATA]

        mood_encoded = self._mood_enc.fit_transform(moods)
        task_encoded = self._task_enc.fit_transform(tasks)

        X = np.column_stack([mood_encoded, loads])
        self._model.fit(X, task_encoded)

    def recommend(self, mood: str, workload: int) -> dict:
        """
        Returns recommended task info for a given mood and workload (1-10).
        """
        mood_cap = mood.capitalize()
        # Handle unseen moods gracefully
        if mood_cap not in self._mood_enc.classes_:
            mood_cap = "Neutral"

        mood_enc = self._mood_enc.transform([mood_cap])[0]
        workload = max(1, min(10, int(workload)))

        pred_enc  = self._model.predict([[mood_enc, workload]])[0]
        task_name = self._task_enc.inverse_transform([pred_enc])[0]

        return {
            "task": task_name,
            "description": TASK_DESCRIPTIONS.get(task_name, ""),
            "mood": mood_cap,
            "workload": workload,
        }

    def all_tasks(self) -> list[str]:
        return list(TASK_DESCRIPTIONS.keys())


# Singleton — import and reuse
_recommender: TaskRecommender | None = None


def get_recommender() -> TaskRecommender:
    global _recommender
    if _recommender is None:
        _recommender = TaskRecommender()
    return _recommender


def recommend_task(mood: str, workload: int) -> dict:
    """Convenience wrapper around the singleton recommender."""
    return get_recommender().recommend(mood, workload)
