"""
Task Recommendation Module
DecisionTreeClassifier trained on mood + workload + priority + deadline urgency.

Features:
  - mood         (encoded)
  - workload     (1-10)
  - priority     (Low=1, Medium=2, High=3, Critical=4)
  - deadline_bucket (Relaxed=1: 8+days, Normal=2: 4-7days, Urgent=3: 1-3days)
"""

from __future__ import annotations
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

PRIORITY_MAP   = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}

def _deadline_bucket(days: int) -> int:
    if days <= 3:  return 3  # Urgent
    if days <= 7:  return 2  # Normal
    return 1                 # Relaxed

# ── Training data ─────────────────────────────────────────────────────────────
# Columns: mood, workload(1-10), priority(1-4), deadline_bucket(1-3), task

TRAINING_DATA = [

    # ── HAPPY ─────────────────────────────────────────────────────────────────
    ("Happy", 1,  1, 1, "Creative Work"),
    ("Happy", 2,  1, 1, "Creative Work"),
    ("Happy", 3,  2, 2, "Creative Work"),
    ("Happy", 4,  2, 2, "Collaborative Work"),
    ("Happy", 5,  2, 2, "Collaborative Work"),
    ("Happy", 6,  3, 2, "Collaborative Work"),
    ("Happy", 7,  3, 2, "Strategic Planning"),
    ("Happy", 8,  3, 3, "Strategic Planning"),
    ("Happy", 9,  4, 3, "Deep Work"),
    ("Happy", 10, 4, 3, "Deep Work"),
    # Happy + urgent/critical deadline
    ("Happy", 3,  4, 3, "Deep Work"),
    ("Happy", 4,  4, 3, "Deep Work"),
    ("Happy", 5,  4, 3, "Deep Work"),
    ("Happy", 6,  4, 3, "Strategic Planning"),
    # Happy + relaxed deadline
    ("Happy", 5,  1, 1, "Creative Work"),
    ("Happy", 6,  2, 1, "Creative Work"),

    # ── CALM ──────────────────────────────────────────────────────────────────
    ("Calm",  1,  1, 1, "Deep Work"),
    ("Calm",  2,  1, 1, "Deep Work"),
    ("Calm",  3,  2, 2, "Deep Work"),
    ("Calm",  4,  2, 2, "Deep Work"),
    ("Calm",  5,  2, 2, "Strategic Planning"),
    ("Calm",  6,  3, 2, "Strategic Planning"),
    ("Calm",  7,  3, 2, "Documentation"),
    ("Calm",  8,  3, 3, "Documentation"),
    ("Calm",  9,  4, 3, "Simple Tasks"),
    ("Calm",  10, 4, 3, "Simple Tasks"),
    # Calm + urgent critical
    ("Calm",  3,  4, 3, "Deep Work"),
    ("Calm",  4,  4, 3, "Deep Work"),
    ("Calm",  5,  4, 3, "Deep Work"),
    ("Calm",  6,  4, 3, "Strategic Planning"),
    # Calm + relaxed low priority
    ("Calm",  5,  1, 1, "Documentation"),
    ("Calm",  6,  1, 1, "Documentation"),

    # ── NEUTRAL ───────────────────────────────────────────────────────────────
    ("Neutral", 1,  1, 1, "Deep Work"),
    ("Neutral", 2,  1, 1, "Deep Work"),
    ("Neutral", 3,  2, 2, "Collaborative Work"),
    ("Neutral", 4,  2, 2, "Collaborative Work"),
    ("Neutral", 5,  2, 2, "Documentation"),
    ("Neutral", 6,  3, 2, "Documentation"),
    ("Neutral", 7,  3, 2, "Simple Tasks"),
    ("Neutral", 8,  3, 3, "Simple Tasks"),
    ("Neutral", 9,  4, 3, "Simple Tasks"),
    ("Neutral", 10, 4, 3, "Relaxation Break"),
    # Neutral + urgent critical
    ("Neutral", 3,  4, 3, "Deep Work"),
    ("Neutral", 4,  4, 3, "Deep Work"),
    ("Neutral", 5,  4, 3, "Strategic Planning"),
    ("Neutral", 6,  4, 3, "Strategic Planning"),
    # Neutral + relaxed low
    ("Neutral", 5,  1, 1, "Documentation"),
    ("Neutral", 6,  1, 1, "Simple Tasks"),

    # ── TIRED ─────────────────────────────────────────────────────────────────
    ("Tired",  1,  1, 1, "Simple Tasks"),
    ("Tired",  2,  1, 1, "Simple Tasks"),
    ("Tired",  3,  2, 2, "Simple Tasks"),
    ("Tired",  4,  2, 2, "Simple Tasks"),
    ("Tired",  5,  2, 2, "Simple Tasks"),
    ("Tired",  6,  3, 2, "Relaxation Break"),
    ("Tired",  7,  3, 2, "Relaxation Break"),
    ("Tired",  8,  3, 3, "Relaxation Break"),
    ("Tired",  9,  4, 3, "Relaxation Break"),
    ("Tired",  10, 4, 3, "Relaxation Break"),
    # Tired + urgent critical → simple tasks (can't do deep work when tired)
    ("Tired",  3,  4, 3, "Simple Tasks"),
    ("Tired",  4,  4, 3, "Simple Tasks"),
    ("Tired",  5,  4, 3, "Simple Tasks"),
    ("Tired",  6,  4, 3, "Relaxation Break"),

    # ── STRESSED ──────────────────────────────────────────────────────────────
    ("Stressed", 1,  1, 1, "Simple Tasks"),
    ("Stressed", 2,  1, 1, "Simple Tasks"),
    ("Stressed", 3,  2, 2, "Relaxation Break"),
    ("Stressed", 4,  2, 2, "Relaxation Break"),
    ("Stressed", 5,  2, 2, "Relaxation Break"),
    ("Stressed", 6,  3, 2, "Relaxation Break"),
    ("Stressed", 7,  3, 2, "Mindfulness Activity"),
    ("Stressed", 8,  3, 3, "Mindfulness Activity"),
    ("Stressed", 9,  4, 3, "Mindfulness Activity"),
    ("Stressed", 10, 4, 3, "Mindfulness Activity"),
    # Stressed + critical urgent → still simple (protect the employee)
    ("Stressed", 3,  4, 3, "Simple Tasks"),
    ("Stressed", 4,  4, 3, "Simple Tasks"),
    ("Stressed", 5,  4, 3, "Relaxation Break"),
    ("Stressed", 6,  4, 3, "Mindfulness Activity"),

    # ── SAD ───────────────────────────────────────────────────────────────────
    ("Sad",  1,  1, 1, "Simple Tasks"),
    ("Sad",  2,  1, 1, "Simple Tasks"),
    ("Sad",  3,  2, 2, "Simple Tasks"),
    ("Sad",  4,  2, 2, "Relaxation Break"),
    ("Sad",  5,  2, 2, "Relaxation Break"),
    ("Sad",  6,  3, 2, "Relaxation Break"),
    ("Sad",  7,  3, 2, "Mindfulness Activity"),
    ("Sad",  8,  3, 3, "Mindfulness Activity"),
    ("Sad",  9,  4, 3, "Mindfulness Activity"),
    ("Sad",  10, 4, 3, "Mindfulness Activity"),
    # Sad + critical urgent
    ("Sad",  3,  4, 3, "Simple Tasks"),
    ("Sad",  4,  4, 3, "Simple Tasks"),
    ("Sad",  5,  4, 3, "Relaxation Break"),

    # ── ANGRY ─────────────────────────────────────────────────────────────────
    ("Angry", 1,  1, 1, "Relaxation Break"),
    ("Angry", 2,  1, 1, "Relaxation Break"),
    ("Angry", 3,  2, 2, "Relaxation Break"),
    ("Angry", 4,  2, 2, "Mindfulness Activity"),
    ("Angry", 5,  2, 2, "Mindfulness Activity"),
    ("Angry", 6,  3, 2, "Mindfulness Activity"),
    ("Angry", 7,  3, 2, "Mindfulness Activity"),
    ("Angry", 8,  4, 3, "Mindfulness Activity"),
    ("Angry", 9,  4, 3, "Mindfulness Activity"),
    ("Angry", 10, 4, 3, "Mindfulness Activity"),
    # Angry + critical urgent → still mindfulness (angry + pressure = bad outcome)
    ("Angry", 3,  4, 3, "Mindfulness Activity"),
    ("Angry", 4,  4, 3, "Mindfulness Activity"),

    # ── SURPRISED / FEAR / DISGUST ────────────────────────────────────────────
    ("Surprised", 3, 2, 2, "Collaborative Work"),
    ("Surprised", 6, 3, 2, "Documentation"),
    ("Surprised", 9, 4, 3, "Simple Tasks"),
    ("Surprised", 4, 4, 3, "Strategic Planning"),
    ("Fear",      3, 2, 2, "Simple Tasks"),
    ("Fear",      6, 3, 2, "Relaxation Break"),
    ("Fear",      9, 4, 3, "Mindfulness Activity"),
    ("Disgust",   3, 2, 2, "Relaxation Break"),
    ("Disgust",   6, 3, 2, "Mindfulness Activity"),
    ("Disgust",   9, 4, 3, "Mindfulness Activity"),
]

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


class TaskRecommender:
    def __init__(self):
        self._mood_enc = LabelEncoder()
        self._task_enc = LabelEncoder()
        self._model    = DecisionTreeClassifier(random_state=42)
        self._train()

    def _train(self):
        moods     = [r[0] for r in TRAINING_DATA]
        workloads = [r[1] for r in TRAINING_DATA]
        priorities= [r[2] for r in TRAINING_DATA]
        deadlines = [r[3] for r in TRAINING_DATA]
        tasks     = [r[4] for r in TRAINING_DATA]

        mood_enc = self._mood_enc.fit_transform(moods)
        task_enc = self._task_enc.fit_transform(tasks)

        X = np.column_stack([mood_enc, workloads, priorities, deadlines])
        self._model.fit(X, task_enc)

    def recommend(self, mood: str, workload: int,
                  priority: str = "Medium", deadline_days: int = 7) -> dict:
        mood_cap = mood.capitalize()
        if mood_cap not in self._mood_enc.classes_:
            mood_cap = "Neutral"

        p_enc    = PRIORITY_MAP.get(priority.capitalize(), 2)
        d_bucket = _deadline_bucket(int(deadline_days))
        workload = max(1, min(10, int(workload)))

        mood_enc  = self._mood_enc.transform([mood_cap])[0]
        pred_enc  = self._model.predict([[mood_enc, workload, p_enc, d_bucket]])[0]
        task_name = self._task_enc.inverse_transform([pred_enc])[0]

        return {
            "task":         task_name,
            "description":  TASK_DESCRIPTIONS.get(task_name, ""),
            "mood":         mood_cap,
            "workload":     workload,
            "priority":     priority,
            "deadline_days": deadline_days,
        }


_recommender: TaskRecommender | None = None

def get_recommender() -> TaskRecommender:
    global _recommender
    if _recommender is None:
        _recommender = TaskRecommender()
    return _recommender

def recommend_task(mood: str, workload: int,
                   priority: str = "Medium", deadline_days: int = 7) -> dict:
    return get_recommender().recommend(mood, workload, priority, deadline_days)
