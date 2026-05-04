"""
Burnout Risk Score Module

Hybrid scoring:
  - ML model (GradientBoostingRegressor trained on Kaggle burnout dataset): 60%
  - Behavioural formula (5 weighted factors from our collected data): 40%

If ML model is not available (not yet trained), falls back to 100% formula.

Behavioural factors & weights (formula component):
  1. Recent average stress (last 14 readings)  — 30%
  2. Consecutive high-stress readings          — 25%
  3. Stress trend (rising slope)               — 20%
  4. Negative mood ratio                       — 15%
  5. Missed check-ins (last 7 days)            — 10%
"""

from __future__ import annotations
import os
import numpy as np
from datetime import date, timedelta
from database import get_mood_history, get_attendance_history, get_task_history

NEGATIVE_MOODS         = {"stressed", "angry", "sad", "fear", "disgust", "tired"}
HIGH_STRESS_THRESHOLD  = 6
CONSECUTIVE_MAX        = 5
TREND_WINDOW           = 10

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "burnout_model.pkl")
META_PATH  = os.path.join(os.path.dirname(__file__), "models", "burnout_model_meta.pkl")

_ml_model = None
_ml_meta  = None


def _load_ml_model():
    global _ml_model, _ml_meta
    if _ml_model is not None:
        return _ml_model, _ml_meta
    if not os.path.exists(MODEL_PATH):
        return None, None
    try:
        import joblib
        _ml_model = joblib.load(MODEL_PATH)
        _ml_meta  = joblib.load(META_PATH) if os.path.exists(META_PATH) else {}
        return _ml_model, _ml_meta
    except Exception:
        return None, None


def get_model_meta() -> dict:
    """Return training metadata for display in the dashboard."""
    _, meta = _load_ml_model()
    return meta or {}


def _avg_workload(user_id: int) -> float:
    """Compute average workload from recent task logs; default 5.0 if none."""
    try:
        tasks = get_task_history(user_id)
        if tasks:
            workloads = [t["workload"] for t in tasks if t.get("workload")]
            if workloads:
                return float(np.mean(workloads[-14:]))
    except Exception:
        pass
    return 5.0


def _ml_score(avg_stress: float, avg_wkl: float) -> float | None:
    """
    Run ML model. Inputs match Kaggle features:
      Mental Fatigue Score = avg_stress  (0-10)
      Resource Allocation  = avg_wkl     (1-10)
    Returns burn rate 0-1 → converted to 0-100, or None if model not ready.
    """
    model, _ = _load_ml_model()
    if model is None:
        return None
    try:
        pred = model.predict([[avg_stress, avg_wkl]])[0]
        return float(np.clip(pred * 100, 0, 100))
    except Exception:
        return None


def _formula_score(stress_levels: list, moods: list, attendance: list) -> tuple[float, dict]:
    """Pure weighted-formula score. Returns (score 0-100, factors dict)."""

    # F1: Recent average stress
    recent_stress = stress_levels[:min(14, len(stress_levels))]
    avg_stress    = float(np.mean(recent_stress))
    f1            = (avg_stress / 10) * 100

    # F2: Consecutive high-stress readings
    consecutive = 0
    for lvl in stress_levels:
        if lvl >= HIGH_STRESS_THRESHOLD:
            consecutive += 1
        else:
            break
    f2 = min(consecutive / CONSECUTIVE_MAX, 1.0) * 100

    # F3: Stress trend (linear slope)
    window = stress_levels[:TREND_WINDOW]
    if len(window) >= 3:
        x     = np.arange(len(window), 0, -1, dtype=float)
        slope = float(np.polyfit(x, window, 1)[0])
        f3    = max(0.0, min(slope * 25, 100.0))
    else:
        f3 = 0.0

    # F4: Negative mood ratio
    recent_moods = moods[:min(14, len(moods))]
    neg_count    = sum(1 for m in recent_moods if m in NEGATIVE_MOODS)
    f4           = (neg_count / len(recent_moods)) * 100 if recent_moods else 0.0

    # F5: Missed check-ins (last 7 calendar days)
    checked_dates = {a["date"] for a in attendance}
    today         = date.today()
    missed        = sum(
        1 for i in range(1, 8)
        if (today - timedelta(days=i)).isoformat() not in checked_dates
    )
    f5 = (missed / 7) * 100

    score = (
        f1 * 0.30 + f2 * 0.25 + f3 * 0.20 + f4 * 0.15 + f5 * 0.10
    )

    factors = {
        "avg_stress":  round(f1, 1),
        "consecutive": round(f2, 1),
        "trend":       round(f3, 1),
        "neg_mood":    round(f4, 1),
        "missed_days": round(f5, 1),
    }
    return float(score), factors


def calculate_burnout_risk(user_id: int) -> dict:
    mood_logs  = get_mood_history(user_id, limit=50)
    attendance = get_attendance_history(user_id, limit=14)

    if not mood_logs:
        return _no_data_result()

    stress_levels = [m["stress_level"] for m in mood_logs]
    moods         = [m["mood"].lower()  for m in mood_logs]

    # ── Formula component ─────────────────────────────────────────────────────
    formula_score, factors = _formula_score(stress_levels, moods, attendance)
    avg_stress_raw         = float(np.mean(stress_levels[:14]))
    consecutive_raw        = sum(
        1 for _ in
        __import__("itertools").takewhile(lambda l: l >= HIGH_STRESS_THRESHOLD, stress_levels)
    )
    missed_raw = sum(
        1 for i in range(1, 8)
        if (date.today() - timedelta(days=i)).isoformat()
           not in {a["date"] for a in attendance}
    )

    # ── ML component ──────────────────────────────────────────────────────────
    avg_wkl  = _avg_workload(user_id)
    ml_score = _ml_score(avg_stress_raw, avg_wkl)

    if ml_score is not None:
        final_score = ml_score * 0.60 + formula_score * 0.40
        ml_used     = True
    else:
        final_score = formula_score
        ml_used     = False

    score = int(round(min(max(final_score, 0), 100)))

    return {
        "score":            score,
        "label":            _label(score),
        "color":            _color(score),
        "factors":          factors,
        "data_points":      len(mood_logs),
        "avg_stress_raw":   round(avg_stress_raw, 1),
        "consecutive_raw":  consecutive_raw,
        "missed_raw":       missed_raw,
        "ml_used":          ml_used,
        "ml_score":         round(ml_score, 1) if ml_score is not None else None,
        "formula_score":    round(formula_score, 1),
    }


def bulk_burnout_risk(users: list) -> list:
    results = []
    for user in users:
        risk = calculate_burnout_risk(user["id"])
        results.append({
            "id":          user["id"],
            "name":        user["name"],
            "employee_id": user["employee_id"],
            "department":  user["department"],
            **risk,
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def _label(score: int) -> str:
    if score >= 75: return "Critical Risk"
    if score >= 50: return "High Risk"
    if score >= 25: return "Moderate Risk"
    return "Low Risk"


def _color(score: int) -> str:
    if score >= 75: return "#e74c3c"
    if score >= 50: return "#e67e22"
    if score >= 25: return "#f1c40f"
    return "#2ecc71"


def _no_data_result() -> dict:
    return {
        "score":           0,
        "label":           "No Data",
        "color":           "#95a5a6",
        "factors":         {},
        "data_points":     0,
        "avg_stress_raw":  0,
        "consecutive_raw": 0,
        "missed_raw":      0,
        "ml_used":         False,
        "ml_score":        None,
        "formula_score":   0,
    }