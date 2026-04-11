"""
Stress Management Alerts Module
Evaluates mood logs and triggers HR alerts when stress thresholds are exceeded.

Thresholds:
  - Single reading  >= STRESS_THRESHOLD        → immediate alert
  - 3 consecutive readings >= CONSECUTIVE_THRESHOLD → burnout alert
  - Average stress over last N readings >= AVG_THRESHOLD → disengagement alert
"""

from __future__ import annotations
from database import (
    log_mood, get_mood_history, create_alert,
    get_unacknowledged_alerts, acknowledge_alert
)

STRESS_THRESHOLD      = 7   # single-reading immediate alert
CONSECUTIVE_THRESHOLD = 6   # 3 consecutive readings at or above this
CONSECUTIVE_COUNT     = 3
AVG_THRESHOLD         = 6.5 # rolling average over last 5 readings
AVG_WINDOW            = 5


AlertResult = dict  # {"triggered": bool, "type": str, "message": str, "level": int}


def evaluate_stress(employee_id: int, employee_name: str,
                    current_stress: int, mood: str) -> list[AlertResult]:
    """
    Evaluate stress for an employee after a new mood log entry.
    Returns a list of alert dicts that were triggered (may be empty).
    """
    alerts: list[AlertResult] = []

    # 1. Immediate single-reading alert
    if current_stress >= STRESS_THRESHOLD:
        msg = (
            f"{employee_name} reported a stress level of {current_stress}/10 "
            f"(mood: {mood}). Immediate attention may be required."
        )
        create_alert(employee_id, "stress", current_stress, msg)
        alerts.append({
            "triggered": True,
            "type": "stress",
            "message": msg,
            "level": current_stress,
        })

    # 2. Consecutive high-stress readings (burnout indicator)
    history = get_mood_history(employee_id, limit=CONSECUTIVE_COUNT)
    if len(history) >= CONSECUTIVE_COUNT:
        recent_levels = [h["stress_level"] for h in history[:CONSECUTIVE_COUNT]]
        if all(lvl >= CONSECUTIVE_THRESHOLD for lvl in recent_levels):
            avg = sum(recent_levels) / len(recent_levels)
            msg = (
                f"{employee_name} has had {CONSECUTIVE_COUNT} consecutive "
                f"high-stress readings (avg {avg:.1f}/10). "
                "This may indicate burnout — please check in with them."
            )
            create_alert(employee_id, "burnout", int(avg), msg)
            alerts.append({
                "triggered": True,
                "type": "burnout",
                "message": msg,
                "level": int(avg),
            })

    # 3. Rolling average disengagement alert
    history_avg = get_mood_history(employee_id, limit=AVG_WINDOW)
    if len(history_avg) >= AVG_WINDOW:
        avg_stress = sum(h["stress_level"] for h in history_avg) / len(history_avg)
        if avg_stress >= AVG_THRESHOLD:
            msg = (
                f"{employee_name}'s average stress over the last {AVG_WINDOW} "
                f"check-ins is {avg_stress:.1f}/10. "
                "Long-term disengagement risk — consider a wellness conversation."
            )
            create_alert(employee_id, "disengagement", int(avg_stress), msg)
            alerts.append({
                "triggered": True,
                "type": "disengagement",
                "message": msg,
                "level": int(avg_stress),
            })

    return alerts


def get_alert_badge_color(alert_type: str) -> str:
    """Return a color string for Streamlit badge rendering."""
    return {
        "stress":        "#e74c3c",
        "burnout":       "#e67e22",
        "disengagement": "#f39c12",
    }.get(alert_type, "#95a5a6")


def alert_emoji(alert_type: str) -> str:
    return {
        "stress":        "🔴",
        "burnout":       "🟠",
        "disengagement": "🟡",
    }.get(alert_type, "⚪")


def stress_level_label(level: int) -> str:
    if level <= 3:
        return "Low"
    elif level <= 6:
        return "Moderate"
    elif level <= 8:
        return "High"
    else:
        return "Critical"


def stress_color(level: int) -> str:
    if level <= 3:
        return "green"
    elif level <= 6:
        return "orange"
    else:
        return "red"
