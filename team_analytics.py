"""
Team Mood Analytics Module
Aggregates mood data across all employees to surface team-level insights.
Returns data structures ready for Plotly/Streamlit charts.
"""

from __future__ import annotations
from collections import Counter
import pandas as pd
from database import get_all_mood_logs, get_all_users


# ── Data loading ──────────────────────────────────────────────────────────────

def load_mood_dataframe() -> pd.DataFrame:
    """Load all mood logs into a DataFrame with parsed timestamps."""
    rows = get_all_mood_logs(limit=10000)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["logged_at"] = pd.to_datetime(df["logged_at"])
    df["date"] = df["logged_at"].dt.date
    df["hour"] = df["logged_at"].dt.hour
    return df


# ── Aggregations ──────────────────────────────────────────────────────────────

def mood_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count of each mood across all employees."""
    if df.empty:
        return pd.DataFrame(columns=["mood", "count"])
    counts = df["mood"].value_counts().reset_index()
    counts.columns = ["mood", "count"]
    return counts


def mood_trend_over_time(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """
    Average stress level per time period (D=daily, W=weekly).
    Returns DataFrame with columns: date, avg_stress, entry_count.
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "avg_stress", "entry_count"])
    df2 = df.copy()
    df2 = df2.set_index("logged_at").resample(freq).agg(
        avg_stress=("stress_level", "mean"),
        entry_count=("stress_level", "count"),
    ).reset_index()
    df2.rename(columns={"logged_at": "date"}, inplace=True)
    df2["avg_stress"] = df2["avg_stress"].round(2)
    return df2


def department_mood_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Average stress and dominant mood per department."""
    if df.empty or "department" not in df.columns:
        return pd.DataFrame()
    summary = df.groupby("department").agg(
        avg_stress=("stress_level", "mean"),
        total_entries=("stress_level", "count"),
        dominant_mood=("mood", lambda x: x.mode()[0] if len(x) > 0 else "N/A"),
    ).reset_index()
    summary["avg_stress"] = summary["avg_stress"].round(2)
    return summary


def employee_stress_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Rank employees by average stress level (highest first)."""
    if df.empty:
        return pd.DataFrame(columns=["employee_name", "avg_stress", "entries"])
    ranking = df.groupby("employee_name").agg(
        avg_stress=("stress_level", "mean"),
        entries=("stress_level", "count"),
    ).reset_index()
    ranking["avg_stress"] = ranking["avg_stress"].round(2)
    ranking = ranking.sort_values("avg_stress", ascending=False).reset_index(drop=True)
    return ranking


def team_morale_score(df: pd.DataFrame) -> dict:
    """
    Compute a single team morale score (0-100) based on inverse avg stress.
    Also returns a label and color.
    """
    if df.empty:
        return {"score": 50, "label": "No data", "color": "gray", "total_logs": 0}

    avg_stress = df["stress_level"].mean()
    # Invert: stress 0 → morale 100, stress 10 → morale 0
    score = round((1 - avg_stress / 10) * 100)
    score = max(0, min(100, score))

    if score >= 75:
        label, color = "High Morale", "green"
    elif score >= 50:
        label, color = "Moderate Morale", "orange"
    elif score >= 25:
        label, color = "Low Morale", "red"
    else:
        label, color = "Critical — Immediate Action Needed", "darkred"

    return {
        "score": score,
        "label": label,
        "color": color,
        "total_logs": len(df),
        "avg_stress": round(avg_stress, 2),
    }


def mood_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a pivot table: rows = employee, columns = mood, values = count.
    Useful for a heatmap visualization.
    """
    if df.empty:
        return pd.DataFrame()
    pivot = df.pivot_table(
        index="employee_name", columns="mood",
        values="stress_level", aggfunc="count", fill_value=0
    )
    return pivot


def recent_mood_counts(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """Mood distribution for the last N days."""
    if df.empty:
        return pd.DataFrame(columns=["mood", "count"])
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    recent = df[df["logged_at"] >= cutoff]
    if recent.empty:
        return pd.DataFrame(columns=["mood", "count"])
    counts = recent["mood"].value_counts().reset_index()
    counts.columns = ["mood", "count"]
    return counts
