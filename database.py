import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "task_optimizer.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    c = conn.cursor()

    # Employees
    c.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            department TEXT DEFAULT 'General',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Mood logs — one entry per detection/check-in
    c.execute("""
        CREATE TABLE IF NOT EXISTS mood_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            mood TEXT NOT NULL,
            stress_level INTEGER NOT NULL DEFAULT 0,
            source TEXT DEFAULT 'manual',   -- 'webcam' | 'manual' | 'text'
            notes TEXT,
            logged_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        )
    """)

    # Task logs — recommended + completed tasks
    c.execute("""
        CREATE TABLE IF NOT EXISTS task_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            mood TEXT,
            workload INTEGER,
            recommended_task TEXT,
            actual_task TEXT,
            estimated_duration REAL,
            priority TEXT DEFAULT 'Medium',
            deadline_days INTEGER,
            status TEXT DEFAULT 'pending',  -- 'pending' | 'in_progress' | 'completed'
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        )
    """)

    # HR alerts
    c.execute("""
        CREATE TABLE IF NOT EXISTS hr_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            alert_type TEXT NOT NULL,       -- 'stress' | 'burnout' | 'disengagement'
            stress_level INTEGER,
            message TEXT,
            acknowledged INTEGER DEFAULT 0, -- 0 = unread, 1 = acknowledged
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        )
    """)

    conn.commit()
    conn.close()


# ── Employee helpers ──────────────────────────────────────────────────────────

def add_employee(name: str, department: str = "General") -> int:
    conn = get_connection()
    try:
        c = conn.cursor()
        c.execute(
            "INSERT OR IGNORE INTO employees (name, department) VALUES (?, ?)",
            (name, department)
        )
        conn.commit()
        c.execute("SELECT id FROM employees WHERE name = ?", (name,))
        row = c.fetchone()
        return row["id"]
    finally:
        conn.close()


def get_all_employees():
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM employees ORDER BY name").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_employee_by_name(name: str):
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM employees WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_employee(employee_id: int):
    """Delete an employee and all their associated data."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM mood_logs  WHERE employee_id = ?", (employee_id,))
        conn.execute("DELETE FROM task_logs  WHERE employee_id = ?", (employee_id,))
        conn.execute("DELETE FROM hr_alerts  WHERE employee_id = ?", (employee_id,))
        conn.execute("DELETE FROM employees  WHERE id = ?",          (employee_id,))
        conn.commit()
    finally:
        conn.close()


# ── Mood log helpers ──────────────────────────────────────────────────────────

def log_mood(employee_id: int, mood: str, stress_level: int,
             source: str = "manual", notes: str = ""):
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO mood_logs
               (employee_id, mood, stress_level, source, notes)
               VALUES (?, ?, ?, ?, ?)""",
            (employee_id, mood, stress_level, source, notes)
        )
        conn.commit()
    finally:
        conn.close()


def get_mood_history(employee_id: int, limit: int = 100):
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT ml.*, e.name as employee_name
               FROM mood_logs ml
               JOIN employees e ON ml.employee_id = e.id
               WHERE ml.employee_id = ?
               ORDER BY ml.logged_at DESC LIMIT ?""",
            (employee_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_all_mood_logs(limit: int = 500):
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT ml.*, e.name as employee_name, e.department
               FROM mood_logs ml
               JOIN employees e ON ml.employee_id = e.id
               ORDER BY ml.logged_at DESC LIMIT ?""",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Task log helpers ──────────────────────────────────────────────────────────

def log_task(employee_id: int, mood: str, workload: int,
             recommended_task: str, priority: str = "Medium",
             deadline_days: int = 7, estimated_duration: float = 0.0):
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO task_logs
               (employee_id, mood, workload, recommended_task,
                priority, deadline_days, estimated_duration)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (employee_id, mood, workload, recommended_task,
             priority, deadline_days, estimated_duration)
        )
        conn.commit()
    finally:
        conn.close()


def get_task_history(employee_id: int, limit: int = 50):
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT * FROM task_logs WHERE employee_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (employee_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── HR alert helpers ──────────────────────────────────────────────────────────

def create_alert(employee_id: int, alert_type: str,
                 stress_level: int, message: str):
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO hr_alerts
               (employee_id, alert_type, stress_level, message)
               VALUES (?, ?, ?, ?)""",
            (employee_id, alert_type, stress_level, message)
        )
        conn.commit()
    finally:
        conn.close()


def get_unacknowledged_alerts():
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT ha.*, e.name as employee_name, e.department
               FROM hr_alerts ha
               JOIN employees e ON ha.employee_id = e.id
               WHERE ha.acknowledged = 0
               ORDER BY ha.created_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def acknowledge_alert(alert_id: int):
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE hr_alerts SET acknowledged = 1 WHERE id = ?", (alert_id,)
        )
        conn.commit()
    finally:
        conn.close()


def get_all_alerts(limit: int = 200):
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT ha.*, e.name as employee_name, e.department
               FROM hr_alerts ha
               JOIN employees e ON ha.employee_id = e.id
               ORDER BY ha.created_at DESC LIMIT ?""",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
