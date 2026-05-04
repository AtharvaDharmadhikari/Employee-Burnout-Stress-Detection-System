import sqlite3
import os
from datetime import datetime, date

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "task_optimizer.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    c = conn.cursor()

    # Users (replaces employees — supports login + roles)
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id  TEXT NOT NULL UNIQUE,
            name         TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            department   TEXT DEFAULT 'General',
            role         TEXT DEFAULT 'employee',  -- 'employee' | 'hr'
            created_at   TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Attendance — one row per employee per day
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            date            TEXT NOT NULL,
            check_in_time   TEXT,
            check_out_time  TEXT,
            entry_mood      TEXT,
            entry_stress    INTEGER DEFAULT 0,
            exit_mood       TEXT,
            exit_stress     INTEGER DEFAULT 0,
            UNIQUE(user_id, date),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Mood logs
    c.execute("""
        CREATE TABLE IF NOT EXISTS mood_logs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            mood         TEXT NOT NULL,
            stress_level INTEGER NOT NULL DEFAULT 0,
            source       TEXT DEFAULT 'manual',
            notes        TEXT,
            logged_at    TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Task logs
    c.execute("""
        CREATE TABLE IF NOT EXISTS task_logs (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id            INTEGER NOT NULL,
            mood               TEXT,
            workload           INTEGER,
            recommended_task   TEXT,
            actual_task        TEXT,
            estimated_duration REAL,
            priority           TEXT DEFAULT 'Medium',
            deadline_days      INTEGER,
            status             TEXT DEFAULT 'pending',
            created_at         TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # HR alerts
    c.execute("""
        CREATE TABLE IF NOT EXISTS hr_alerts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            alert_type   TEXT NOT NULL,
            stress_level INTEGER,
            message      TEXT,
            acknowledged INTEGER DEFAULT 0,
            created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


# ── User helpers ──────────────────────────────────────────────────────────────

def create_user(employee_id: str, name: str, password_hash: str,
                department: str = "General", role: str = "employee") -> bool:
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO users (employee_id, name, password_hash, department, role)
               VALUES (?, ?, ?, ?, ?)""",
            (employee_id.upper(), name, password_hash, department, role)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_user_by_employee_id(employee_id: str):
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE employee_id = ?", (employee_id.upper(),)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_id(user_id: int):
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_all_users():
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, employee_id, name, department, role, created_at FROM users ORDER BY name"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_user(user_id: int):
    conn = get_connection()
    try:
        conn.execute("DELETE FROM mood_logs  WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM task_logs  WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM hr_alerts  WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM attendance WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users      WHERE id = ?",      (user_id,))
        conn.commit()
    finally:
        conn.close()


# ── Attendance helpers ────────────────────────────────────────────────────────

def get_today_attendance(user_id: int):
    today = date.today().isoformat()
    conn  = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM attendance WHERE user_id = ? AND date = ?",
            (user_id, today)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def check_in(user_id: int, mood: str, stress: int):
    today = date.today().isoformat()
    now   = datetime.now().strftime("%H:%M:%S")
    conn  = get_connection()
    try:
        conn.execute(
            """INSERT INTO attendance (user_id, date, check_in_time, entry_mood, entry_stress)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(user_id, date) DO NOTHING""",
            (user_id, today, now, mood, stress)
        )
        conn.commit()
    finally:
        conn.close()


def check_out(user_id: int, mood: str, stress: int):
    today = date.today().isoformat()
    now   = datetime.now().strftime("%H:%M:%S")
    conn  = get_connection()
    try:
        conn.execute(
            """UPDATE attendance
               SET check_out_time = ?, exit_mood = ?, exit_stress = ?
               WHERE user_id = ? AND date = ?""",
            (now, mood, stress, user_id, today)
        )
        conn.commit()
    finally:
        conn.close()


def get_attendance_history(user_id: int, limit: int = 30):
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT * FROM attendance WHERE user_id = ?
               ORDER BY date DESC LIMIT ?""",
            (user_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_all_attendance_today():
    today = date.today().isoformat()
    conn  = get_connection()
    try:
        rows = conn.execute(
            """SELECT a.*, u.name, u.department
               FROM attendance a
               JOIN users u ON a.user_id = u.id
               WHERE a.date = ?
               ORDER BY a.check_in_time""",
            (today,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Mood log helpers ──────────────────────────────────────────────────────────

def log_mood(user_id: int, mood: str, stress_level: int,
             source: str = "manual", notes: str = ""):
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO mood_logs (user_id, mood, stress_level, source, notes)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, mood, stress_level, source, notes)
        )
        conn.commit()
    finally:
        conn.close()


def get_mood_history(user_id: int, limit: int = 100):
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT ml.*, u.name as employee_name
               FROM mood_logs ml
               JOIN users u ON ml.user_id = u.id
               WHERE ml.user_id = ?
               ORDER BY ml.logged_at DESC LIMIT ?""",
            (user_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_all_mood_logs(limit: int = 1000):
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT ml.*, u.name as employee_name, u.department
               FROM mood_logs ml
               JOIN users u ON ml.user_id = u.id
               ORDER BY ml.logged_at DESC LIMIT ?""",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Task log helpers ──────────────────────────────────────────────────────────

def log_task(user_id: int, mood: str, workload: int,
             recommended_task: str, priority: str = "Medium",
             deadline_days: int = 7, estimated_duration: float = 0.0):
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO task_logs
               (user_id, mood, workload, recommended_task,
                priority, deadline_days, estimated_duration)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, mood, workload, recommended_task,
             priority, deadline_days, estimated_duration)
        )
        conn.commit()
    finally:
        conn.close()


def get_task_history(user_id: int, limit: int = 50):
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT * FROM task_logs WHERE user_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (user_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── HR alert helpers ──────────────────────────────────────────────────────────

def create_alert(user_id: int, alert_type: str, stress_level: int, message: str):
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO hr_alerts (user_id, alert_type, stress_level, message)
               VALUES (?, ?, ?, ?)""",
            (user_id, alert_type, stress_level, message)
        )
        conn.commit()
    finally:
        conn.close()


def get_unacknowledged_alerts():
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT ha.*, u.name as employee_name, u.department
               FROM hr_alerts ha
               JOIN users u ON ha.user_id = u.id
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
            """SELECT ha.*, u.name as employee_name, u.department
               FROM hr_alerts ha
               JOIN users u ON ha.user_id = u.id
               ORDER BY ha.created_at DESC LIMIT ?""",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()