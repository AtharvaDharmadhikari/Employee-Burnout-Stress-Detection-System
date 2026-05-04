"""
Employee Burnout & Stress Detection System — Flask App
Run with: python flask_app.py
Requires: pip install flask
"""

from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, flash)
from functools import wraps
import base64, io, numpy as np, cv2, pandas as pd
from PIL import Image
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from database import (
    init_db, get_all_users, delete_user,
    log_mood, get_mood_history,
    log_task, get_task_history,
    get_unacknowledged_alerts, acknowledge_alert, get_all_alerts,
    get_today_attendance, check_in, check_out, get_attendance_history,
    get_all_attendance_today,
)
from auth import login, register
from emotion_detection import MOOD_OPTIONS, mood_to_stress, detect_from_frame
from task_recommendation import recommend_task
from stress_alerts import evaluate_stress
from burnout_risk import calculate_burnout_risk, bulk_burnout_risk, get_model_meta
from team_analytics import (
    load_mood_dataframe, mood_trend_over_time, department_mood_summary,
    employee_stress_ranking, team_morale_score, recent_mood_counts,
)

app = Flask(__name__)
app.secret_key = "burnout_detection_secret_2024"

PRIORITY_OPTIONS = ["Low", "Medium", "High", "Critical"]

init_db()


# ── Decorators ────────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


def hr_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login_page"))
        if session["user"]["role"] != "hr":
            return redirect(url_for("employee_dashboard"))
        return f(*args, **kwargs)
    return decorated


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_python(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if hasattr(obj, 'item'):   # numpy scalar (int32, float32, float64, etc.)
        return obj.item()
    return obj


def make_chart(fig, height=350):
    fig.update_layout(
        height=height,
        margin=dict(t=30, b=20, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12),
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


@app.context_processor
def inject_globals():
    count = 0
    if "user" in session and session["user"]["role"] == "hr":
        count = len(get_unacknowledged_alerts())
    return {"unread_alerts_count": count, "current_user": session.get("user")}


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def login_page():
    if "user" in session:
        return redirect(url_for("hr_dashboard") if session["user"]["role"] == "hr"
                        else url_for("employee_dashboard"))
    if request.method == "POST":
        result = login(request.form.get("emp_id", ""),
                       request.form.get("password", ""))
        if result["success"]:
            session["user"] = result["user"]
            return redirect(url_for("hr_dashboard") if result["user"]["role"] == "hr"
                            else url_for("employee_dashboard"))
        flash(result["message"], "danger")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register_page():
    if request.method == "POST":
        pw = request.form.get("password", "")
        if pw != request.form.get("confirm_pw", ""):
            flash("Passwords do not match.", "danger")
        else:
            result = register(
                request.form.get("emp_id", ""),
                request.form.get("name", ""),
                pw,
                request.form.get("department", "General"),
                request.form.get("hr_code", ""),
            )
            if result["success"]:
                flash(result["message"] + " Please sign in.", "success")
                return redirect(url_for("login_page"))
            flash(result["message"], "danger")
    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


# ── Employee: Dashboard ───────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def employee_dashboard():
    user = session["user"]
    uid  = user["id"]

    attendance = get_today_attendance(uid)
    history    = get_mood_history(uid, limit=30)
    stats, stress_chart, latest_rec = {}, None, None

    if history:
        df = pd.DataFrame(history)
        df["logged_at"] = pd.to_datetime(df["logged_at"])
        df = df.sort_values("logged_at")
        df["stress_level"] = pd.to_numeric(df["stress_level"])
        stats = {
            "total":       len(df),
            "avg_stress":  round(df["stress_level"].mean(), 1),
            "common_mood": df["mood"].mode()[0],
        }
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["logged_at"].tolist(),
            y=df["stress_level"].tolist(),
            mode="lines+markers",
            line=dict(color="#4f46e5", width=2),
            marker=dict(size=8, color="#4f46e5"),
            name="Stress",
        ))
        fig.add_hline(y=7, line_dash="dash", line_color="#ef4444",
                      annotation_text="Alert Threshold")
        fig.update_layout(
            xaxis_title="Time",
            yaxis=dict(title="Stress", range=[0, 10]),
            showlegend=False,
        )
        stress_chart = make_chart(fig)
        latest_rec   = recommend_task(history[0]["mood"], 5)

    burnout = calculate_burnout_risk(uid)

    return render_template("employee/dashboard.html",
                           user=user, attendance=attendance,
                           stats=stats, stress_chart=stress_chart,
                           latest_rec=latest_rec, burnout=burnout)


# ── Employee: Attendance Check-In ─────────────────────────────────────────────

@app.route("/checkin")
@login_required
def checkin_page():
    user = session["user"]
    return render_template("employee/checkin.html",
                           user=user,
                           attendance=get_today_attendance(user["id"]),
                           history=get_attendance_history(user["id"], limit=14),
                           mood_options=MOOD_OPTIONS,
                           today=date.today().strftime("%A, %d %B %Y"))


@app.route("/api/checkin-manual", methods=["POST"])
@login_required
def api_checkin_manual():
    user   = session["user"]
    uid    = user["id"]
    mood   = request.form.get("mood", "Neutral")
    stress = int(request.form.get("stress", 4))
    action = request.form.get("action", "checkin")

    if action == "checkin":
        check_in(uid, mood, stress)
        log_mood(uid, mood, stress, "manual")
        evaluate_stress(uid, user["name"], stress, mood)
        rec = recommend_task(mood, 5)
        flash(f"Checked in — Mood: {mood}. Suggested: {rec['task']}", "success")
    else:
        check_out(uid, mood, stress)
        log_mood(uid, mood, stress, "manual", notes="exit check-in")
        triggered = evaluate_stress(uid, user["name"], stress, mood)
        if triggered:
            flash(f"Checked out. {len(triggered)} stress alert(s) sent to HR.", "warning")
        else:
            flash(f"Checked out — Exit mood: {mood}. See you tomorrow!", "success")
    return redirect(url_for("checkin_page"))


@app.route("/api/ping", methods=["GET", "POST"])
def api_ping():
    return jsonify({"ok": True, "logged_in": "user" in session})


@app.route("/api/detect-emotion", methods=["POST"])
@login_required
def api_detect_emotion():
    import traceback
    try:
        data = request.get_json(force=True, silent=True) or {}
        img_data = data.get("image", "")
        if not img_data:
            return jsonify({"success": False, "error": "No image data received",
                            "mood": "Neutral", "stress_level": 4})
        if "," in img_data:
            img_data = img_data.split(",")[1]
        img_bytes = base64.b64decode(img_data)
        img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        result = detect_from_frame(frame_bgr)
        return jsonify({"success": True, **_to_python(result)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e),
                        "mood": "Neutral", "stress_level": 4})


@app.route("/api/save-emotion", methods=["POST"])
@login_required
def api_save_emotion():
    user   = session["user"]
    uid    = user["id"]
    data   = request.get_json()
    mood   = data.get("mood", "Neutral")
    stress = int(data.get("stress_level", 4))
    source = data.get("source", "webcam")
    action = data.get("action", "checkin")

    if action == "checkin":
        check_in(uid, mood, stress)
        log_mood(uid, mood, stress, source)
        evaluate_stress(uid, user["name"], stress, mood)
        rec = recommend_task(mood, 5)
        return jsonify({"success": True,
                        "message": f"Checked in — Mood: {mood}",
                        "recommendation": rec["task"],
                        "description": rec["description"]})
    else:
        check_out(uid, mood, stress)
        log_mood(uid, mood, stress, source, notes="exit check-in")
        evaluate_stress(uid, user["name"], stress, mood)
        return jsonify({"success": True,
                        "message": f"Checked out — Exit mood: {mood}. See you tomorrow!"})


# ── Employee: Task Manager ────────────────────────────────────────────────────

@app.route("/tasks", methods=["GET", "POST"])
@login_required
def tasks_page():
    user           = session["user"]
    recommendation = None
    form_data      = {}

    if request.method == "POST":
        mood     = request.form.get("mood", "Neutral")
        workload = int(request.form.get("workload", 5))
        priority = request.form.get("priority", "Medium")
        deadline = int(request.form.get("deadline", 7))
        form_data = {"mood": mood, "workload": workload,
                     "priority": priority, "deadline": deadline}
        recommendation = recommend_task(mood, workload, priority, deadline)
        log_task(user["id"], mood, workload, recommendation["task"], priority, deadline)

    return render_template("employee/tasks.html",
                           user=user,
                           recommendation=recommendation,
                           form_data=form_data,
                           history=get_task_history(user["id"]),
                           mood_options=MOOD_OPTIONS,
                           priority_options=PRIORITY_OPTIONS)


# ── Employee: Wellness Report ─────────────────────────────────────────────────

@app.route("/wellness")
@login_required
def wellness_page():
    user    = session["user"]
    history = get_mood_history(user["id"], limit=200)

    if not history:
        return render_template("employee/wellness.html", user=user,
                               no_data=True, stats={},
                               stress_chart=None, mood_pie=None,
                               mood_box=None, history=[])

    df = pd.DataFrame(history)
    df["logged_at"] = pd.to_datetime(df["logged_at"])
    df = df.sort_values("logged_at")
    df["stress_level"] = pd.to_numeric(df["stress_level"])

    stats = {
        "total":       len(df),
        "avg_stress":  round(df["stress_level"].mean(), 1),
        "common_mood": df["mood"].mode()[0],
    }

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df["logged_at"].tolist(),
        y=df["stress_level"].tolist(),
        mode="lines+markers",
        line=dict(color="#4f46e5", width=2),
        marker=dict(size=8, color="#4f46e5"),
        name="Stress Level",
    ))
    fig1.add_hline(y=7, line_dash="dash", line_color="#ef4444",
                   annotation_text="Alert Threshold")
    fig1.update_layout(
        xaxis_title="Date",
        yaxis=dict(title="Stress Level", range=[0, 10]),
        showlegend=False,
    )

    counts = df["mood"].value_counts().reset_index()
    counts.columns = ["mood", "count"]
    fig2 = px.pie(counts, names="mood", values="count",
                  color_discrete_sequence=px.colors.qualitative.Pastel)

    fig3 = px.box(df, x="mood", y="stress_level", color="mood")
    fig3.update_layout(showlegend=False)
    fig3.update_yaxes(range=[0, 10])

    return render_template("employee/wellness.html",
                           user=user, no_data=False, stats=stats,
                           stress_chart=make_chart(fig1),
                           mood_pie=make_chart(fig2, height=320),
                           mood_box=make_chart(fig3, height=320),
                           history=history[:50])


# ── HR: Dashboard ─────────────────────────────────────────────────────────────

@app.route("/hr/dashboard")
@hr_required
def hr_dashboard():
    user      = session["user"]
    df_all    = load_mood_dataframe()
    users     = get_all_users()
    alerts    = get_unacknowledged_alerts()
    employees = [u for u in users if u["role"] == "employee"]
    morale    = team_morale_score(df_all)
    meta      = get_model_meta()
    risk_data = bulk_burnout_risk(employees)

    gauge_chart = pie_chart = stress_chart = ranking_chart = risk_chart = None

    if not df_all.empty:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=morale["score"],
            title={"text": "Team Morale Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#4f46e5"},
                "steps": [
                    {"range": [0,  25], "color": "#fee2e2"},
                    {"range": [25, 50], "color": "#fef3c7"},
                    {"range": [50, 75], "color": "#d1fae5"},
                    {"range": [75,100], "color": "#a7f3d0"},
                ],
            },
        ))
        gauge_chart = make_chart(fig_g, height=260)

        recent = recent_mood_counts(df_all, days=7)
        if not recent.empty:
            fig_p = px.pie(recent, names="mood", values="count",
                           color_discrete_sequence=px.colors.qualitative.Set3)
            pie_chart = make_chart(fig_p, height=260)

        trend = mood_trend_over_time(df_all, freq="D")
        if not trend.empty:
            fig_t = px.area(trend, x="date", y="avg_stress",
                            color_discrete_sequence=["#ef4444"],
                            labels={"avg_stress": "Avg Stress", "date": "Date"})
            fig_t.add_hline(y=7, line_dash="dash", line_color="red",
                            annotation_text="Alert Threshold")
            fig_t.update_yaxes(range=[0, 10])
            stress_chart = make_chart(fig_t)

        ranking = employee_stress_ranking(df_all)
        if not ranking.empty:
            ranking = ranking.copy()
            ranking["avg_stress"] = pd.to_numeric(ranking["avg_stress"]).round(1)
            r_colors = []
            for s in ranking["avg_stress"]:
                if s >= 7:
                    r_colors.append("#ef4444")
                elif s >= 5:
                    r_colors.append("#eab308")
                else:
                    r_colors.append("#22c55e")
            fig_r = go.Figure(go.Bar(
                x=ranking["employee_name"].tolist(),
                y=ranking["avg_stress"].tolist(),
                text=[f"{s:.1f}" for s in ranking["avg_stress"]],
                textposition="outside",
                marker_color=r_colors,
            ))
            fig_r.update_layout(
                yaxis=dict(range=[0, 11], title="Avg Stress"),
                xaxis_title="",
            )
            ranking_chart = make_chart(fig_r)

    if risk_data:
        names  = [r["name"]          for r in risk_data]
        scores = [float(r["score"])  for r in risk_data]
        labels = [str(int(round(s))) for s in scores]
        bar_colors = []
        for s in scores:
            if s >= 75:
                bar_colors.append("#ef4444")
            elif s >= 50:
                bar_colors.append("#f97316")
            elif s >= 25:
                bar_colors.append("#eab308")
            else:
                bar_colors.append("#22c55e")
        fig_risk = go.Figure(go.Bar(
            x=names, y=scores, text=labels,
            textposition="outside",
            marker_color=bar_colors,
        ))
        fig_risk.add_hline(y=50, line_dash="dash", line_color="orange",
                           annotation_text="High Risk")
        fig_risk.add_hline(y=75, line_dash="dash", line_color="red",
                           annotation_text="Critical")
        fig_risk.update_layout(
            yaxis=dict(range=[0, 115], title="Risk Score"),
            xaxis_title="Employee",
        )
        risk_chart = make_chart(fig_risk)

    dept = department_mood_summary(df_all)

    return render_template("hr/dashboard.html",
                           user=user, employees=employees,
                           alerts=alerts, morale=morale, meta=meta,
                           risk_data=risk_data,
                           gauge_chart=gauge_chart, pie_chart=pie_chart,
                           stress_chart=stress_chart, ranking_chart=ranking_chart,
                           risk_chart=risk_chart,
                           dept=dept.to_dict("records") if not dept.empty else [])


# ── HR: Attendance ────────────────────────────────────────────────────────────

@app.route("/hr/attendance")
@hr_required
def hr_attendance():
    user      = session["user"]
    today_att = get_all_attendance_today()
    all_users = [u for u in get_all_users() if u["role"] == "employee"]
    checked   = {a["user_id"] for a in today_att}
    absent    = [u for u in all_users if u["id"] not in checked]
    return render_template("hr/attendance.html",
                           user=user, today_att=today_att, absent=absent,
                           today=date.today().strftime("%A, %d %B %Y"))


# ── HR: Alerts ────────────────────────────────────────────────────────────────

@app.route("/hr/alerts")
@hr_required
def hr_alerts():
    user       = session["user"]
    unread     = get_unacknowledged_alerts()
    all_alerts = get_all_alerts()
    alert_chart = None

    if all_alerts:
        df_a   = pd.DataFrame(all_alerts)
        counts = df_a["alert_type"].value_counts().reset_index()
        counts.columns = ["type", "count"]
        fig = px.bar(counts, x="type", y="count", color="type",
                     color_discrete_map={"stress":        "#ef4444",
                                         "burnout":       "#f97316",
                                         "disengagement": "#eab308"})
        fig.update_layout(showlegend=False)
        alert_chart = make_chart(fig, height=280)

    return render_template("hr/alerts.html",
                           user=user, unread=unread,
                           all_alerts=all_alerts, alert_chart=alert_chart)


@app.route("/hr/alerts/acknowledge/<int:alert_id>", methods=["POST"])
@hr_required
def hr_acknowledge(alert_id):
    acknowledge_alert(alert_id)
    flash("Alert acknowledged.", "success")
    return redirect(url_for("hr_alerts"))


# ── HR: Employee Management ───────────────────────────────────────────────────

@app.route("/hr/employees")
@hr_required
def hr_employees():
    user      = session["user"]
    users     = get_all_users()
    employees = [u for u in users if u["role"] == "employee"]
    hr_users  = [u for u in users if u["role"] == "hr"]
    risk_list = bulk_burnout_risk(employees)
    risk_map  = {r["id"]: r for r in risk_list}
    return render_template("hr/employees.html",
                           user=user, employees=employees,
                           hr_users=hr_users, risk_map=risk_map)


@app.route("/hr/employees/delete/<int:emp_id>", methods=["POST"])
@hr_required
def hr_delete_employee(emp_id):
    delete_user(emp_id)
    flash("Employee removed successfully.", "success")
    return redirect(url_for("hr_employees"))


def _prewarm():
    """Import DeepFace + TF at startup so the first webcam request doesn't time out."""
    try:
        import threading
        def _load():
            try:
                from deepface import DeepFace
                print("✅ DeepFace pre-warmed.")
            except Exception as e:
                print(f"⚠️  DeepFace pre-warm skipped: {e}")
        threading.Thread(target=_load, daemon=True).start()
    except Exception:
        pass


if __name__ == "__main__":
    _prewarm()
    app.run(debug=True, port=5000, threaded=True, use_reloader=False)