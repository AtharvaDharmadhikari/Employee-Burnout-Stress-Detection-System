"""
Employee Burnout & Stress Detection System
Main Streamlit dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

from database import (
    init_db,
    get_all_users, delete_user,
    log_mood, get_mood_history,
    log_task, get_task_history,
    get_unacknowledged_alerts, acknowledge_alert, get_all_alerts,
    get_today_attendance, check_in, check_out, get_attendance_history,
    get_all_attendance_today,
)
from auth import login, register
from emotion_detection import (
    MOOD_OPTIONS, mood_to_stress, streamlit_webcam_widget,
)
from task_recommendation import recommend_task
PRIORITY_OPTIONS = ["Low", "Medium", "High", "Critical"]
from stress_alerts import (
    evaluate_stress, alert_emoji,
)
from burnout_risk import calculate_burnout_risk, bulk_burnout_risk, get_model_meta
from team_analytics import (
    load_mood_dataframe, mood_trend_over_time,
    department_mood_summary, employee_stress_ranking,
    team_morale_score, recent_mood_counts,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Employee Burnout & Stress Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()


def apply_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .stApp { background: #f0f2f6; }

    /* ── Sidebar ────────────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 55%, #24243e 100%) !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div { color: #cbd5e1 !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #ffffff !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }
    section[data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0 !important; padding: 6px 10px; border-radius: 8px;
        transition: background 0.2s; font-size: 0.93rem;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        color: #ffffff !important; border-radius: 8px !important;
        transition: all 0.2s !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.2) !important;
    }

    /* ── Metric cards ───────────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: #ffffff; border-radius: 14px; padding: 20px 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07); border-left: 5px solid #4f46e5;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.7rem !important; font-weight: 700 !important; color: #1e293b !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important; font-weight: 600 !important;
        color: #64748b !important; text-transform: uppercase; letter-spacing: 0.06em;
    }

    /* ── Buttons ────────────────────────────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        border: none !important; border-radius: 10px !important;
        color: white !important; font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(79,70,229,0.35) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(79,70,229,0.45) !important;
    }
    .stButton > button[kind="secondary"] {
        border-radius: 10px !important; border: 2px solid #4f46e5 !important;
        color: #4f46e5 !important; font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #4f46e5 !important; color: white !important;
    }

    /* ── Typography ─────────────────────────────────────────────────────────── */
    h1 { color: #1e293b !important; font-weight: 700 !important; font-size: 1.85rem !important; }
    h2 { color: #334155 !important; font-weight: 600 !important; }
    h3 { color: #475569 !important; font-weight: 600 !important; }

    /* ── Inputs ─────────────────────────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        border-radius: 8px !important; border: 1.5px solid #e2e8f0 !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 3px rgba(79,70,229,0.12) !important;
    }

    /* ── Tabs ───────────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: #f1f5f9; border-radius: 10px; padding: 4px; gap: 2px;
    }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; font-weight: 500; }
    .stTabs [aria-selected="true"] {
        background: white !important; box-shadow: 0 1px 6px rgba(0,0,0,0.1) !important;
        color: #4f46e5 !important; font-weight: 600 !important;
    }

    /* ── Dataframe ──────────────────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border-radius: 12px; overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }

    /* ── Alerts ─────────────────────────────────────────────────────────────── */
    [data-testid="stAlert"] { border-radius: 10px !important; border: none !important; }

    /* ── Containers with border ─────────────────────────────────────────────── */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        border-radius: 14px !important; border: 1px solid #e2e8f0 !important;
        box-shadow: 0 1px 6px rgba(0,0,0,0.05) !important; background: white;
    }

    /* ── Auth card ──────────────────────────────────────────────────────────── */
    .auth-card {
        background: white; border-radius: 20px; padding: 40px 36px;
        box-shadow: 0 8px 40px rgba(0,0,0,0.12); margin: 10px auto;
    }
    .auth-logo {
        text-align: center; font-size: 3rem; margin-bottom: 6px;
    }
    .auth-title {
        text-align: center; font-size: 1.4rem; font-weight: 700;
        color: #1e293b; margin-bottom: 4px;
    }
    .auth-subtitle {
        text-align: center; font-size: 0.85rem; color: #64748b; margin-bottom: 24px;
    }

    /* ── Page header banner ─────────────────────────────────────────────────── */
    .page-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white; padding: 22px 28px; border-radius: 16px; margin-bottom: 28px;
    }
    .page-header h1 {
        color: white !important; margin: 0; font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    .page-header p { color: rgba(255,255,255,0.82); margin: 5px 0 0; font-size: 0.88rem; }
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="page-header"><h1>{title}</h1>{sub}</div>',
                unsafe_allow_html=True)


apply_custom_css()

# ── Session state ─────────────────────────────────────────────────────────────

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
if "auth_page" not in st.session_state:
    st.session_state.auth_page = "login"


# ── Auth helpers ──────────────────────────────────────────────────────────────

def logout():
    st.session_state.logged_in = False
    st.session_state.user      = None
    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# AUTH PAGES (shown when not logged in)
# ═════════════════════════════════════════════════════════════════════════════

def page_login():
    _, col2, _ = st.columns([1, 1.1, 1])
    with col2:
        st.markdown("""
        <div class="auth-card">
            <div class="auth-logo">🧠</div>
            <div class="auth-title">Employee Burnout & Stress Detection</div>
            <div class="auth-subtitle">Sign in to your account</div>
        </div>
        """, unsafe_allow_html=True)

        emp_id   = st.text_input("Employee ID", placeholder="e.g. EMP001")
        password = st.text_input("Password", type="password")

        if st.button("Sign In", use_container_width=True, type="primary"):
            if emp_id and password:
                result = login(emp_id, password)
                if result["success"]:
                    st.session_state.logged_in = True
                    st.session_state.user      = result["user"]
                    st.rerun()
                else:
                    st.error(result["message"])
            else:
                st.warning("Please enter your Employee ID and password.")

        st.markdown("<p style='text-align:center;color:#64748b;font-size:0.85rem;margin-top:12px'>Don't have an account?</p>",
                    unsafe_allow_html=True)
        if st.button("Create Account", use_container_width=True):
            st.session_state.auth_page = "register"
            st.rerun()


def page_register():
    _, col2, _ = st.columns([1, 1.1, 1])
    with col2:
        st.markdown("""
        <div class="auth-card">
            <div class="auth-logo">🧠</div>
            <div class="auth-title">Create Account</div>
            <div class="auth-subtitle">Register to access the wellness platform</div>
        </div>
        """, unsafe_allow_html=True)

        emp_id     = st.text_input("Employee ID *", placeholder="e.g. EMP001")
        name       = st.text_input("Full Name *", placeholder="e.g. John Smith")
        department = st.text_input("Department", placeholder="e.g. Engineering", value="General")
        password   = st.text_input("Password *", type="password",
                                   help="Minimum 6 characters")
        confirm_pw = st.text_input("Confirm Password *", type="password")
        hr_code    = st.text_input("HR Access Code",
                                   placeholder="Leave blank if you are an employee",
                                   type="password",
                                   help="Enter HR code only if you are an HR manager")

        if st.button("Create Account", use_container_width=True, type="primary"):
            if password != confirm_pw:
                st.error("Passwords do not match.")
            else:
                result = register(emp_id, name, password, department, hr_code)
                if result["success"]:
                    st.success(result["message"])
                    st.info("Please login with your new credentials.")
                    st.session_state.auth_page = "login"
                    st.rerun()
                else:
                    st.error(result["message"])

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Back to Login", use_container_width=True):
            st.session_state.auth_page = "login"
            st.rerun()


# ── Show auth pages if not logged in ─────────────────────────────────────────

if not st.session_state.logged_in:
    if st.session_state.auth_page == "register":
        page_register()
    else:
        page_login()
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# LOGGED IN — load user info
# ═════════════════════════════════════════════════════════════════════════════

user    = st.session_state.user
is_hr   = user["role"] == "hr"
user_id = user["id"]

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=56)
    st.markdown(f"### {'🏢 HR Dashboard' if is_hr else '👤 My Dashboard'}")
    st.caption(f"Logged in as **{user['name']}** ({user['employee_id']})")
    st.caption(f"Role: {'🔑 HR Manager' if is_hr else '👷 Employee'} · {user['department']}")
    st.divider()

    # Navigation — different options for HR vs Employee
    if is_hr:
        page = st.radio("Navigate", [
            "🏠 HR Dashboard",
            "📅 Today's Attendance",
            "🔔 HR Alerts",
            "👤 Employee Management",
        ], label_visibility="collapsed")
    else:
        page = st.radio("Navigate", [
            "🏠 My Dashboard",
            "📸 Attendance Check-In",
            "✅ Task Manager",
            "📊 My Wellness Report",
        ], label_visibility="collapsed")

    st.divider()

    # Unread alerts badge (HR only)
    if is_hr:
        unread = get_unacknowledged_alerts()
        if unread:
            st.warning(f"⚠️ {len(unread)} unread alert(s)")

    if st.button("🚪 Logout", use_container_width=True):
        logout()


# ═════════════════════════════════════════════════════════════════════════════
# EMPLOYEE PAGES
# ═════════════════════════════════════════════════════════════════════════════

# ── Employee: My Dashboard ────────────────────────────────────────────────────
if not is_hr and page == "🏠 My Dashboard":
    page_header(f"👋 Welcome, {user['name']}", f"{user['department']} · {user['employee_id']}")

    # Attendance status today
    attendance = get_today_attendance(user_id)
    col1, col2, col3 = st.columns(3)

    if attendance:
        col1.metric("📅 Check-In", attendance["check_in_time"] or "—")
        col2.metric("🚪 Check-Out", attendance["check_out_time"] or "Pending")
        col3.metric("😊 Entry Mood", attendance["entry_mood"] or "—")
    else:
        col1.metric("📅 Check-In", "Not yet")
        col2.metric("🚪 Check-Out", "Not yet")
        col3.metric("😊 Entry Mood", "—")
        st.info("You haven't checked in today. Go to **📸 Attendance Check-In** to check in.")

    st.divider()

    # Personal mood summary
    history = get_mood_history(user_id, limit=30)
    if history:
        df = pd.DataFrame(history)
        df["logged_at"] = pd.to_datetime(df["logged_at"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Check-Ins", len(df))
        col2.metric("Avg Stress", f"{df['stress_level'].mean():.1f}/10")
        col3.metric("Most Common Mood", df["mood"].mode()[0])

        st.subheader("Your Stress Trend (Last 30 logs)")
        fig = px.line(df, x="logged_at", y="stress_level", markers=True,
                      color_discrete_sequence=["#3498db"],
                      labels={"stress_level": "Stress Level", "logged_at": "Time"})
        fig.add_hline(y=7, line_dash="dash", line_color="red",
                      annotation_text="Alert Threshold")
        fig.update_layout(yaxis_range=[0, 10])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No mood data yet. Start with a **Mood Check-In**.")

    # Today's task recommendation based on latest mood
    if history:
        latest_mood = history[0]["mood"]
        rec = recommend_task(latest_mood, 5)
        st.subheader("💡 Suggested Task Based on Latest Mood")
        st.success(f"**{rec['task']}** — {rec['description']}")


# ── Employee: Attendance Check-In ─────────────────────────────────────────────
elif not is_hr and page == "📸 Attendance Check-In":
    page_header("📸 Attendance Check-In",
                f"{user['name']} · {date.today().strftime('%A, %d %B %Y')} · Mood captured automatically")
    st.divider()

    attendance = get_today_attendance(user_id)

    # ── MORNING ENTRY ──────────────────────────────────────────────────────────
    if not attendance:
        st.subheader("🌅 Morning Entry")
        st.markdown("Capture your face to detect your mood and mark attendance for today.")

        tab1, tab2 = st.tabs(["📷 Webcam (auto-detect mood)", "✏️ Select mood manually"])

        with tab1:
            result = streamlit_webcam_widget(key="checkin_camera")
            if result and "error" not in result:
                mood, stress = result["mood"], result["stress_level"]
                st.success(f"Detected mood: **{mood}** — Stress: {stress}/10")
                check_in(user_id, mood, stress)
                log_mood(user_id, mood, stress, result.get("source", "webcam"))
                evaluate_stress(user_id, user["name"], stress, mood)
                rec = recommend_task(mood, 5)
                st.info(f"📋 **Today's task:** {rec['task']} — {rec['description']}")
                st.success("✅ Attendance marked! Have a great day.")

        with tab2:
            m = st.selectbox("How are you feeling?", MOOD_OPTIONS, key="entry_mood_sel")
            s = st.slider("Stress level", 0, 10, mood_to_stress(m), key="entry_stress_sel")
            if st.button("✅ Check In", type="primary", use_container_width=True,
                         key="checkin_btn"):
                check_in(user_id, m, s)
                log_mood(user_id, m, s, "manual")
                evaluate_stress(user_id, user["name"], s, m)
                rec = recommend_task(m, 5)
                st.success(f"✅ Checked in — Mood: **{m}**")
                st.info(f"📋 **Today's task:** {rec['task']} — {rec['description']}")

    # ── ALREADY CHECKED IN ────────────────────────────────────────────────────
    else:
        st.success(f"✅ Checked in at **{attendance['check_in_time']}** — "
                   f"Entry mood: **{attendance['entry_mood']}**")

        # ── EVENING EXIT ───────────────────────────────────────────────────────
        if not attendance["check_out_time"]:
            st.divider()
            st.subheader("🌆 Evening Exit")
            st.markdown("Capture your mood before leaving to complete today's attendance.")

            tab1, tab2 = st.tabs(["📷 Webcam (auto-detect mood)", "✏️ Select mood manually"])

            with tab1:
                result = streamlit_webcam_widget(key="checkout_camera")
                if result and "error" not in result:
                    mood, stress = result["mood"], result["stress_level"]
                    st.success(f"Detected mood: **{mood}** — Stress: {stress}/10")
                    check_out(user_id, mood, stress)
                    log_mood(user_id, mood, stress, result.get("source", "webcam"),
                             notes="exit check-in")
                    triggered = evaluate_stress(user_id, user["name"], stress, mood)
                    for alert in triggered:
                        st.error(f"{alert_emoji(alert['type'])} {alert['message']}")
                    st.success("✅ Checked out. See you tomorrow!")

            with tab2:
                m = st.selectbox("End-of-day mood", MOOD_OPTIONS, key="exit_mood_sel")
                s = st.slider("Stress level", 0, 10, mood_to_stress(m), key="exit_stress_sel")
                if st.button("🚪 Check Out", type="primary", use_container_width=True,
                             key="checkout_btn"):
                    check_out(user_id, m, s)
                    log_mood(user_id, m, s, "manual", notes="exit check-in")
                    triggered = evaluate_stress(user_id, user["name"], s, m)
                    for alert in triggered:
                        st.error(f"{alert_emoji(alert['type'])} {alert['message']}")
                    st.success(f"✅ Checked out — Exit mood: **{m}**. See you tomorrow!")
        else:
            st.success(f"✅ Checked out at **{attendance['check_out_time']}** — "
                       f"Exit mood: **{attendance['exit_mood']}**")
            st.info("Attendance complete for today. See you tomorrow!")

    # Attendance history
    st.divider()
    st.subheader("📋 Attendance History")
    hist = get_attendance_history(user_id, limit=14)
    if hist:
        df_att = pd.DataFrame(hist)
        st.dataframe(
            df_att[["date", "check_in_time", "check_out_time",
                    "entry_mood", "entry_stress", "exit_mood", "exit_stress"]],
            use_container_width=True,
        )
    else:
        st.info("No attendance records yet.")


# ── Employee: Task Manager ────────────────────────────────────────────────────
elif not is_hr and page == "✅ Task Manager":
    page_header("✅ Task Manager", "Get an AI-powered task recommendation based on your mood and workload")

    col1, col2 = st.columns(2)
    with col1:
        rec_mood     = st.selectbox("Current Mood", MOOD_OPTIONS)
        rec_workload = st.slider("Workload", 1, 10, 5)
    with col2:
        rec_priority = st.selectbox("Priority", PRIORITY_OPTIONS, index=1)
        rec_deadline = st.number_input("Days until deadline", 1, 365, 7)

    if st.button("Get Recommendation", type="primary", use_container_width=True):
        rec = recommend_task(rec_mood, rec_workload, rec_priority, rec_deadline)
        st.success(f"**Recommended:** {rec['task']}")
        st.info(rec["description"])
        log_task(user_id, rec_mood, rec_workload, rec["task"], rec_priority, rec_deadline)
        st.caption("Saved to your task history.")

    st.divider()
    st.subheader("Task History")
    history = get_task_history(user_id)
    if history:
        st.dataframe(pd.DataFrame(history)[
            ["created_at", "mood", "workload", "recommended_task",
             "priority", "deadline_days", "status"]
        ], use_container_width=True)
    else:
        st.info("No tasks recorded yet.")


# ── Employee: Mood History ────────────────────────────────────────────────────
elif not is_hr and page == "📊 My Wellness Report":
    page_header("📊 My Wellness Report", "Your personal mood and stress trends")

    history = get_mood_history(user_id, limit=200)
    if not history:
        st.info("No mood logs yet.")
        st.stop()

    df = pd.DataFrame(history)
    df["logged_at"] = pd.to_datetime(df["logged_at"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Check-Ins", len(df))
    col2.metric("Avg Stress", f"{df['stress_level'].mean():.1f}/10")
    col3.metric("Most Common Mood", df["mood"].mode()[0])

    st.subheader("Stress Over Time")
    fig = px.line(df, x="logged_at", y="stress_level", markers=True,
                  color_discrete_sequence=["#3498db"],
                  labels={"stress_level": "Stress Level", "logged_at": "Date"})
    fig.add_hline(y=7, line_dash="dash", line_color="red", annotation_text="Alert Threshold")
    fig.update_layout(yaxis_range=[0, 10])
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mood Distribution")
        counts = df["mood"].value_counts().reset_index()
        counts.columns = ["mood", "count"]
        fig2 = px.pie(counts, names="mood", values="count",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Stress by Mood")
        fig3 = px.box(df, x="mood", y="stress_level", color="mood")
        fig3.update_layout(showlegend=False, yaxis_range=[0, 10])
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Full Log")
    st.dataframe(df[["logged_at", "mood", "stress_level", "source", "notes"]],
                 use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# HR PAGES
# ═════════════════════════════════════════════════════════════════════════════

# ── HR: Overview ──────────────────────────────────────────────────────────────
# ── HR: Combined Dashboard ───────────────────────────────────────────────────
elif is_hr and page == "🏠 HR Dashboard":
    page_header("🏠 HR Dashboard", "Team wellness overview and burnout risk monitoring")

    df_all = load_mood_dataframe()
    morale = team_morale_score(df_all)
    users  = get_all_users()
    alerts = get_unacknowledged_alerts()

    # ── Top metrics ────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Employees", len([u for u in users if u["role"] == "employee"]))
    col2.metric("📊 Total Check-Ins", morale["total_logs"])
    col3.metric("🎯 Team Morale", f"{morale['score']}/100")
    col4.metric("🔔 Unread Alerts", len(alerts))

    if df_all.empty:
        st.info("No mood data yet.")
        st.stop()

    st.divider()

    # ── Morale gauge + mood distribution ──────────────────────────────────────
    col1, col2 = st.columns([1, 1.5])
    with col1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=morale["score"],
            title={"text": "Team Morale Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0,  25], "color": "darkred"},
                    {"range": [25, 50], "color": "red"},
                    {"range": [50, 75], "color": "orange"},
                    {"range": [75, 100],"color": "green"},
                ],
            },
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.subheader("Mood Distribution (Last 7 Days)")
        recent = recent_mood_counts(df_all, days=7)
        if not recent.empty:
            fig = px.pie(recent, names="mood", values="count",
                         color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(margin=dict(t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Stress trend + employee ranking ───────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Daily Stress Trend")
        trend = mood_trend_over_time(df_all, freq="D")
        if not trend.empty:
            fig2 = px.area(trend, x="date", y="avg_stress",
                           color_discrete_sequence=["#e74c3c"],
                           labels={"avg_stress": "Avg Stress", "date": "Date"})
            fig2.add_hline(y=7, line_dash="dash", line_color="red",
                           annotation_text="Alert Threshold")
            fig2.update_layout(yaxis_range=[0, 10])
            st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Employee Stress Ranking")
        ranking = employee_stress_ranking(df_all)
        if not ranking.empty:
            fig3 = px.bar(ranking, x="employee_name", y="avg_stress",
                          color="avg_stress",
                          color_continuous_scale=["green", "yellow", "red"],
                          range_color=[0, 10], text="avg_stress",
                          labels={"avg_stress": "Avg Stress", "employee_name": "Employee"})
            fig3.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig3.update_layout(coloraxis_showscale=False, yaxis_range=[0, 11])
            st.plotly_chart(fig3, use_container_width=True)

    # ── Department summary ─────────────────────────────────────────────────────
    dept = department_mood_summary(df_all)
    if not dept.empty and len(dept) > 1:
        st.divider()
        st.subheader("Department Summary")
        st.dataframe(dept, use_container_width=True)

    # ── Burnout Risk Overview ──────────────────────────────────────────────────
    st.divider()
    st.subheader("🔥 Burnout Risk Overview")

    meta = get_model_meta()
    if meta:
        st.success(
            f"🤖 **ML-Powered** — GradientBoostingRegressor trained on {meta.get('training_rows', 0):,} real employee records "
            f"| R² = {meta.get('r2', 'N/A')} | MAE = {meta.get('mae', 'N/A')} "
            f"| Scoring: 60% ML + 40% behavioural"
        )
    else:
        st.info("⚙️ Formula-based scoring active. Run `python train_burnout_model.py` after placing `data/train.csv` to enable ML model.")

    employees = [u for u in users if u["role"] == "employee"]
    risk_data  = bulk_burnout_risk(employees)

    if risk_data:
        critical = [r for r in risk_data if r["score"] >= 75]
        high     = [r for r in risk_data if 50 <= r["score"] < 75]

        if critical:
            st.error(f"⚠️ {len(critical)} employee(s) at **Critical Risk** — immediate attention needed")
        if high:
            st.warning(f"🟠 {len(high)} employee(s) at **High Risk**")

        # Risk bar chart
        df_risk = pd.DataFrame([{
            "Employee": r["name"],
            "Risk Score": r["score"],
            "Risk Level": r["label"],
        } for r in risk_data])

        fig_risk = px.bar(
            df_risk, x="Employee", y="Risk Score",
            color="Risk Score",
            color_continuous_scale=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"],
            range_color=[0, 100], text="Risk Score",
            labels={"Risk Score": "Burnout Risk (0-100)"},
        )
        fig_risk.add_hline(y=50, line_dash="dash", line_color="orange",
                           annotation_text="High Risk threshold")
        fig_risk.add_hline(y=75, line_dash="dash", line_color="red",
                           annotation_text="Critical threshold")
        fig_risk.update_traces(texttemplate="%{text}", textposition="outside")
        fig_risk.update_layout(coloraxis_showscale=False, yaxis_range=[0, 110])
        st.plotly_chart(fig_risk, use_container_width=True)


# ── HR: Today's Attendance ────────────────────────────────────────────────────
elif is_hr and page == "📅 Today's Attendance":
    page_header("📅 Today's Attendance", date.today().strftime('%A, %d %B %Y'))

    today_att = get_all_attendance_today()
    all_users = [u for u in get_all_users() if u["role"] == "employee"]

    checked_in_ids = {a["user_id"] for a in today_att}
    not_checked_in = [u for u in all_users if u["id"] not in checked_in_ids]

    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Checked In", len(today_att))
    col2.metric("⏳ Not Checked In", len(not_checked_in))
    col3.metric("👥 Total Employees", len(all_users))

    st.divider()

    if today_att:
        st.subheader("✅ Present Today")
        df_att = pd.DataFrame(today_att)
        df_att["status"] = df_att["check_out_time"].apply(
            lambda x: "✅ Checked Out" if x else "🏢 In Office"
        )
        st.dataframe(
            df_att[["name", "department", "check_in_time", "check_out_time",
                    "entry_mood", "entry_stress", "exit_mood", "exit_stress", "status"]],
            use_container_width=True,
        )

    if not_checked_in:
        st.subheader("⚠️ Not Checked In Today")
        df_absent = pd.DataFrame(not_checked_in)
        st.dataframe(df_absent[["employee_id", "name", "department"]],
                     use_container_width=True)


# ── HR: Team Analytics ────────────────────────────────────────────────────────
elif is_hr and page == "👥 Team Analytics":
    page_header("👥 Team Analytics", "Department-wide mood and stress insights")

    df_all = load_mood_dataframe()
    if df_all.empty:
        st.info("No team mood data yet.")
        st.stop()

    morale = team_morale_score(df_all)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=morale["score"],
        title={"text": "Team Morale Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 25],  "color": "darkred"},
                {"range": [25, 50], "color": "red"},
                {"range": [50, 75], "color": "orange"},
                {"range": [75, 100],"color": "green"},
            ],
        },
    ))
    fig_gauge.update_layout(height=250, margin=dict(t=30, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mood Distribution (Last 7 Days)")
        recent = recent_mood_counts(df_all, days=7)
        if not recent.empty:
            fig = px.pie(recent, names="mood", values="count",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Stress Trend")
        trend = mood_trend_over_time(df_all, freq="D")
        if not trend.empty:
            fig2 = px.area(trend, x="date", y="avg_stress",
                           color_discrete_sequence=["#e74c3c"],
                           labels={"avg_stress": "Avg Stress", "date": "Date"})
            fig2.add_hline(y=7, line_dash="dash", line_color="red")
            fig2.update_layout(yaxis_range=[0, 10])
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Employee Stress Ranking")
    ranking = employee_stress_ranking(df_all)
    if not ranking.empty:
        fig3 = px.bar(ranking, x="employee_name", y="avg_stress",
                      color="avg_stress",
                      color_continuous_scale=["green", "yellow", "red"],
                      range_color=[0, 10], text="avg_stress",
                      labels={"avg_stress": "Avg Stress", "employee_name": "Employee"})
        fig3.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig3.update_layout(coloraxis_showscale=False, yaxis_range=[0, 11])
        st.plotly_chart(fig3, use_container_width=True)

    dept = department_mood_summary(df_all)
    if not dept.empty and len(dept) > 1:
        st.subheader("Department Summary")
        st.dataframe(dept, use_container_width=True)


# ── HR: Alerts ────────────────────────────────────────────────────────────────
elif is_hr and page == "🔔 HR Alerts":
    page_header("🔔 HR Stress Alerts", "Monitor and acknowledge employee stress events")

    tab1, tab2 = st.tabs(["🔴 Unread", "📋 All Alerts"])

    with tab1:
        unread = get_unacknowledged_alerts()
        if not unread:
            st.success("✅ No unread alerts — team appears to be doing well!")
        else:
            st.warning(f"{len(unread)} alert(s) require attention.")
            for alert in unread:
                with st.container(border=True):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"### {alert_emoji(alert['alert_type'])} "
                                    f"{alert['alert_type'].upper()}")
                        st.markdown(f"**Employee:** {alert['employee_name']} "
                                    f"({alert['department']})")
                        st.markdown(f"**Message:** {alert['message']}")
                        st.caption(f"Reported at: {alert['created_at']}")
                    with col2:
                        if st.button("✅ Acknowledge", key=f"ack_{alert['id']}",
                                     use_container_width=True):
                            acknowledge_alert(alert["id"])
                            st.rerun()

    with tab2:
        all_alerts = get_all_alerts()
        if not all_alerts:
            st.info("No alerts recorded yet.")
        else:
            df_a = pd.DataFrame(all_alerts)
            df_a["status"] = df_a["acknowledged"].map({0: "🔴 Unread", 1: "✅ Done"})
            counts = df_a["alert_type"].value_counts().reset_index()
            counts.columns = ["type", "count"]
            fig = px.bar(counts, x="type", y="count", color="type",
                         color_discrete_map={"stress": "#e74c3c",
                                             "burnout": "#e67e22",
                                             "disengagement": "#f39c12"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                df_a[["created_at", "employee_name", "department",
                       "alert_type", "stress_level", "message", "status"]],
                use_container_width=True,
            )


# ── HR: Employee Management ───────────────────────────────────────────────────
elif is_hr and page == "👤 Employee Management":
    page_header("👤 Employee Management", "View, monitor, and manage registered employees")

    users = get_all_users()
    employees = [u for u in users if u["role"] == "employee"]
    hr_users  = [u for u in users if u["role"] == "hr"]

    col1, col2 = st.columns(2)
    col1.metric("👷 Employees", len(employees))
    col2.metric("🔑 HR Managers", len(hr_users))

    st.divider()
    st.subheader("All Employees")

    if employees:
        risk_list = bulk_burnout_risk(employees)
        risk_map  = {r["id"]: r for r in risk_list}

        for emp in employees:
            risk = risk_map.get(emp["id"], {})
            score = risk.get("score", 0)
            color = risk.get("color", "#95a5a6")
            label = risk.get("label", "No Data")

            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([2.5, 1.5, 2, 1])
                with col1:
                    st.markdown(f"**{emp['name']}** ({emp['employee_id']})")
                    st.caption(f"Dept: {emp['department']} · Joined: {emp['created_at'][:10]}")
                with col2:
                    st.markdown(
                        f"<div style='padding:6px 10px;border-radius:8px;"
                        f"background:{color};color:white;text-align:center;"
                        f"font-weight:bold'>{score}/100</div>",
                        unsafe_allow_html=True
                    )
                    st.caption(label)
                with col3:
                    if risk.get("factors"):
                        f = risk["factors"]
                        st.caption(f"Avg stress: {risk.get('avg_stress_raw',0)}/10 · "
                                   f"Consecutive: {risk.get('consecutive_raw',0)} · "
                                   f"Missed: {risk.get('missed_raw',0)} days")
                with col4:
                    if st.button("🗑️ Remove", key=f"del_{emp['id']}",
                                 use_container_width=True):
                        delete_user(emp["id"])
                        st.success(f"Removed {emp['name']}")
                        st.rerun()
    else:
        st.info("No employees registered yet.")