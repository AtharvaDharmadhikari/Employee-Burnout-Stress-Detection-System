"""
Employee Burnout & Stress Detection System
Main Streamlit dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from database import (
    init_db, add_employee, get_all_employees,
    get_employee_by_name, log_mood, get_mood_history,
    log_task, get_task_history,
    get_unacknowledged_alerts, acknowledge_alert, get_all_alerts,
    delete_employee,
)
from emotion_detection import (
    MOOD_OPTIONS, manual_detection, streamlit_webcam_widget, mood_to_stress,
)
from task_recommendation import recommend_task
from task_duration import predict_task_duration, PRIORITY_OPTIONS
from stress_alerts import (
    evaluate_stress, alert_emoji, stress_level_label, stress_color,
)
from team_analytics import (
    load_mood_dataframe, mood_distribution, mood_trend_over_time,
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

# ── Init DB ───────────────────────────────────────────────────────────────────

init_db()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=64)
    st.title("Burnout & Stress")
    st.caption("Employee Burnout & Stress Detection System")

    st.divider()

    # Employee selector / creator
    st.subheader("👤 Employee")
    employees = get_all_employees()
    emp_names = [e["name"] for e in employees]

    new_name = st.text_input("Add new employee", placeholder="Full name")
    new_dept = st.text_input("Department", placeholder="e.g. Engineering", value="General")
    if st.button("➕ Add Employee", use_container_width=True):
        if new_name.strip():
            add_employee(new_name.strip(), new_dept.strip() or "General")
            st.success(f"Added {new_name}")
            st.rerun()
        else:
            st.warning("Please enter a name.")

    st.divider()

    employees = get_all_employees()
    emp_names = [e["name"] for e in employees]
    if emp_names:
        selected_name = st.selectbox("Select Employee", emp_names)
        selected_emp = get_employee_by_name(selected_name)

        # Remove employee
        with st.expander("🗑️ Remove Employee"):
            st.warning(f"This will permanently delete **{selected_name}** and all their mood, task, and alert data.")
            if st.button("Delete Employee", type="primary", use_container_width=True):
                delete_employee(selected_emp["id"])
                st.success(f"{selected_name} removed.")
                st.rerun()
    else:
        st.info("Add an employee to get started.")
        selected_emp = None
        selected_name = None

    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "😊 Mood Check-In", "✅ Task Manager",
         "📈 Mood History", "🔔 HR Alerts", "👥 Team Analytics"],
        label_visibility="collapsed",
    )

# ── Unread alert badge in sidebar ─────────────────────────────────────────────
unread = get_unacknowledged_alerts()
if unread:
    st.sidebar.warning(f"⚠️ {len(unread)} unread HR alert(s)")

# ── Pages ─────────────────────────────────────────────────────────────────────

# ── 1. Dashboard ──────────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.title("🧠 Employee Burnout & Stress Detection System")
    st.markdown("Real-time emotion-driven task management and employee wellness platform.")

    col1, col2, col3, col4 = st.columns(4)
    employees_all = get_all_employees()
    df_all = load_mood_dataframe()
    morale = team_morale_score(df_all)
    alerts_all = get_unacknowledged_alerts()

    col1.metric("👥 Employees", len(employees_all))
    col2.metric("📊 Total Check-Ins", morale["total_logs"])
    col3.metric("🎯 Team Morale Score", f"{morale['score']}/100", morale["label"])
    col4.metric("🔔 Unread Alerts", len(alerts_all))

    st.divider()

    if not df_all.empty:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Mood Distribution (All Time)")
            dist = mood_distribution(df_all)
            fig = px.pie(dist, names="mood", values="count",
                         color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Team Stress Trend (Daily Avg)")
            trend = mood_trend_over_time(df_all, freq="D")
            if not trend.empty:
                fig2 = px.line(trend, x="date", y="avg_stress",
                               labels={"avg_stress": "Avg Stress (0-10)", "date": "Date"},
                               markers=True, color_discrete_sequence=["#e74c3c"])
                fig2.add_hline(y=7, line_dash="dash", line_color="red",
                               annotation_text="Alert Threshold")
                fig2.update_layout(margin=dict(t=20, b=20), yaxis_range=[0, 10])
                st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Employee Stress Ranking")
        ranking = employee_stress_ranking(df_all)
        if not ranking.empty:
            fig3 = px.bar(ranking, x="employee_name", y="avg_stress",
                          color="avg_stress",
                          color_continuous_scale=["green", "yellow", "red"],
                          range_color=[0, 10],
                          labels={"avg_stress": "Avg Stress", "employee_name": "Employee"})
            fig3.update_layout(margin=dict(t=20, b=20), coloraxis_showscale=False)
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No mood data yet. Use **Mood Check-In** to get started.")

    st.divider()
    st.subheader("🚀 Features")
    fcol1, fcol2, fcol3 = st.columns(3)
    fcol1.info("**😊 Emotion Detection**\nWebcam or manual mood input with DeepFace AI")
    fcol2.info("**✅ Task Recommendation**\nML-powered task suggestions based on mood")
    fcol3.info("**⏱ Duration Prediction**\nEstimate task time using priority & deadline")
    fcol1.success("**📈 Mood History**\nPersonal mood timeline and trend charts")
    fcol2.success("**🔔 Stress Alerts**\nAutomatic HR notifications on high stress")
    fcol3.success("**👥 Team Analytics**\nAggregate mood insights across your whole team")


# ── 2. Mood Check-In ──────────────────────────────────────────────────────────
elif page == "😊 Mood Check-In":
    st.title("😊 Mood Check-In")
    if not selected_emp:
        st.warning("Please add and select an employee first.")
        st.stop()

    st.markdown(f"Checking in as **{selected_name}** ({selected_emp['department']})")
    st.divider()

    tab1, tab2 = st.tabs(["📷 Webcam Detection", "✏️ Manual Entry"])

    detection_result = None

    with tab1:
        st.markdown("Take a photo and our AI will detect your emotion automatically.")
        result = streamlit_webcam_widget()
        if result:
            detection_result = result
            if "error" in result:
                st.warning(f"Detection issue: {result['error']} — mood defaulted to Neutral.")
            else:
                src = result.get("source", "")
                conf = result.get("confidence")
                conf_text = f" — Confidence: {conf}%" if conf else ""
                fallback_note = " *(DeepFace fallback)*" if "deepface" in src else ""
                st.success(f"Detected mood: **{result['mood']}** "
                           f"(Stress: {result['stress_level']}/10){conf_text}{fallback_note}")

                # Show per-emotion confidence bar chart
                raw = result.get("raw_emotions")
                if raw and "deepface" not in src:
                    import pandas as pd
                    df_em = pd.DataFrame(
                        {"Emotion": list(raw.keys()),
                         "Confidence": [round(v * 100, 1) for v in raw.values()]}
                    ).sort_values("Confidence", ascending=False)
                    st.bar_chart(df_em.set_index("Emotion"))

    with tab2:
        st.markdown("Select your current mood and workload manually.")
        manual_mood = st.selectbox("How are you feeling?", MOOD_OPTIONS)
        manual_stress = st.slider("Stress level", 0, 10,
                                  value=mood_to_stress(manual_mood),
                                  help="0 = no stress, 10 = extreme stress")
        notes = st.text_area("Optional notes", placeholder="Anything on your mind?")
        if st.button("Submit Manual Check-In", use_container_width=True, type="primary"):
            detection_result = {
                "mood": manual_mood,
                "stress_level": manual_stress,
                "source": "manual",
            }

    if detection_result:
        mood = detection_result["mood"]
        stress = detection_result["stress_level"]
        source = detection_result.get("source", "manual")
        notes_val = notes if "notes" in dir() else ""

        # Save mood log
        log_mood(selected_emp["id"], mood, stress, source, notes_val)

        # Evaluate for stress alerts
        triggered = evaluate_stress(selected_emp["id"], selected_name, stress, mood)

        # Show result card
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            color = stress_color(stress)
            st.markdown(f"""
            <div style='padding:16px; border-radius:10px; border: 2px solid {color};'>
                <h3>Current State</h3>
                <p><b>Mood:</b> {mood}</p>
                <p><b>Stress Level:</b>
                   <span style='color:{color}'>{stress}/10 — {stress_level_label(stress)}</span>
                </p>
                <p><b>Source:</b> {source.capitalize()}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Quick task recommendation
            workload = st.slider("Current workload (for recommendation)", 1, 10, 5,
                                 key="checkin_workload")
            rec = recommend_task(mood, workload)
            st.markdown(f"""
            <div style='padding:16px; border-radius:10px; background:#f0f9f0; border: 2px solid green;'>
                <h3>Recommended Task</h3>
                <p><b>{rec['task']}</b></p>
                <p style='color:gray'>{rec['description']}</p>
            </div>
            """, unsafe_allow_html=True)

        # Stress alerts
        for alert in triggered:
            st.error(f"{alert_emoji(alert['type'])} **{alert['type'].upper()} ALERT** — {alert['message']}")


# ── 3. Task Manager ───────────────────────────────────────────────────────────
elif page == "✅ Task Manager":
    st.title("✅ Task Manager")
    if not selected_emp:
        st.warning("Please add and select an employee first.")
        st.stop()

    tab1, tab2 = st.tabs(["🎯 Get Recommendation", "⏱ Duration Predictor"])

    with tab1:
        st.subheader("Task Recommendation")
        st.markdown("Enter your current mood and workload to get a personalized task suggestion.")

        col1, col2 = st.columns(2)
        with col1:
            rec_mood = st.selectbox("Current Mood", MOOD_OPTIONS, key="rec_mood")
            rec_workload = st.slider("Current Workload (1=light, 10=overloaded)", 1, 10, 5,
                                     key="rec_workload")
        with col2:
            rec_priority = st.selectbox("Task Priority", PRIORITY_OPTIONS, index=1,
                                        key="rec_priority")
            rec_deadline = st.number_input("Days until deadline", 1, 365, 7, key="rec_deadline")

        if st.button("Get Recommendation", type="primary", use_container_width=True):
            rec = recommend_task(rec_mood, rec_workload)
            st.success(f"**Recommended:** {rec['task']}")
            st.info(rec['description'])

            log_task(
                selected_emp["id"], rec_mood, rec_workload,
                rec["task"], rec_priority, rec_deadline
            )
            st.caption("Task saved to your history.")

    with tab2:
        st.subheader("Task Duration Predictor")
        st.markdown("Estimate how long a task will take based on its description and context.")

        task_desc = st.text_input("Task description",
                                  placeholder="e.g. Implement user authentication module")
        col1, col2, col3 = st.columns(3)
        with col1:
            dur_priority = st.selectbox("Priority", PRIORITY_OPTIONS, index=1, key="dur_priority")
        with col2:
            dur_deadline = st.number_input("Days until deadline", 1, 365, 7, key="dur_deadline")
        with col3:
            dur_workload = st.slider("Current workload", 1, 10, 5, key="dur_workload")

        if st.button("Predict Duration", type="primary", use_container_width=True):
            if task_desc.strip():
                result = predict_task_duration(
                    task_desc, dur_priority, dur_deadline, dur_workload
                )
                st.metric("Estimated Duration", result["label"])
                st.caption(f"Raw estimate: {result['estimated_hours']} hours")
            else:
                st.warning("Please enter a task description.")

    st.divider()
    st.subheader("📋 Task History")
    history = get_task_history(selected_emp["id"])
    if history:
        df_tasks = pd.DataFrame(history)
        st.dataframe(
            df_tasks[["created_at", "mood", "workload", "recommended_task",
                       "priority", "deadline_days", "estimated_duration", "status"]],
            use_container_width=True,
        )
    else:
        st.info("No tasks recorded yet.")


# ── 4. Mood History ───────────────────────────────────────────────────────────
elif page == "📈 Mood History":
    st.title("📈 Mood History")
    if not selected_emp:
        st.warning("Please add and select an employee first.")
        st.stop()

    st.markdown(f"Historical mood data for **{selected_name}**")

    history = get_mood_history(selected_emp["id"], limit=200)
    if not history:
        st.info("No mood logs yet. Check in on the Mood Check-In page.")
        st.stop()

    df = pd.DataFrame(history)
    df["logged_at"] = pd.to_datetime(df["logged_at"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Check-Ins", len(df))
    col2.metric("Avg Stress Level", f"{df['stress_level'].mean():.1f}/10")
    col3.metric("Most Common Mood", df["mood"].mode()[0])

    st.divider()

    # Stress over time
    st.subheader("Stress Level Over Time")
    fig = px.line(df, x="logged_at", y="stress_level", markers=True,
                  labels={"stress_level": "Stress Level", "logged_at": "Date"},
                  color_discrete_sequence=["#3498db"])
    fig.add_hline(y=7, line_dash="dash", line_color="red",
                  annotation_text="Alert Threshold (7)")
    fig.update_layout(yaxis_range=[0, 10])
    st.plotly_chart(fig, use_container_width=True)

    # Mood distribution pie
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mood Distribution")
        mood_counts = df["mood"].value_counts().reset_index()
        mood_counts.columns = ["mood", "count"]
        fig2 = px.pie(mood_counts, names="mood", values="count",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Stress by Mood")
        fig3 = px.box(df, x="mood", y="stress_level",
                      color="mood",
                      labels={"stress_level": "Stress Level", "mood": "Mood"})
        fig3.update_layout(showlegend=False, yaxis_range=[0, 10])
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Full Log")
    st.dataframe(
        df[["logged_at", "mood", "stress_level", "source", "notes"]],
        use_container_width=True,
    )


# ── 5. HR Alerts ─────────────────────────────────────────────────────────────
elif page == "🔔 HR Alerts":
    st.title("🔔 HR Stress Alerts")

    tab1, tab2 = st.tabs(["🔴 Unread Alerts", "📋 All Alerts"])

    with tab1:
        unread_alerts = get_unacknowledged_alerts()
        if not unread_alerts:
            st.success("✅ No unread alerts — all employees appear to be doing well!")
        else:
            st.warning(f"{len(unread_alerts)} alert(s) require attention.")
            for alert in unread_alerts:
                with st.container(border=True):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        emoji = alert_emoji(alert["alert_type"])
                        st.markdown(f"### {emoji} {alert['alert_type'].upper()}")
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
        all_alerts = get_all_alerts(limit=200)
        if not all_alerts:
            st.info("No alerts recorded yet.")
        else:
            df_alerts = pd.DataFrame(all_alerts)
            df_alerts["status"] = df_alerts["acknowledged"].map(
                {0: "🔴 Unread", 1: "✅ Acknowledged"}
            )

            # Summary chart
            alert_counts = df_alerts["alert_type"].value_counts().reset_index()
            alert_counts.columns = ["type", "count"]
            fig = px.bar(alert_counts, x="type", y="count",
                         color="type",
                         color_discrete_map={
                             "stress": "#e74c3c",
                             "burnout": "#e67e22",
                             "disengagement": "#f39c12",
                         },
                         labels={"type": "Alert Type", "count": "Count"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                df_alerts[["created_at", "employee_name", "department",
                            "alert_type", "stress_level", "message", "status"]],
                use_container_width=True,
            )


# ── 6. Team Analytics ─────────────────────────────────────────────────────────
elif page == "👥 Team Analytics":
    st.title("👥 Team Mood Analytics")

    df_all = load_mood_dataframe()

    if df_all.empty:
        st.info("No team data yet. Have employees check in on the Mood Check-In page.")
        st.stop()

    morale = team_morale_score(df_all)

    # Morale gauge
    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Team Morale Score", f"{morale['score']}/100", morale["label"])
    col2.metric("📊 Avg Team Stress", f"{morale['avg_stress']}/10")
    col3.metric("📋 Total Check-Ins", morale["total_logs"])

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
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": morale["score"],
            },
        },
    ))
    fig_gauge.update_layout(height=250, margin=dict(t=30, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Mood Distribution (Last 7 Days)")
        recent = recent_mood_counts(df_all, days=7)
        if not recent.empty:
            fig = px.pie(recent, names="mood", values="count",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data in the last 7 days.")

    with col2:
        st.subheader("Daily Average Stress Trend")
        trend = mood_trend_over_time(df_all, freq="D")
        if not trend.empty:
            fig2 = px.area(trend, x="date", y="avg_stress",
                           labels={"avg_stress": "Avg Stress", "date": "Date"},
                           color_discrete_sequence=["#e74c3c"])
            fig2.add_hline(y=7, line_dash="dash", line_color="red",
                           annotation_text="Alert Threshold")
            fig2.update_layout(yaxis_range=[0, 10])
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Employee Stress Ranking")
    ranking = employee_stress_ranking(df_all)
    if not ranking.empty:
        fig3 = px.bar(
            ranking, x="employee_name", y="avg_stress",
            color="avg_stress",
            color_continuous_scale=["green", "yellow", "red"],
            range_color=[0, 10],
            text="avg_stress",
            labels={"avg_stress": "Avg Stress (0-10)", "employee_name": "Employee"},
        )
        fig3.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig3.update_layout(coloraxis_showscale=False, yaxis_range=[0, 11])
        st.plotly_chart(fig3, use_container_width=True)

    dept_summary = department_mood_summary(df_all)
    if not dept_summary.empty and len(dept_summary) > 1:
        st.subheader("Department Summary")
        st.dataframe(dept_summary, use_container_width=True)
