# Employee Burnout & Stress Detection System

An AI-driven workplace wellness monitoring platform that detects employee stress and burnout risk in real time using facial emotion recognition, behavioral analytics, and machine learning.

---

## 1) Project Overview 

### Problem
Organizations detect burnout too late because they rely on delayed surveys/manual HR observation.

### Solution
Daily attendance is combined with mood/stress capture (webcam or manual), then used to:
- detect high-risk stress patterns in near real time,
- recommend safer/more suitable task types,
- provide HR with actionable dashboards and alerts.

### Users
- **Employee**: personal dashboard, check-in/check-out, task recommendation, wellness history
- **HR manager**: team dashboard, alerts queue, attendance monitoring, employee risk management

### Core Value
- Early warning for burnout risk
- Role-based visibility (employee privacy + HR intervention capability)
- Data-driven wellness decisions instead of reactive follow-up

---

## 2) Implemented Features

### Employee Features
1. Secure login/registration (role assigned at registration)
2. Attendance check-in/check-out with mood and stress capture
3. Webcam emotion detection API and manual fallback
4. Mood/stress logging over time
5. AI task recommendation based on mood + workload + urgency
6. Personal wellness report (trend, distribution, stress-by-mood, log table)
7. Burnout risk score shown on dashboard

### HR Features
1. Team wellness dashboard with KPIs and charts
2. Unread and historical stress alerts with acknowledge workflow
3. Today attendance view (present, in-office, absent)
4. Employee management with burnout risk badges and delete action
5. Team morale score and stress ranking
6. Department-level summary

---

## 3) System Architecture

### Presentation Layer
- Flask server-rendered templates (`templates/*`) + Bootstrap + custom CSS/JS
- Streamlit single-file dashboard (`app.py`)
- Plotly charts for trend/gauge/pie/bar visualizations

### API / Interaction Layer (Flask)
- `POST /api/detect-emotion`: receives base64 image, runs emotion detection
- `POST /api/save-emotion`: saves check-in/check-out emotion and logs
- `POST /api/checkin-manual`: manual mood submit
- `GET /api/ping`: app/session health check

### Domain Logic Layer
- `auth.py`: authentication and role assignment
- `emotion_detection.py`: DeepFace + Keras fallback
- `task_recommendation.py`: Decision Tree classifier for task type
- `stress_alerts.py`: stress/burnout/disengagement trigger logic
- `burnout_risk.py`: hybrid burnout score (ML + behavior)
- `team_analytics.py`: aggregation for team dashboards

### Data Layer
- SQLite database: `data/task_optimizer.db` (auto-created)
- Schema managed in `database.py`

---

## 4) Database Schema

### `users`
- identity + auth + role
- key fields: `employee_id`, `name`, `password_hash`, `department`, `role`, `created_at`

### `attendance`
- one row per user per day
- key fields: `date`, `check_in_time`, `check_out_time`, `entry_mood`, `entry_stress`, `exit_mood`, `exit_stress`

### `mood_logs`
- event-level mood history
- key fields: `mood`, `stress_level`, `source`, `notes`, `logged_at`

### `task_logs`
- recommendation history
- key fields: `mood`, `workload`, `recommended_task`, `priority`, `deadline_days`, `status`, `created_at`

### `hr_alerts`
- alert queue for HR follow-up
- key fields: `alert_type`, `stress_level`, `message`, `acknowledged`, `created_at`

---

## 5) AI/ML Components (Exact)

### A) Emotion Detection (`emotion_detection.py`)
- **Primary**: DeepFace emotion analysis
- **Fallback**: `model_optimal.keras` (48x48 grayscale, FER-style classes)
- Returns: `mood`, `stress_level`, `source`, optional `confidence`, raw emotion map
- **Model Training Notebook**: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hy39pxCQ0GL91zfayg-d3xv6omqoDuKM#scrollTo=8h610RzShRrM)
- **Training Dataset**: [FER2013 — Facial Expression Recognition (Kaggle)](https://www.kaggle.com/datasets/msambare/fer2013?resource=download)

### B) Mood to Stress Mapping (0-10)
- Calm=1, Happy=2, Neutral=4, Surprised=5, Tired=6, Sad=6, Fear=7, Angry=8, Disgust=8, Stressed=9

### C) Task Recommendation (`task_recommendation.py`)
- Model: `DecisionTreeClassifier(random_state=42)`
- Inputs:
  - mood (encoded),
  - workload (1-10),
  - priority (Low/Medium/High/Critical -> 1..4),
  - deadline bucket (Relaxed/Normal/Urgent -> 1..3)
- Output task classes:
  - Creative Work, Collaborative Work, Strategic Planning, Deep Work,
  - Documentation, Simple Tasks, Relaxation Break, Mindfulness Activity
- Training source: curated rule-driven dataset in code (`TRAINING_DATA`)

### D) Stress Alerts (`stress_alerts.py`)
Thresholds:
- Immediate stress alert: `current_stress >= 7`
- Burnout streak alert: last 3 readings all `>= 6`
- Disengagement alert: average of last 5 readings `>= 6.5`

Alert types:
- `stress`, `burnout`, `disengagement`

### E) Burnout Risk Scoring (`burnout_risk.py`)
- Hybrid scoring:
  - 60% ML score (if trained model exists),
  - 40% behavioral formula
- If ML model unavailable: uses formula-only score.

Behavioral factor weights:
- Recent average stress (last 14): 30%
- Consecutive high stress (>=6): 25%
- Stress trend slope: 20%
- Negative mood ratio: 15%
- Missed check-ins (last 7 days): 10%

Risk bands:
- 0-24: Low
- 25-49: Moderate
- 50-74: High
- 75-100: Critical

### F) Burnout ML Training (`train_burnout_model.py`)
- Dataset: Kaggle "Are Your Employees Burning Out?"
- Model: `GradientBoostingRegressor`
- Features:
  - Mental Fatigue Score
  - Resource Allocation
- Target: Burn Rate (0-1)
- Saves:
  - `models/burnout_model.pkl`
  - `models/burnout_model_meta.pkl`

### G) Task Duration Module (`task_duration.py`)
- Present in repository but **not integrated into active app routes**.

---

## 6) Security and Access Control

- Password hashing: PBKDF2-HMAC-SHA256 with random salt and 200,000 iterations
- Role assignment via HR access code at registration:
  - blank code -> employee
  - correct HR code -> HR
  - incorrect code -> registration rejected
- Flask session-based auth and route decorators for role restrictions

Important current hardcoded secrets (should be moved to env vars for production):
- `auth.py`: `HR_SECRET_CODE = "HR@2024"`
- `flask_app.py`: `app.secret_key = "burnout_detection_secret_2024"`

---

## 7) End-to-End Workflow

1. User registers/logs in.
2. Employee checks in (webcam or manual mood).
3. Mood/stress saved into attendance + mood logs.
4. Stress rules evaluate and may create HR alerts.
5. Employee receives task recommendation.
6. Employee checks out with end-of-day mood.
7. HR dashboards aggregate trends, morale, rankings, and burnout risk.

---

## 8) Tech Stack

- Python
- Flask, Jinja2, Bootstrap 5 (web frontend)
- Streamlit (alternate frontend)
- SQLite (`sqlite3`)
- scikit-learn (Decision Tree, Gradient Boosting)
- TensorFlow/Keras (fallback emotion model)
- DeepFace + OpenCV + Pillow (emotion pipeline)
- Pandas + NumPy
- Plotly

---

## 9) Repository Structure

```text
AI Task Optimization/
  app.py                       # Streamlit app
  flask_app.py                 # Flask app entrypoint
  database.py                  # DB schema + CRUD helpers
  auth.py                      # Registration/login/password hashing
  emotion_detection.py         # Emotion inference + stress mapping
  task_recommendation.py       # ML task recommendation
  stress_alerts.py             # Alert trigger logic
  team_analytics.py            # Team-level aggregations
  burnout_risk.py              # Hybrid burnout scoring
  train_burnout_model.py       # Kaggle model training script
  task_duration.py             # Standalone duration predictor (not wired)
  static/
    css/style.css
    js/webcam.js
  templates/
    base.html
    login.html
    register.html
    employee/*.html
    hr/*.html
  data/
    task_optimizer.db          # auto-created runtime DB
  models/
    burnout_model.pkl          # created after training
    burnout_model_meta.pkl     # created after training
```

---

## 10) Setup and Run

## Prerequisites
- Python 3.10+ recommended
- Webcam access (for emotion capture flow)

## Install
```bash
python -m venv .venv
.venv\Scripts\activate
pip install flask streamlit pandas numpy plotly scikit-learn pillow opencv-python deepface tensorflow joblib
```

If you already have a `requirements.txt`, use:
```bash
pip install -r requirements.txt
```

## Run Flask App (recommended)
```bash
python flask_app.py
```
Open: `http://127.0.0.1:5000`

---

## 11) Optional: Train Burnout ML Model

1. Download Kaggle CSV (`train.csv`)
2. Place it at `data/train.csv`
3. Run:
```bash
python train_burnout_model.py
```
4. Confirm files generated:
- `models/burnout_model.pkl`
- `models/burnout_model_meta.pkl`

Without these files, burnout scoring still works using formula-only mode.

---

## 12) Known Limitations

- Mood capture is point-in-time (single image), not continuous monitoring.
- Self-reported workload can be subjective.
- SQLite is suitable for prototype/small deployment, not high concurrency production.
- Hardcoded secrets should be migrated to environment configuration.
- Ethical/privacy governance (consent, retention policy, access audit) must be formalized before production rollout.

---

## 13) Suggested 40-Page PPT Outline (from this README)

1. Title, motivation, problem
2. Burnout context and business impact
3. Project objectives and scope
4. User roles and use cases
5. System architecture overview
6. Flask and Streamlit interface comparison
7. Database design
8. Emotion detection pipeline
9. Mood-to-stress mapping
10. Task recommendation model design
11. Burnout hybrid model design
12. Stress alert thresholds and logic
13. HR analytics and decision support
14. Security model
15. Workflow walkthrough
16. Limitations, risks, ethics, and future roadmap

---

## 14) Quick Demo Checklist

- Register one HR user (with HR code)
- Register 2-3 employee users
- Run check-in/check-out with varied moods
- Trigger stress alerts (stress >= 7 and repeated high stress)
- Open HR dashboard and alerts page
- Show burnout risk ranking and attendance status

---
