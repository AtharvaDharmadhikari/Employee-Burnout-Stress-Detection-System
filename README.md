# 🧠 Employee Burnout & Stress Detection System

An AI-powered system that analyzes employee emotions and moods in real-time to detect stress and burnout early, recommend appropriate tasks, and notify HR when intervention is needed.

---

## Features

1. **Real-Time Emotion Detection** — Detects employee emotions via webcam using a custom trained Keras model (48×48 grayscale) with DeepFace as fallback
2. **Task Recommendation** — ML model suggests tasks based on detected mood and current workload (e.g. Creative Work, Deep Work, Relaxation Break)
3. **Historical Mood Tracking** — Personal mood timeline with trend charts to identify long-term well-being patterns
4. **Stress Management Alerts** — Automatically triggers HR alerts when stress exceeds thresholds (single reading, consecutive highs, rolling average)
5. **Team Mood Analytics** — Aggregates mood data across teams with a morale score, department breakdown, and productivity trends
6. **Task Duration Prediction** — Estimates task completion time based on description, priority, and deadline

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Emotion Detection | Custom Keras Model + OpenCV + DeepFace |
| ML Models | scikit-learn (DecisionTree, GradientBoosting) |
| Database | SQLite |
| Charts | Plotly |
| Language | Python 3.10 |

---

## Project Structure

```
├── app.py                  # Main Streamlit dashboard (6 pages)
├── database.py             # SQLite setup and all queries
├── emotion_detection.py    # Custom Keras model + DeepFace fallback
├── task_recommendation.py  # Task recommendation ML model
├── task_duration.py        # Task duration prediction ML model
├── stress_alerts.py        # Stress/burnout alert logic
├── team_analytics.py       # Team mood aggregations and charts
├── requirements.txt        # Python dependencies
└── data/                   # SQLite database (auto-created at runtime)
```

---

## Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. Create and activate virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Model Setup

The emotion detection model (`model_optimal.keras`) is not included in this repository due to its size.

**Download:** [model_optimal.keras — Google Drive](https://drive.google.com/file/d/1PhGu7v6lssv8tfaxmn8I5mmvXD9Cs6tJ/view?usp=sharing)

After downloading, place the file in the **project root folder**:
```
your-repo-name/
└── model_optimal.keras   ← here
```

> The model detects 7 emotions: `angry`, `disgust`, `fear`, `happy`, `sad`, `surprised`, `neutral`
> Input: 48×48 grayscale image — Accuracy: ~70%

---

## Run the App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.