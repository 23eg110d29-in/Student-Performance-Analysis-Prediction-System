# Student Performance Analysis & Prediction System

An end-to-end Machine Learning, Data Science, and interactive visualization system designed to analyze lifestyle behaviors, academic habits, and student engagement factors, and predict final exam scores.

---

## 🌟 Features

*   **Synthetic Data Generation**: Creates a robust, statistically sound dataset of 1200 student records with realistic correlations, random noise, and injected anomalies (missing data and duplicate rows).
*   **Preprocessing Pipeline**: Handles missing data imputation, duplicate dropping, type casting, outlier checking, and value boundary validation automatically.
*   **Exploratory Data Analysis (EDA)**: Produces 8 distinct high-fidelity visualizations saved locally to `screenshots/` and provides automated statistical narratives.
*   **Predictive Model (Random Forest)**: An optimized `RandomForestRegressor` (tuned via Grid Search over tree depths) achieving high predictive accuracy.
*   **SQLite Database Integration**: Integrates SQLite databases to persist the primary student records and record prediction query histories dynamically.
*   **Streamlit Dashboard**: A high-fidelity, premium dark-themed web dashboard containing:
    *   *Dashboard*: High-level KPI summary cards and student status counts.
    *   *Data Analysis*: Interactive data tables, statistical summaries, and dynamic charts.
    *   *Model Performance*: Hyperparameter logs, error metrics (MAE, RMSE, R²), and Gini feature importances.
    *   *Predict Score*: Interactive prediction form with sliders/inputs and input validation.
    *   *Prediction History*: Detailed log logs and direct CSV downloader.
*   **Success Recommendations**: Intelligent rules engine offering custom guidance based on student lifestyle inputs (sleep, study, social media).

---

## 🛠️ Technology Stack

*   **Frontend**: Streamlit
*   **Backend**: Python
*   **Database**: SQLite
*   **Data Analysis**: Pandas, NumPy
*   **Visualization**: Matplotlib, Seaborn
*   **Machine Learning**: Scikit-Learn (Random Forest Regressor)
*   **Model Serialization**: Joblib

---

## 📁 Project Structure

```text
Student-Performance-Prediction/
├── streamlit_app.py            # Streamlit application dashboard
├── data/
│   └── students.csv            # Generated synthetic dataset
├── database/
│   └── student_performance.db  # SQLite database storing data & predictions
├── model/
│   ├── model.pkl               # Serialized RandomForest model
│   └── metrics.json            # Model performance logs
├── notebooks/
│   └── analysis.ipynb          # Jupyter notebook outlining execution steps
├── screenshots/                # Saved EDA and feature importance charts
│   ├── study_vs_final.png
│   ├── correlation_heatmap.png
│   ├── participation_impact.png
│   ├── attendance_dist.png
│   ├── sleep_dist.png
│   ├── social_media_dist.png
│   ├── prev_vs_final.png
│   └── feature_importance.png
├── src/
│   ├── __init__.py
│   ├── data_generator.py      # Dataset generation logic
│   ├── preprocess.py          # Data cleaning pipeline
│   ├── eda.py                 # Visualization and analytics generator
│   ├── train_model.py         # Model training and optimization
│   ├── predictor.py           # Model inference and validation wrapper
│   ├── database.py            # SQLite database connector
│   └── utils.py               # Custom styles and utility functions
├── requirements.txt            # Python environment packages
├── README.md                   # System documentation
└── .gitignore                  # Git files exclusion configuration
```

---

## 📊 Dataset Schema

The generated dataset contains the following attributes for each student:

| Column Name | Data Type | Range | Description |
| :--- | :--- | :--- | :--- |
| **StudentID** | Text | STU0001 - STU1200 | Unique primary key identifying the student |
| **StudyHoursPerDay** | Float | 1.0 - 10.0 | Daily hours dedicated to self-study |
| **AttendancePercentage** | Float | 50.0 - 100.0 | Attendance percentage across the semester |
| **SleepHours** | Float | 4.0 - 10.0 | Average hours of sleep per night |
| **SocialMediaHours** | Float | 0.0 - 8.0 | Daily average screen time on social media |
| **PreviousExamScore** | Float | 40.0 - 100.0 | Score on previous assessment tests |
| **ParticipationInActivities** | Integer | 0 (No) / 1 (Yes) | Extracurricular program participation |
| **InternetUsageHours** | Float | 0.0 - 8.0 | Daily hours browsing the internet |
| **FinalExamScore** | Float | 0.0 - 100.0 | Target Variable: Calculated final exam mark |

---

## ⚡ Setup & Installation

Follow these steps to set up the environment and run the application:

### Prerequisites
*   Python 3.8 or higher installed.

### 1. Clone or copy files
Ensure the files are placed in your working directory.

### 2. Create and Activate Virtual Environment
Open a terminal in the project root directory:

**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1: Initializing data, database, and model
To execute the pipeline end-to-end (generate raw data, clean it, initialize database, generate plots, and train model), run:

```bash
# Activate virtual environment if not already activated
.\venv\Scripts\Activate.ps1   # (Windows)
# source venv/bin/activate    # (macOS/Linux)

# Run full pipeline
python src/data_generator.py
python src/preprocess.py
python src/database.py
python src/eda.py
python src/train_model.py
```

*Alternatively, the Streamlit app contains a setup wizard button on its dashboard tab if no database is found.*

### Step 2: Running the Web App
Start the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 🧠 EDA Insights & Model Performance

### Exploratory Insights
*   **Positive Drivers**: Attendance percentage, daily study hours, and previous academic scores show the strongest positive impact on final performance. Extracurricular participants achieve, on average, a 3% boost in performance.
*   **Negative Impact**: Excessive screen time on social media (>3 hrs/day) and internet surfing (>4 hrs/day) degrade performance linearly due to screen fatigue and distraction.
*   **Rest Balance**: Optimal sleep duration (between 6.5 to 8.5 hours) correlates with high academic retention.

### Predictive Model Metrics
*   **Algorithm**: RandomForestRegressor
*   **Evaluated Trees**: 200
*   **R² Score**: ~0.92+ (Model explains 92%+ of the target variable variance)
*   **MAE**: ~2.8 points (Average prediction is within 2.8% of the student's actual score)
*   **RMSE**: ~3.5 points (Standard deviation of prediction residuals)
