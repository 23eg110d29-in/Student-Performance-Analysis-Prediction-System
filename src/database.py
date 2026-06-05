import os
import sqlite3
import pandas as pd
from datetime import datetime

DEFAULT_DB_PATH = os.path.join('database', 'student_performance.db')

def get_db_connection(db_path=DEFAULT_DB_PATH):
    """
    Establishes a connection to the SQLite database.
    """
    # Ensure database directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    # Enable dict-like row access
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path=DEFAULT_DB_PATH):
    """
    Initializes the database schema by creating the required tables.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # Create students table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        StudentID TEXT PRIMARY KEY,
        StudyHoursPerDay REAL,
        AttendancePercentage REAL,
        SleepHours REAL,
        SocialMediaHours REAL,
        PreviousExamScore REAL,
        ParticipationInActivities INTEGER,
        InternetUsageHours REAL,
        FinalExamScore REAL
    )
    ''')
    
    # Create predictions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        study_hours REAL,
        attendance REAL,
        sleep_hours REAL,
        social_media_hours REAL,
        previous_score REAL,
        participation INTEGER,
        internet_hours REAL,
        predicted_score REAL,
        performance_category TEXT,
        timestamp TEXT
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized successfully at: {db_path}")

def save_students_to_db(df, db_path=DEFAULT_DB_PATH):
    """
    Saves the cleaned student DataFrame into the 'students' table.
    Overwrites the table if it already exists to stay synchronized.
    """
    conn = get_db_connection(db_path)
    # Write dataframe to sql
    df.to_sql('students', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    print(f"Successfully loaded {len(df)} student records into 'students' table.")

def save_prediction(study_hours, attendance, sleep_hours, social_media_hours, 
                    previous_score, participation, internet_hours, 
                    predicted_score, performance_category, db_path=DEFAULT_DB_PATH):
    """
    Saves a new user prediction into the 'predictions' table.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    cursor.execute('''
    INSERT INTO predictions (
        study_hours, attendance, sleep_hours, social_media_hours, 
        previous_score, participation, internet_hours, 
        predicted_score, performance_category, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        float(study_hours), float(attendance), float(sleep_hours), float(social_media_hours),
        float(previous_score), int(participation), float(internet_hours),
        float(predicted_score), str(performance_category), timestamp
    ))
    
    conn.commit()
    conn.close()
    print(f"Prediction saved: Predicted Score = {predicted_score:.2f} ({performance_category})")

def get_prediction_history(db_path=DEFAULT_DB_PATH):
    """
    Fetches the prediction history from the database as a DataFrame.
    """
    conn = get_db_connection(db_path)
    query = "SELECT * FROM predictions ORDER BY id DESC"
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error fetching prediction history: {e}")
        df = pd.DataFrame()
    conn.close()
    return df

def get_student_data(db_path=DEFAULT_DB_PATH):
    """
    Fetches all student records from the database as a DataFrame.
    """
    conn = get_db_connection(db_path)
    query = "SELECT * FROM students"
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error fetching student records: {e}")
        df = pd.DataFrame()
    conn.close()
    return df

def get_dashboard_metrics(db_path=DEFAULT_DB_PATH):
    """
    Calculates summary metrics (Total, Average, Max, Min score) from the student database.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    metrics = {
        'total_students': 0,
        'average_score': 0.0,
        'highest_score': 0.0,
        'lowest_score': 0.0
    }
    
    try:
        cursor.execute("SELECT COUNT(*), AVG(FinalExamScore), MAX(FinalExamScore), MIN(FinalExamScore) FROM students")
        row = cursor.fetchone()
        if row and row[0] > 0:
            metrics['total_students'] = int(row[0])
            metrics['average_score'] = float(round(row[1], 1)) if row[1] is not None else 0.0
            metrics['highest_score'] = float(row[2]) if row[2] is not None else 0.0
            metrics['lowest_score'] = float(row[3]) if row[3] is not None else 0.0
    except Exception as e:
        print(f"Error computing dashboard metrics: {e}")
        
    conn.close()
    return metrics

if __name__ == '__main__':
    # Simple test run
    init_db()
    print("Dashboard metrics:", get_dashboard_metrics())
