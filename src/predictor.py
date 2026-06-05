import os
import joblib
import pandas as pd
try:
    from src.database import save_prediction
except ModuleNotFoundError:
    from database import save_prediction


MODEL_PATH = os.path.join('model', 'model.pkl')

def get_performance_category(score):
    """
    Categorizes the score into a standard performance range:
      90 - 100: Excellent
      75 - 89: Good
      50 - 74: Average
      0 - 49: Needs Improvement
    """
    if score >= 90.0:
        return 'Excellent'
    elif score >= 75.0:
        return 'Good'
    elif score >= 50.0:
        return 'Average'
    else:
        return 'Needs Improvement'

def load_predictor_model():
    """
    Loads the serialized RandomForestRegressor model.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run train_model.py first.")
    return joblib.load(MODEL_PATH)

def validate_inputs(study_hours, attendance, sleep_hours, social_media_hours, 
                    previous_score, participation, internet_hours):
    """
    Validates input variables for boundaries, logical correctness, and types.
    Raises ValueError with user-friendly explanations on failure.
    """
    # 1. Type and Emptiness Validation
    inputs = {
        'Study Hours': study_hours,
        'Attendance Percentage': attendance,
        'Sleep Hours': sleep_hours,
        'Social Media Hours': social_media_hours,
        'Previous Exam Score': previous_score,
        'Participation in Activities': participation,
        'Internet Usage Hours': internet_hours
    }
    
    for name, val in inputs.items():
        if val is None or str(val).strip() == "":
            raise ValueError(f"Input '{name}' cannot be empty.")
            
        try:
            # Force conversion
            float_val = float(val)
        except ValueError:
            raise ValueError(f"Input '{name}' must be a numeric value.")
            
        # Specific ranges check
        if name in ['Study Hours', 'Sleep Hours', 'Social Media Hours', 'Internet Usage Hours']:
            if float_val < 0.0 or float_val > 24.0:
                raise ValueError(f"'{name}' must be a logical number of hours between 0.0 and 24.0 per day.")
                
        elif name in ['Attendance Percentage', 'Previous Exam Score']:
            if float_val < 0.0 or float_val > 100.0:
                raise ValueError(f"'{name}' must be a percentage value between 0.0% and 100.0%.")
                
        elif name == 'Participation in Activities':
            if int(float_val) not in [0, 1]:
                raise ValueError(f"'{name}' must be 0 (No) or 1 (Yes).")

def predict_student_performance(study_hours, attendance, sleep_hours, social_media_hours, 
                               previous_score, participation, internet_hours, db_path=None):
    """
    Validates inputs, predicts final exam score, categorizes performance,
    logs results to SQLite database, and returns prediction details.
    """
    # Validate inputs
    validate_inputs(study_hours, attendance, sleep_hours, social_media_hours, 
                    previous_score, participation, internet_hours)
    
    # Load model
    model = load_predictor_model()
    
    # Prepare input feature vector (matching column names during training)
    features = pd.DataFrame([{
        'StudyHoursPerDay': float(study_hours),
        'AttendancePercentage': float(attendance),
        'SleepHours': float(sleep_hours),
        'SocialMediaHours': float(social_media_hours),
        'PreviousExamScore': float(previous_score),
        'ParticipationInActivities': int(participation),
        'InternetUsageHours': float(internet_hours)
    }])
    
    # Predict
    predicted_score = float(model.predict(features)[0])
    # Clip just in case
    predicted_score = max(0.0, min(100.0, predicted_score))
    predicted_score = round(predicted_score, 2)
    
    # Categorize
    category = get_performance_category(predicted_score)
    
    # Save prediction to DB
    if db_path:
        save_prediction(
            study_hours, attendance, sleep_hours, social_media_hours, 
            previous_score, participation, internet_hours, 
            predicted_score, category, db_path=db_path
        )
    else:
        save_prediction(
            study_hours, attendance, sleep_hours, social_media_hours, 
            previous_score, participation, internet_hours, 
            predicted_score, category
        )
        
    return {
        'predicted_score': predicted_score,
        'category': category
    }

if __name__ == '__main__':
    # Test validation & prediction (assumes model/model.pkl exists)
    try:
        res = predict_student_performance(
            study_hours=6.5, attendance=92.0, sleep_hours=7.5, 
            social_media_hours=2.0, previous_score=85.0, 
            participation=1, internet_hours=3.5
        )
        print("Test prediction result:", res)
    except Exception as e:
        print("Predictor Test Output (Expected if model doesn't exist yet):", e)
