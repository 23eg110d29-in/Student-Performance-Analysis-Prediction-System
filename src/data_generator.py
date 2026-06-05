import os
import numpy as np
import pandas as pd

def generate_student_data(num_records=1200, seed=42):
    """
    Generates a synthetic dataset of student performance with realistic correlations,
    incorporating anomalies (missing values and duplicates) to test preprocessing.
    """
    np.random.seed(seed)
    
    # 1. Generate features
    student_ids = [f"STU{i:04d}" for i in range(1, num_records + 1)]
    
    # Study hours: uniform between 1.0 and 10.0 hours
    study_hours = np.random.uniform(1.0, 10.0, num_records)
    
    # Attendance percentage: normal centered at 85%, min 50%, max 100%
    attendance = np.random.normal(85.0, 10.0, num_records)
    attendance = np.clip(attendance, 50.0, 100.0)
    
    # Sleep hours: normal centered at 7.0 hours, min 4.0, max 10.0
    sleep_hours = np.random.normal(7.0, 1.2, num_records)
    sleep_hours = np.clip(sleep_hours, 4.0, 10.0)
    
    # Social media hours: uniform between 0.0 and 8.0 hours
    social_media = np.random.uniform(0.0, 8.0, num_records)
    
    # Previous exam score: normal centered at 70%, min 40%, max 100%
    prev_score = np.random.normal(70.0, 12.0, num_records)
    prev_score = np.clip(prev_score, 40.0, 100.0)
    
    # Participation in activities: binary with 40% probability of participation
    participation = np.random.binomial(1, 0.40, num_records)
    
    # Internet usage: uniform between 0.0 and 8.0 hours
    internet_usage = np.random.uniform(0.0, 8.0, num_records)
    
    # 2. Calculate final exam score with positive/negative impacts and noise
    # Base score
    final_score = 15.0
    
    # Positive impacts
    final_score += 2.5 * study_hours
    final_score += 0.35 * attendance
    final_score += 1.2 * sleep_hours
    final_score += 0.38 * prev_score
    final_score += 3.0 * participation
    
    # Negative impacts (excessive usage penalties)
    social_media_penalty = np.where(social_media > 3.0, -2.0 * (social_media - 3.0), 0)
    internet_penalty = np.where(internet_usage > 4.0, -1.5 * (internet_usage - 4.0), 0)
    
    final_score += social_media_penalty
    final_score += internet_penalty
    
    # Add random normal noise (std = 3.5)
    noise = np.random.normal(0, 3.5, num_records)
    final_score += noise
    
    # Clip final scores to 0-100 scale and round
    final_score = np.clip(final_score, 0.0, 100.0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'StudentID': student_ids,
        'StudyHoursPerDay': np.round(study_hours, 1),
        'AttendancePercentage': np.round(attendance, 1),
        'SleepHours': np.round(sleep_hours, 1),
        'SocialMediaHours': np.round(social_media, 1),
        'PreviousExamScore': np.round(prev_score, 1),
        'ParticipationInActivities': participation,
        'InternetUsageHours': np.round(internet_usage, 1),
        'FinalExamScore': np.round(final_score, 1)
    })
    
    # 3. Inject Anomalies (to test preprocessing)
    # Missing values: ~2% in SleepHours and SocialMediaHours
    missing_sleep_idx = np.random.choice(num_records, int(num_records * 0.02), replace=False)
    missing_social_idx = np.random.choice(num_records, int(num_records * 0.02), replace=False)
    
    df.loc[missing_sleep_idx, 'SleepHours'] = np.nan
    df.loc[missing_social_idx, 'SocialMediaHours'] = np.nan
    
    # Duplicates: select 5 random rows and append them
    dup_idx = np.random.choice(num_records, 5, replace=False)
    duplicates = df.iloc[dup_idx].copy()
    
    # Append duplicates and shuffle
    df = pd.concat([df, duplicates], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return df

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate dataset
    print("Generating student dataset...")
    df = generate_student_data(num_records=1200)
    
    # Save to CSV
    output_path = os.path.join('data', 'students.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Dataset successfully saved to: {output_path}")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows of generated data:")
    print(df.head())
    print(f"\nMissing values check:\n{df.isnull().sum()}")
    print(f"\nDuplicate rows check: {df.duplicated().sum()}")
