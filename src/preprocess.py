import pandas as pd
import numpy as np

def clean_data(df):
    """
    Cleans the input DataFrame by:
    1. Removing duplicates.
    2. Imputing missing values in specific columns using their medians.
    3. Validating boundaries (outliers/logical errors) and correcting them.
    4. Casting columns to proper data types.
    
    Returns:
        cleaned_df (pd.DataFrame): The preprocessed DataFrame.
        summary (dict): A summary of the preprocessing steps.
    """
    summary = {}
    cleaned_df = df.copy()
    
    # Record initial state
    summary['initial_shape'] = df.shape
    summary['initial_missing_values'] = df.isnull().sum().to_dict()
    
    # 1. Remove duplicate rows
    duplicate_count = cleaned_df.duplicated().sum()
    cleaned_df.drop_duplicates(inplace=True)
    summary['duplicates_removed'] = int(duplicate_count)
    
    # 2. Impute missing values
    # In our generator, SleepHours and SocialMediaHours have ~2% missing values
    missing_cols = ['SleepHours', 'SocialMediaHours']
    imputation_details = {}
    
    for col in missing_cols:
        if col in cleaned_df.columns:
            missing_count = cleaned_df[col].isnull().sum()
            if missing_count > 0:
                median_val = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(median_val)
                imputation_details[col] = {
                    'imputed_count': int(missing_count),
                    'imputed_value': float(np.round(median_val, 1))
                }
    summary['imputations'] = imputation_details
    
    # 3. Data type correction & validation
    # Cast StudentID to string, Participation to int, and others to float
    type_conversions = {
        'StudentID': str,
        'ParticipationInActivities': int,
        'StudyHoursPerDay': float,
        'AttendancePercentage': float,
        'SleepHours': float,
        'SocialMediaHours': float,
        'PreviousExamScore': float,
        'InternetUsageHours': float,
        'FinalExamScore': float
    }
    
    for col, dtype in type_conversions.items():
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype(dtype)
            
    # 4. Outlier checking and logical validation (Capping values)
    # Check boundaries and log corrections
    corrections = {}
    
    # StudyHoursPerDay: should be [0, 24]
    study_hours_out = (cleaned_df['StudyHoursPerDay'] < 0) | (cleaned_df['StudyHoursPerDay'] > 24)
    if study_hours_out.any():
        corrections['StudyHoursPerDay'] = int(study_hours_out.sum())
        cleaned_df['StudyHoursPerDay'] = np.clip(cleaned_df['StudyHoursPerDay'], 0, 24)
        
    # AttendancePercentage: should be [0, 100]
    attendance_out = (cleaned_df['AttendancePercentage'] < 0) | (cleaned_df['AttendancePercentage'] > 100)
    if attendance_out.any():
        corrections['AttendancePercentage'] = int(attendance_out.sum())
        cleaned_df['AttendancePercentage'] = np.clip(cleaned_df['AttendancePercentage'], 0, 100)
        
    # PreviousExamScore & FinalExamScore: should be [0, 100]
    for score_col in ['PreviousExamScore', 'FinalExamScore']:
        score_out = (cleaned_df[score_col] < 0) | (cleaned_df[score_col] > 100)
        if score_out.any():
            corrections[score_col] = int(score_out.sum())
            cleaned_df[score_col] = np.clip(cleaned_df[score_col], 0, 100)
            
    # SleepHours, SocialMediaHours, InternetUsageHours: should be [0, 24] and non-negative
    for hours_col in ['SleepHours', 'SocialMediaHours', 'InternetUsageHours']:
        hours_out = (cleaned_df[hours_col] < 0) | (cleaned_df[hours_col] > 24)
        if hours_out.any():
            corrections[hours_col] = int(hours_out.sum())
            cleaned_df[hours_col] = np.clip(cleaned_df[hours_col], 0, 24)
            
    summary['outlier_corrections'] = corrections
    summary['final_shape'] = cleaned_df.shape
    
    return cleaned_df, summary

def load_and_preprocess(file_path):
    """
    Loads raw CSV data and runs it through the preprocessing pipeline.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    cleaned_df, summary = clean_data(df)
    
    print("\n--- Preprocessing Summary ---")
    print(f"Initial Dataset Shape: {summary['initial_shape']}")
    print(f"Duplicates Removed: {summary['duplicates_removed']}")
    print("Missing Values Handled:")
    for col, details in summary['imputations'].items():
        print(f"  - Column '{col}': Imputed {details['imputed_count']} missing values with median ({details['imputed_value']})")
    if summary['outlier_corrections']:
        print("Outliers / Boundaries Corrected:")
        for col, count in summary['outlier_corrections'].items():
            print(f"  - Column '{col}': Corrected {count} out-of-bound values")
    else:
        print("Outliers / Boundaries Check: No out-of-bound values detected.")
    print(f"Final Cleaned Dataset Shape: {summary['final_shape']}")
    print("-----------------------------\n")
    
    return cleaned_df, summary

if __name__ == '__main__':
    import os
    csv_path = os.path.join('data', 'students.csv')
    if os.path.exists(csv_path):
        load_and_preprocess(csv_path)
    else:
        print(f"Error: {csv_path} does not exist. Run data_generator.py first.")
