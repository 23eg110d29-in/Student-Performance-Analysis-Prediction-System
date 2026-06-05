import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def set_premium_plot_style():
    sns.set_theme(style="darkgrid")
    plt.rcParams.update({
        'figure.facecolor': '#1E1E24',
        'axes.facecolor': '#282A36',
        'text.color': '#F8F8F2',
        'axes.labelcolor': '#F8F8F2',
        'xtick.color': '#F8F8F2',
        'ytick.color': '#F8F8F2',
        'axes.edgecolor': '#44475A',
        'grid.color': '#44475A',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })

def train_and_evaluate(csv_path=os.path.join('data', 'students.csv'), model_dir='model', plot_dir='screenshots'):
    """
    Performs preprocessing, train-test splitting, hyperparameter optimization,
    model training, evaluation, feature importance generation, and model export.
    """
    try:
        from src.preprocess import load_and_preprocess
    except ModuleNotFoundError:
        from preprocess import load_and_preprocess
    
    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Load and Preprocess Data
    df, _ = load_and_preprocess(csv_path)
    
    # Save cleaned dataset to SQLite Database
    try:
        from src.database import init_db, save_students_to_db
    except ModuleNotFoundError:
        from database import init_db, save_students_to_db
    init_db()
    save_students_to_db(df)
    
    # Define features and target
    feature_cols = [
        'StudyHoursPerDay', 'AttendancePercentage', 'SleepHours', 
        'SocialMediaHours', 'PreviousExamScore', 'ParticipationInActivities', 
        'InternetUsageHours'
    ]
    target_col = 'FinalExamScore'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 2. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Model Training & Hyperparameter Tuning
    # Grid search for max_depth optimization
    param_grid = {
        'max_depth': [5, 8, 10, 12, 15, None]
    }
    
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    
    print("Optimizing RandomForestRegressor hyperparameter (max_depth)...")
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Optimal max_depth found: {best_params['max_depth']}")
    
    # 4. Model Evaluation
    y_pred = best_rf.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'MAE': float(np.round(mae, 4)),
        'RMSE': float(np.round(rmse, 4)),
        'R2': float(np.round(r2, 4)),
        'best_max_depth': best_params['max_depth'],
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    print("\n--- Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print("------------------------\n")
    
    # Save metrics JSON
    metrics_path = os.path.join(model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # 5. Save Model
    model_path = os.path.join(model_dir, 'model.pkl')
    joblib.dump(best_rf, model_path)
    print(f"Trained model saved to: {model_path}")
    
    # 6. Feature Importance Generation
    importances = best_rf.feature_importances_
    feat_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    most_influential = feat_importance_df.iloc[0]['Feature']
    least_influential = feat_importance_df.iloc[-1]['Feature']
    
    print("\n--- Feature Importance Ranking ---")
    for idx, row in feat_importance_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")
    print(f"\nMost Influential Feature: {most_influential}")
    print(f"Least Influential Feature: {least_influential}")
    print("----------------------------------\n")
    
    # Update metrics with importance metadata
    metrics['most_influential_feature'] = most_influential
    metrics['least_influential_feature'] = least_influential
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot Feature Importance
    set_premium_plot_style()
    plt.figure(figsize=(9, 5))
    
    # Color palette
    colors = ['#BD93F9' if (x == most_influential) else '#44475A' for x in feat_importance_df['Feature']]
    # Give the second most important a slightly lighter color and others gray
    for idx, row in feat_importance_df.reset_index().iterrows():
        if idx == 0:
            colors[idx] = '#50FA7B' # green for the most important
        elif idx == 1:
            colors[idx] = '#8BE9FD' # cyan for second most important
        elif idx == len(feat_importance_df) - 1:
            colors[idx] = '#FF5555' # red for the least important
            
    sns.barplot(
        data=feat_importance_df, x='Importance', y='Feature',
        palette=colors
    )
    plt.title('Feature Importances in Predicting Student Performance')
    plt.xlabel('Gini Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'feature_importance.png'), dpi=300, facecolor='#1E1E24')
    plt.close()
    print(f"Feature importance graph saved to: {os.path.join(plot_dir, 'feature_importance.png')}")
    
    return best_rf, metrics

if __name__ == '__main__':
    train_and_evaluate()
