import os
import json

def create_jupyter_notebook(output_path=os.path.join('notebooks', 'analysis.ipynb')):
    """
    Programmatically generates a structured, well-documented Jupyter notebook
    covering the entire Student Performance Analysis & Prediction pipeline.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define the cells
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Student Performance Analysis & Prediction System\n",
                "### End-to-End Data Science and Machine Learning Pipeline\n",
                "\n",
                "This notebook guides you through the full data science workflow for analyzing and predicting student performance based on academic habits and lifestyle factors. It covers:\n",
                "1. **Synthetic Data Generation**: Creating a dataset of 1200 students with realistic correlations and anomalies.\n",
                "2. **Data Preprocessing**: Handling duplicates, missing values, and validating bounds.\n",
                "3. **Exploratory Data Analysis (EDA)**: Visualizing study hours, attendance, sleep, social media, and participation impact.\n",
                "4. **Machine Learning**: Training and optimizing a RandomForestRegressor to predict final exam scores.\n",
                "5. **Evaluation**: Assessing the model with metrics like MAE, RMSE, and R².\n",
                "6. **Feature Importance**: Understanding which habits influence academic outcomes most."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "--- \n",
                "## Step 1: Environment Setup and Library Imports"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Add src directory to system path to import modules\n",
                "sys.path.append(os.path.abspath(os.path.join('..')))\n",
                "from src.data_generator import generate_student_data\n",
                "from src.preprocess import clean_data\n",
                "from src.eda import generate_insights_summary, set_premium_plot_style\n",
                "\n",
                "print(\"Libraries successfully imported!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "--- \n",
                "## Step 2: Data Generation and Initial Inspection\n",
                "We generate the raw synthetic dataset containing 1200 records. Note that our generator intentionally injects duplicates and missing values to simulate a real-world scenario."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate raw dataset\n",
                "raw_df = generate_student_data(num_records=1200, seed=42)\n",
                "print(f\"Raw dataset shape: {raw_df.shape}\")\n",
                "\n",
                "# Check missing values\n",
                "print(\"\\nMissing values per feature:\")\n",
                "print(raw_df.isnull().sum())\n",
                "\n",
                "# Check duplicates\n",
                "print(f\"\\nDuplicate rows count: {raw_df.duplicated().sum()}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "--- \n",
                "## Step 3: Data Preprocessing\n",
                "We clean the raw dataset by:\n",
                "- Dropping duplicate rows.\n",
                "- Imputing missing values in `SleepHours` and `SocialMediaHours` with their median.\n",
                "- Standardizing data types.\n",
                "- Checking boundaries (capping outliers)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cleaned_df, preprocess_summary = clean_data(raw_df)\n",
                "print(f\"Cleaned dataset shape: {cleaned_df.shape}\")\n",
                "print(\"\\nPreprocessing Summary details:\")\n",
                "for k, v in preprocess_summary.items():\n",
                "    print(f\"{k}: {v}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "--- \n",
                "## Step 4: Exploratory Data Analysis (EDA)\n",
                "Let's visualize the academic and lifestyle factors to see how they impact final exam scores."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Setup plot aesthetics\n",
                "set_premium_plot_style()\n",
                "palette = ['#8BE9FD', '#50FA7B', '#FF79C6', '#BD93F9']\n",
                "\n",
                "# 1. Study Hours vs Final Exam Score\n",
                "plt.figure(figsize=(8, 4.5))\n",
                "sns.regplot(data=cleaned_df, x='StudyHoursPerDay', y='FinalExamScore', \n",
                "            scatter_kws={'alpha': 0.5, 'color': palette[0]}, line_kws={'color': '#FF5555'})\n",
                "plt.title('Study Hours Per Day vs. Final Exam Score')\n",
                "plt.xlabel('Study Hours Per Day')\n",
                "plt.ylabel('Final Exam Score')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Correlation Heatmap\n",
                "plt.figure(figsize=(8, 6))\n",
                "corr = cleaned_df.select_dtypes(include=[np.number]).corr()\n",
                "sns.heatmap(corr, annot=True, fmt=\".2f\", cmap='coolwarm', vmin=-1, vmax=1, square=True)\n",
                "plt.title('Correlation Matrix of Student Attributes')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Print automated statistical narrative\n",
                "_, text_report = generate_insights_summary(cleaned_df)\n",
                "print(text_report)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "--- \n",
                "## Step 5: Model Training and Evaluation\n",
                "We divide the dataset into 80% train and 20% test sets, then train a Random Forest Regressor and optimize its depth."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from sklearn.model_selection import train_test_split, GridSearchCV\n",
                "from sklearn.ensemble import RandomForestRegressor\n",
                "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
                "\n",
                "# Setup features and target\n",
                "features = ['StudyHoursPerDay', 'AttendancePercentage', 'SleepHours', \n",
                "            'SocialMediaHours', 'PreviousExamScore', 'ParticipationInActivities', \n",
                "            'InternetUsageHours']\n",
                "target = 'FinalExamScore'\n",
                "\n",
                "X = cleaned_df[features]\n",
                "y = cleaned_df[target]\n",
                "\n",
                "# Train-Test Split\n",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
                "\n",
                "# Grid Search for max_depth\n",
                "param_grid = {'max_depth': [5, 8, 10, 12, 15, None]}\n",
                "rf = RandomForestRegressor(n_estimators=200, random_state=42)\n",
                "grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)\n",
                "grid_search.fit(X_train, y_train)\n",
                "\n",
                "best_model = grid_search.best_estimator_\n",
                "print(f\"Best max_depth hyperparameter: {grid_search.best_params_['max_depth']}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Predictions and Metrics\n",
                "y_pred = best_model.predict(X_test)\n",
                "mae = mean_absolute_error(y_test, y_pred)\n",
                "rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n",
                "r2 = r2_score(y_test, y_pred)\n",
                "\n",
                "print(f\"Model Performance on test partition:\")\n",
                "print(f\"- MAE:  {mae:.4f}\")\n",
                "print(f\"- RMSE: {rmse:.4f}\")\n",
                "print(f\"- R² Score: {r2:.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "--- \n",
                "## Step 6: Feature Importances\n",
                "Let's look at which attributes hold the highest weight in the model's decision-making."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "importances = best_model.feature_importances_\n",
                "feat_importance_df = pd.DataFrame({\n",
                "    'Feature': features,\n",
                "    'Importance': importances\n",
                "}).sort_values(by='Importance', ascending=False)\n",
                "\n",
                "# Plot Feature Importance\n",
                "plt.figure(figsize=(8, 4))\n",
                "sns.barplot(data=feat_importance_df, x='Importance', y='Feature', palette='viridis')\n",
                "plt.title('Feature Importances')\n",
                "plt.xlabel('Gini Importance')\n",
                "plt.ylabel('Feature')\n",
                "plt.show()"
            ]
        }
    ]
    
    # Define metadata
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
        
    print(f"Jupyter Notebook successfully created at: {output_path}")

if __name__ == '__main__':
    create_jupyter_notebook()
