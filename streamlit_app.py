import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Adjust system path to import src modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.database import get_dashboard_metrics, get_student_data, get_prediction_history
from src.predictor import predict_student_performance, get_performance_category
from src.utils import get_premium_css, get_category_color, load_model_metrics
from src.eda import generate_insights_summary

# 1. Page Configuration
st.set_page_config(
    page_title="Student Performance Analysis & Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject CSS styles
st.markdown(get_premium_css(), unsafe_allow_html=True)

# 2. Database & Data Checks
DB_PATH = os.path.join('database', 'student_performance.db')
METRICS_PATH = os.path.join('model', 'metrics.json')

# Helper: Check if system data exists
data_generated = os.path.exists(os.path.join('data', 'students.csv'))
model_trained = os.path.exists(os.path.join('model', 'model.pkl'))

# Initialize session state for sidebar database status
if 'db_status' not in st.st_vars if hasattr(st, 'st_vars') else st.session_state:
    st.session_state.db_initialized = os.path.exists(DB_PATH)

# 3. Sidebar Configuration
with st.sidebar:
    st.markdown("<h3 style='text-align: center; font-size: 20px; font-weight: 700; margin-bottom: 0px;'>🎓 Student Performance Analysis & Prediction System</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6272A4; font-size: 14px;'>Academic Performance Insights</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Navigation Radio
    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "📈 Data Analysis", "🤖 Model Performance", "🔮 Predict Score", "📜 Prediction History"]
    )
    
    st.markdown("---")
    # System Status widget
    st.markdown("#### ⚙️ System Status")
    
    if data_generated:
        st.success("✅ Dataset: Generated")
    else:
        st.warning("⚠️ Dataset: Missing")
        
    if model_trained:
        st.success("✅ Model: Trained & Ready")
    else:
        st.warning("⚠️ Model: Not Trained")
        
    if os.path.exists(DB_PATH):
        st.success("✅ Database: Connected")
    else:
        st.error("❌ Database: Offline")
        
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 11px; color: #6272A4;'>© 2026 Student Analytics System</p>", unsafe_allow_html=True)

# 4. Page Routing & Rendering
if page == "📊 Dashboard":
    st.title("📊 Student Performance Analysis & Prediction System")
    st.markdown("Analyze academic and lifestyle factors affecting student performance and predict final exam scores using Machine Learning.")
    st.markdown("---")
    
    if not os.path.exists(DB_PATH) or not data_generated:
        st.info("👋 Welcome! It looks like the data has not been generated or loaded yet. Run the data generator to initialize the system.")
        if st.button("🚀 Run Setup (Generate Data & Train Model)"):
            with st.spinner("Setting up system..."):
                try:
                    # Run commands via importing
                    from src.data_generator import generate_student_data
                    from src.preprocess import load_and_preprocess
                    from src.database import init_db, save_students_to_db
                    from src.eda import generate_eda_visualizations
                    from src.train_model import train_and_evaluate
                    from src.generate_notebook.py import create_jupyter_notebook
                    
                    df = generate_student_data()
                    df.to_csv(os.path.join('data', 'students.csv'), index=False)
                    cleaned_df, _ = load_and_preprocess(os.path.join('data', 'students.csv'))
                    init_db()
                    save_students_to_db(cleaned_df)
                    generate_eda_visualizations(cleaned_df)
                    train_and_evaluate()
                    
                    st.success("Setup complete! Refreshing...")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during setup: {e}")
    else:
        # Load metrics
        metrics = get_dashboard_metrics()
        
        # Display KPI cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">👥 Total Students</div>
                <div class="metric-value">{metrics['total_students']:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">📈 Average Score</div>
                <div class="metric-value">{metrics['average_score']}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">🏆 Highest Score</div>
                <div class="metric-value">{metrics['highest_score']}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">📉 Lowest Score</div>
                <div class="metric-value">{metrics['lowest_score']}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("### 🔍 System Overview")
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown("""
            This **Student Performance Analysis & Prediction System** provides institutional administrators and educators 
            with data-driven insights to predict final exam scores. 
            
            By tracking indicators such as daily study hours, sleep habits, participation in extracurriculars, 
            attendance rates, and digital usage, the system helps target early interventions for at-risk students.
            
            #### Core Capabilities:
            1. **Habit Tracking**: Examine correlations between lifestyle behaviors and academic scores.
            2. **Predictive Modeling**: Calculate student outcomes in real-time using a Random Forest Regressor.
            3. **Decision Support**: Flag students who need improvements and offer custom, automated advice.
            4. **Prediction Logging**: Build a historical ledger of predictions for longitudinal research.
            """)
            
        with col_right:
            # Let's show a summary metrics bar
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("#### 🎓 Performance Categories")
            df = get_student_data()
            if not df.empty:
                categories = df['FinalExamScore'].apply(get_performance_category)
                cat_counts = categories.value_counts()
                for cat in ['Excellent', 'Good', 'Average', 'Needs Improvement']:
                    count = cat_counts.get(cat, 0)
                    pct = (count / len(df)) * 100
                    color = get_category_color(cat)['hex']
                    st.markdown(f"**{cat}** ({pct:.1f}%):")
                    st.progress(int(pct), text=f"{count} students")
            st.markdown("</div>", unsafe_allow_html=True)

elif page == "📈 Data Analysis":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("Explore distribution profiles, correlation metrics, and habits affecting grades.")
    st.markdown("---")
    
    df = get_student_data()
    
    if df.empty:
        st.warning("⚠️ No student records found. Initialize the database first.")
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Visual Analytics", 
            "📋 Dataset Preview", 
            "📈 Summary Statistics", 
            "🛡️ Data Quality Report", 
            "💡 Summary Observations"
        ])
        
        with tab1:
            st.markdown("### Core Visualizations")
            col1, col2 = st.columns(2)
            
            # Study Hours vs Final Score (Scatter)
            with col1:
                st.markdown("#### Study Hours vs Final Score")
                if os.path.exists("screenshots/study_vs_final.png"):
                    st.image("screenshots/study_vs_final.png", use_container_width=True)
                else:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.regplot(data=df, x='StudyHoursPerDay', y='FinalExamScore', 
                                scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax)
                    st.pyplot(fig)
            
            # Correlation Matrix (Heatmap)
            with col2:
                st.markdown("#### Feature Correlation Heatmap")
                if os.path.exists("screenshots/correlation_heatmap.png"):
                    st.image("screenshots/correlation_heatmap.png", use_container_width=True)
                else:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                    st.pyplot(fig)
                    
            st.markdown("---")
            col3, col4 = st.columns(2)
            
            # Extracurriculars
            with col3:
                st.markdown("#### Participation in Extracurriculars")
                if os.path.exists("screenshots/participation_impact.png"):
                    st.image("screenshots/participation_impact.png", use_container_width=True)
                else:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.barplot(data=df, x='ParticipationInActivities', y='FinalExamScore', ax=ax)
                    st.pyplot(fig)
            
            # Previous Exam vs Final Exam
            with col4:
                st.markdown("#### Academic Consistency (Previous vs Final)")
                if os.path.exists("screenshots/prev_vs_final.png"):
                    st.image("screenshots/prev_vs_final.png", use_container_width=True)
                else:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.regplot(data=df, x='PreviousExamScore', y='FinalExamScore', 
                                scatter_kws={'alpha':0.5}, line_kws={'color':'green'}, ax=ax)
                    st.pyplot(fig)
                    
            st.markdown("---")
            st.markdown("### Lifestyle Metric Distributions")
            col5, col6, col7 = st.columns(3)
            
            with col5:
                st.markdown("#### Attendance Rate")
                if os.path.exists("screenshots/attendance_dist.png"):
                    st.image("screenshots/attendance_dist.png", use_container_width=True)
                else:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.histplot(df['AttendancePercentage'], kde=True, ax=ax)
                    st.pyplot(fig)
                    
            with col6:
                st.markdown("#### Sleep Duration")
                if os.path.exists("screenshots/sleep_dist.png"):
                    st.image("screenshots/sleep_dist.png", use_container_width=True)
                else:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.histplot(df['SleepHours'], kde=True, ax=ax)
                    st.pyplot(fig)
                    
            with col7:
                st.markdown("#### Social Media Consumption")
                if os.path.exists("screenshots/social_media_dist.png"):
                    st.image("screenshots/social_media_dist.png", use_container_width=True)
                else:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.histplot(df['SocialMediaHours'], kde=True, ax=ax)
                    st.pyplot(fig)
        
        with tab2:
            st.markdown("### 📋 Dataset Preview")
            st.markdown("Interactive preview of the first 10 student records:")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("#### 📊 Dataset Dimensions")
            shape_col1, shape_col2, shape_col3 = st.columns(3)
            with shape_col1:
                st.markdown(f"""
                <div class="metric-card" style="padding: 15px; text-align: center; margin-bottom: 0px;">
                    <div class="metric-title" style="font-size: 11px;">Total Records (Rows)</div>
                    <div style="font-size: 26px; font-weight: 700; color: #50FA7B;">{df.shape[0]:,}</div>
                </div>
                """, unsafe_allow_html=True)
            with shape_col2:
                st.markdown(f"""
                <div class="metric-card" style="padding: 15px; text-align: center; margin-bottom: 0px;">
                    <div class="metric-title" style="font-size: 11px;">Total Features (Columns)</div>
                    <div style="font-size: 26px; font-weight: 700; color: #8BE9FD;">{df.shape[1]}</div>
                </div>
                """, unsafe_allow_html=True)
            with shape_col3:
                st.markdown(f"""
                <div class="metric-card" style="padding: 15px; text-align: center; margin-bottom: 0px;">
                    <div class="metric-title" style="font-size: 11px;">Dataset Shape</div>
                    <div style="font-size: 26px; font-weight: 700; color: #FFB86C;">({df.shape[0]}, {df.shape[1]})</div>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### 📈 Descriptive Summary Statistics")
            st.dataframe(df.describe(), use_container_width=True)
            
            st.markdown("#### 📚 Statistical Terms Explained")
            st.markdown("""
            *   **Mean (Average)**: The mathematical center of the data. Represents the sum of all values divided by total records.
            *   **Standard Deviation (Std)**: Represents the spread or variation of values. High values show a wider variety of student profiles.
            *   **Minimum (Min) & Maximum (Max)**: The boundaries showing the extreme values in the data.
            *   **25% (First Quartile)**: The threshold below which 25% of student values lie.
            *   **50% (Median)**: The exact middle value. 50% of students scored higher and 50% scored lower.
            *   **75% (Third Quartile)**: The threshold below which 75% of student values lie.
            """)
            
        with tab4:
            st.markdown("### 🛡️ Data Quality & Completeness Report")
            
            missing_per_col = df.isnull().sum()
            total_missing = missing_per_col.sum()
            duplicates = df.duplicated().sum()
            completeness_pct = (df.notnull().sum().sum() / df.size) * 100
            
            q_col1, q_col2, q_col3 = st.columns(3)
            with q_col1:
                st.markdown(f"""
                <div class="metric-card" style="padding: 15px; text-align: center; margin-bottom: 0px;">
                    <div class="metric-title" style="font-size: 11px;">Total Missing Values</div>
                    <div style="font-size: 26px; font-weight: 700; color: {'#50FA7B' if total_missing == 0 else '#FF5555'};">{total_missing}</div>
                </div>
                """, unsafe_allow_html=True)
            with q_col2:
                st.markdown(f"""
                <div class="metric-card" style="padding: 15px; text-align: center; margin-bottom: 0px;">
                    <div class="metric-title" style="font-size: 11px;">Duplicate Records</div>
                    <div style="font-size: 26px; font-weight: 700; color: {'#50FA7B' if duplicates == 0 else '#FF5555'};">{duplicates}</div>
                </div>
                """, unsafe_allow_html=True)
            with q_col3:
                st.markdown(f"""
                <div class="metric-card" style="padding: 15px; text-align: center; margin-bottom: 0px;">
                    <div class="metric-title" style="font-size: 11px;">Completeness Percentage</div>
                    <div style="font-size: 26px; font-weight: 700; color: #50FA7B;">{completeness_pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            
            if total_missing == 0 and duplicates == 0:
                st.success("✅ **Dataset Quality is Excellent!** No missing values or duplicates detected in the SQL tables. Preprocessing has cleaned the dataset.")
            else:
                st.warning("⚠️ **Dataset Quality holds anomalies.** See missing column distribution below.")
                
            st.markdown("#### Data Quality Per Column")
            quality_df = pd.DataFrame({
                "Missing Values": missing_per_col,
                "Completeness %": [f"{(df[c].notnull().sum() / len(df)) * 100:.1f}%" for c in df.columns]
            })
            st.dataframe(quality_df, use_container_width=True)
            
        with tab5:
            st.markdown("### Automated Summary Observations")
            summary, text_report = generate_insights_summary(df)
            
            st.markdown(f"""
            <div class="metric-card" style="font-size: 15px; line-height: 1.6;">
                <p>💡 <b>Most Influential Positive Factor</b>: <code>{summary['strongest_positive_factor']}</code> with a correlation coefficient of <code>r = {summary['strongest_positive_correlation']:.3f}</code>.</p>
                <p>🛑 <b>Most Influential Negative Distractor</b>: <code>{summary['strongest_negative_factor']}</code> with a correlation coefficient of <code>r = {summary['strongest_negative_correlation']:.3f}</code>.</p>
                <p>🏆 <b>Extracurricular Bonus</b>: Students participating in extracurricular activities scored an average of <b>{summary['extracurricular_bonus']}%</b> higher than those who did not participate.</p>
                <p>😴 <b>Sleep Insights</b>: Students sleeping between 6 and 8 hours averaged a final exam score of <b>{summary['sleep_hours_binned_averages'].get('(6, 8]', 'N/A')}%</b>.</p>
                <p>📱 <b>Social Media Impact</b>: Heavy social media users (&gt; 5 hours/day) showed a degraded exam performance, averaging <b>{summary['social_media_binned_averages'].get('(5, 24]', 'N/A')}%</b>, compared to light users (&lt; 2 hours/day) who averaged <b>{summary['social_media_binned_averages'].get('(0, 2]', 'N/A')}%</b>.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.text_area("Full Statistical Log", value=text_report, height=220, disabled=True)
            
        # 6. Data Science Insights Panel (Requirement 6)
        st.markdown("---")
        st.markdown("### 💡 Data Science Insights Panel")
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        if 'FinalExamScore' in corr_matrix.columns:
            target_corr = corr_matrix['FinalExamScore'].drop('FinalExamScore').sort_values(ascending=False)
            strong_pos = target_corr.index[0]
            strong_pos_val = target_corr.iloc[0]
            strong_neg = target_corr.index[-1]
            strong_neg_val = target_corr.iloc[-1]
            
            st.markdown(f"""
            <div class="metric-card" style="font-size: 15px; line-height: 1.7; margin-bottom: 0px;">
                <ul style="margin: 0; padding-left: 20px;">
                    <li>🚀 <b>{strong_pos}</b> has the strongest positive correlation with Final Score (<code>r = {strong_pos_val:.3f}</code>). Dedicated revision duration is the single greatest driver of final outcomes.</li>
                    <li>📱 <b>Social Media Usage</b> (<code>{strong_neg}</code>) has a significant negative impact (<code>r = {strong_neg_val:.3f}</code>), representing the primary distractor that degrades grades.</li>
                    <li>📖 <b>Previous Exam Score</b> shows academic consistency is a strong predictor of final outcomes (<code>r = {corr_matrix.loc['PreviousExamScore', 'FinalExamScore']:.3f}</code>).</li>
                    <li>⚽ Students participating in <b>extracurricular activities</b> perform slightly better in final marks (<code>r = {corr_matrix.loc['ParticipationInActivities', 'FinalExamScore']:.3f}</code>), supporting soft skills development.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

elif page == "🤖 Model Performance":
    st.title("🤖 Predictive Model Details")
    st.markdown("Review hyperparameters, performance metrics, and Gini feature importances.")
    st.markdown("---")
    
    metrics = load_model_metrics()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-title">📉 Mean Absolute Error (MAE)</div>
            <div class="metric-value">{metrics['MAE']}</div>
            <div style="font-size: 11px; color:#6272A4; margin-top:5px;">Average absolute error in exam points</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-title">📐 Root Mean Squared Error (RMSE)</div>
            <div class="metric-value">{metrics['RMSE']}</div>
            <div style="font-size: 11px; color:#6272A4; margin-top:5px;">Standard deviation of residuals</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-title">🎯 R-Squared Coefficient (R²)</div>
            <div class="metric-value">{metrics['R2']}</div>
            <div style="font-size: 11px; color:#6272A4; margin-top:5px;">Proportion of variance explained by model</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### Hyperparameter Configuration")
    st.code(f"""
    - Model Algorithm: RandomForestRegressor
    - Number of Estimators (Trees): 200
    - Random State: 42
    - Train-Test Split Partition: 80% / 20%
    - Optimized Max Depth: {metrics['best_max_depth']}
    - Training Sample Size: {metrics['train_samples']} students
    - Testing Sample Size: {metrics['test_samples']} students
    """)
    
    st.markdown("### Feature Importance Matrix")
    col_img, col_text = st.columns([3, 2])
    
    with col_img:
        if os.path.exists("screenshots/feature_importance.png"):
            st.image("screenshots/feature_importance.png", use_container_width=True)
        else:
            st.info("Feature importance plot not generated yet. Run model training.")
            
    with col_text:
        st.markdown(f"""
        <div class="metric-card">
            <h4>💡 Key Modeling Observations</h4>
            <p><b>Top Influencing Driver</b>: The model relies heavily on <code>{metrics['most_influential_feature']}</code> to partition decision trees, representing the primary driving force in predictive outcome.</p>
            <p><b>Minor Influencing Driver</b>: The feature with the least predictive weight is <code>{metrics['least_influential_feature']}</code>.</p>
            <hr style="border-color:#44475A;"/>
            <p style="font-size: 13px; color:#6272A4;">Gini importance values represent the total reduction in the mean squared error criterion brought by that feature. Features with higher values explain more variance in the final exam grade.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "🔮 Predict Score":
    st.title("🔮 Student Performance Predictor")
    st.markdown("Input student academic habits and lifestyle behaviors to predict their final exam score.")
    st.markdown("---")
    
    if not model_trained:
        st.warning("⚠️ Model file model.pkl not detected. Please execute model training steps first.")
    else:
        st.markdown("### Enter Student Metrics")
        
        # Inputs split in columns
        col1, col2 = st.columns(2)
        
        with col1:
            study_hours = st.slider("Study Hours Per Day", min_value=0.0, max_value=12.0, value=5.0, step=0.5)
            attendance = st.slider("Attendance Percentage (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
            sleep_hours = st.slider("Sleep Hours Per Day", min_value=0.0, max_value=14.0, value=7.0, step=0.5)
            participation = st.selectbox("Participation in Extracurriculars", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
            
        with col2:
            social_media = st.slider("Social Media Usage (Hours/Day)", min_value=0.0, max_value=12.0, value=2.0, step=0.5)
            internet_hours = st.slider("Internet Surfing Hours (Hours/Day)", min_value=0.0, max_value=12.0, value=3.0, step=0.5)
            previous_score = st.number_input("Previous Exam Score (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
            
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Predict Exam Performance")
        
        if predict_btn:
            try:
                # Predict
                res = predict_student_performance(
                    study_hours=study_hours,
                    attendance=attendance,
                    sleep_hours=sleep_hours,
                    social_media_hours=social_media,
                    previous_score=previous_score,
                    participation=participation,
                    internet_hours=internet_hours,
                    db_path=DB_PATH
                )
                
                score = res['predicted_score']
                category = res['category']
                colors = get_category_color(category)
                
                st.markdown("---")
                st.markdown("### Prediction Results")
                
                # Render beautiful custom card
                st.markdown(f"""
                <div class="result-container" style="background-color: {colors['background']}4D; border: 2px solid {colors['border']};">
                    <h2 style="margin: 0; font-size: 20px; color: #FFFFFF;">Predicted Final Exam Grade</h2>
                    <div style="font-size: 64px; font-weight: 800; color: {colors['hex']}; margin: 10px 0;">
                        {score:.2f}%
                    </div>
                    <span class="result-badge" style="background-color: {colors['background']}; color: {colors['text']}; border-color: {colors['border']};">
                        Category: {category}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # Generate dynamic recommendations (Requirement 5)
                recs = []
                if attendance < 90.0:
                    recs.append("Improve attendance above 90%.")
                if study_hours < 5.0:
                    recs.append("Increase daily study time to at least 5–6 hours.")
                if social_media > 3.0:
                    recs.append("Reduce social media usage to improve concentration.")
                if sleep_hours < 6.0:
                    recs.append("Maintain 7–8 hours of sleep for better academic performance.")
                if category == 'Excellent':
                    recs.append("Keep maintaining your current habits.")
                
                if not recs:
                    recs.append("Your habits are well-balanced. Keep up the good work!")
                
                st.markdown("### 💡 Smart Recommendations")
                recs_markdown = "\n".join([f"* 🎯 **{r}**" for r in recs])
                
                if category in ['Excellent', 'Good']:
                    st.success(recs_markdown)
                else:
                    st.warning(recs_markdown)
                    
                # Feedback recommendations (Bonus Resume Feature)
                st.markdown("### 📋 Detailed Habits Breakdown")
                
                rec_cols = st.columns(3)
                
                # Academic Advice
                with rec_cols[0]:
                    st.markdown("#### 📚 Academic Habits")
                    if study_hours < 4.0:
                        st.warning("⚠️ **Low study time detected.** Advise increasing daily study hours to at least 4.0 - 5.0 hours for grade stability.")
                    else:
                        st.success("✅ **Study duration is healthy.** Encourage maintaining this consistency.")
                        
                    if attendance < 80.0:
                        st.error("🚨 **Critical attendance risk.** Low attendance strongly impacts outcomes. Strongly recommend counselor follow-up.")
                    else:
                        st.success("✅ **Attendance rate is in good standing.**")
                
                # Lifestyle Advice
                with rec_cols[1]:
                    st.markdown("#### 😴 Lifestyle Balance")
                    if sleep_hours < 6.5:
                        st.warning("⚠️ **Sleep deprivation risk.** Studies show sleep deprivation impairs memory retention. Recommend target of 7-8 hours.")
                    elif sleep_hours > 9.5:
                        st.info("ℹ️ **High sleep duration.** Ensure sleep is active rather than sluggish.")
                    else:
                        st.success("✅ **Healthy sleep hours.** Promotes strong academic retention.")
                        
                    if participation == 0:
                        st.info("ℹ️ **Extracurricular participation boosts scores.** Encourage joining a club or student sport.")
                    else:
                        st.success("✅ **Active in extracurriculars. Helps build soft skills and engagement.**")
                        
                # Digital Hygiene Advice
                with rec_cols[2]:
                    st.markdown("#### 📱 Digital Hygiene")
                    if social_media > 4.0:
                        st.error("🚨 **Excessive social media usage.** A strong negative correlation is observed with excessive use. Recommend capping to <2.5 hours.")
                    else:
                        st.success("✅ **Social media consumption is well-balanced.**")
                        
                    if internet_hours > 5.0:
                        st.warning("⚠️ **High internet surfing hours.** Ensure non-study web browsing doesn't encroach on rest or revision periods.")
                    else:
                        st.success("✅ **Internet usage hours are reasonable.**")
                        
            except ValueError as val_err:
                st.error(f"Validation Error: {val_err}")
            except Exception as e:
                st.error(f"System Error: {e}")

elif page == "📜 Prediction History":
    st.title("📜 Prediction History logs")
    st.markdown("Inspect historical records of user predictions logged in the SQLite database.")
    st.markdown("---")
    
    df_history = get_prediction_history(DB_PATH)
    
    if df_history.empty:
        st.info("ℹ️ No predictions logged in history yet. Go to 'Predict Score' to run your first simulation.")
    else:
        st.markdown("### Historical Log Registry")
        
        # Search and Filters UI (Requirement 7)
        st.markdown("### 🔍 Search & Filter History")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            search_query = st.text_input("🔍 Search Logs (by Category or Timestamp)", value="")
            
        with filter_col2:
            cat_list = ["All", "Excellent", "Good", "Average", "Needs Improvement"]
            selected_cat = st.selectbox("🎯 Filter by Category", options=cat_list)
            
        with filter_col3:
            sort_order = st.selectbox("↕️ Sort by Predicted Score", ["None", "Highest First", "Lowest First"])
            
        # Apply Filters
        filtered_df = df_history.copy()
        
        if search_query:
            search_query = search_query.lower()
            mask = (
                filtered_df['performance_category'].astype(str).str.lower().str.contains(search_query, na=False) |
                filtered_df['timestamp'].astype(str).str.lower().str.contains(search_query, na=False) |
                filtered_df['id'].astype(str).str.lower().str.contains(search_query, na=False) |
                filtered_df['predicted_score'].astype(str).str.contains(search_query, na=False) |
                filtered_df['previous_score'].astype(str).str.contains(search_query, na=False) |
                filtered_df['attendance'].astype(str).str.contains(search_query, na=False) |
                filtered_df['study_hours'].astype(str).str.contains(search_query, na=False)
            )
            filtered_df = filtered_df[mask]
            
        if selected_cat != "All":
            filtered_df = filtered_df[filtered_df['performance_category'] == selected_cat]
            
        if sort_order == "Highest First":
            filtered_df = filtered_df.sort_values(by='predicted_score', ascending=False)
        elif sort_order == "Lowest First":
            filtered_df = filtered_df.sort_values(by='predicted_score', ascending=True)
            
        display_df = filtered_df.rename(columns={
            'id': 'ID',
            'study_hours': 'Study Hours',
            'attendance': 'Attendance %',
            'sleep_hours': 'Sleep Hours',
            'social_media_hours': 'Social Media Hours',
            'previous_score': 'Prev Exam Score',
            'participation': 'Activities Part.',
            'internet_hours': 'Internet Hours',
            'predicted_score': 'Predicted Score',
            'performance_category': 'Category',
            'timestamp': 'Timestamp'
        })
        
        st.markdown(f"**Found {len(display_df)} matching prediction records**")
        
        # Download report
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download History as CSV",
            data=csv_data,
            file_name="student_prediction_history.csv",
            mime="text/csv",
            key="download-csv-btn"
        )
        
        # Show table
        st.dataframe(display_df, use_container_width=True)
        
        # Option to clear database predictions table
        if st.checkbox("⚠️ Enable Database Operations"):
            if st.button("🗑️ Clear History Logs"):
                from src.database import get_db_connection
                try:
                    conn = get_db_connection(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM predictions")
                    conn.commit()
                    conn.close()
                    st.success("History log successfully deleted. Reloading...")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error resetting database table: {e}")
  
# 5. Professional Global Footer (Requirement 8)
st.markdown("---")
footer_html = """
<div style="text-align: center; margin-top: 40px; padding: 20px; color: #6272A4; font-size: 13px;">
    <p style="margin: 0; font-weight: 600;">Developed for Data Science Internship Project</p>
    <p style="margin: 5px 0 0 0; font-size: 11px;"><b>Technology Stack:</b> Python | Streamlit | SQLite | Pandas | Seaborn | Random Forest Regressor</p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
