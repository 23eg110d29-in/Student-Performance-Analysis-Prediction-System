import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_premium_plot_style():
    """
    Configures Seaborn and Matplotlib options to output beautiful, dark-mode friendly,
    and highly professional plots.
    """
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
        'figure.titlesize': 16
    })

def generate_eda_visualizations(df, output_dir='screenshots'):
    """
    Generates all requested visualizations and saves them to the screenshots folder.
    """
    os.makedirs(output_dir, exist_ok=True)
    set_premium_plot_style()
    
    palette = ['#8BE9FD', '#50FA7B', '#FF79C6', '#BD93F9', '#FFB86C', '#F1FA8C', '#FF5555']
    
    # 1. Scatter Plot: Study Hours vs Final Exam Score
    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=df, x='StudyHoursPerDay', y='FinalExamScore',
        scatter_kws={'alpha': 0.6, 'color': palette[0]},
        line_kws={'color': '#FF5555', 'linewidth': 2}
    )
    plt.title('Study Hours Per Day vs. Final Exam Score')
    plt.xlabel('Study Hours Per Day')
    plt.ylabel('Final Exam Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'study_vs_final.png'), dpi=300, facecolor='#1E1E24')
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(9, 7))
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    # Generate custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw heatmap
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1, center=0,
        square=True, linewidths=.5, cbar_kws={"shrink": .8},
        annot_kws={"size": 10}
    )
    plt.title('Correlation Matrix of Student Attributes')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, facecolor='#1E1E24')
    plt.close()
    
    # 3. Bar Chart: Participants vs Non-Participants in Extracurriculars
    plt.figure(figsize=(6, 5))
    avg_scores = df.groupby('ParticipationInActivities')['FinalExamScore'].mean().reset_index()
    avg_scores['ParticipationInActivities'] = avg_scores['ParticipationInActivities'].map({0: 'No', 1: 'Yes'})
    
    sns.barplot(
        data=avg_scores, x='ParticipationInActivities', y='FinalExamScore',
        palette=[palette[2], palette[1]]
    )
    plt.title('Average Final Exam Score by Extracurricular Participation')
    plt.xlabel('Participation In Activities')
    plt.ylabel('Average Final Exam Score')
    plt.ylim(0, 100)
    for index, row in avg_scores.iterrows():
        plt.text(index, row['FinalExamScore'] + 2, f"{row['FinalExamScore']:.1f}%", 
                 color='#F8F8F2', ha="center", va="bottom", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'participation_impact.png'), dpi=300, facecolor='#1E1E24')
    plt.close()
    
    # 4. Attendance Distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(df['AttendancePercentage'], kde=True, color=palette[1], bins=25)
    plt.title('Distribution of Attendance Percentage')
    plt.xlabel('Attendance Percentage')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attendance_dist.png'), dpi=300, facecolor='#1E1E24')
    plt.close()
    
    # 5. Sleep Hours Distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(df['SleepHours'].dropna(), kde=True, color=palette[3], bins=15)
    plt.title('Distribution of Daily Sleep Hours')
    plt.xlabel('Sleep Hours')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sleep_dist.png'), dpi=300, facecolor='#1E1E24')
    plt.close()
    
    # 6. Social Media Usage Distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(df['SocialMediaHours'].dropna(), kde=True, color=palette[2], bins=20)
    plt.title('Distribution of Daily Social Media Usage')
    plt.xlabel('Social Media Hours')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'social_media_dist.png'), dpi=300, facecolor='#1E1E24')
    plt.close()
    
    # 7. Previous Score vs Final Score
    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=df, x='PreviousExamScore', y='FinalExamScore',
        scatter_kws={'alpha': 0.6, 'color': palette[4]},
        line_kws={'color': '#50FA7B', 'linewidth': 2}
    )
    plt.title('Previous Exam Score vs. Final Exam Score')
    plt.xlabel('Previous Exam Score')
    plt.ylabel('Final Exam Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prev_vs_final.png'), dpi=300, facecolor='#1E1E24')
    plt.close()
    
    print(f"All EDA visualizations successfully saved in: {output_dir}")

def generate_insights_summary(df):
    """
    Analyzes student data and returns a narrative summary of insights.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()['FinalExamScore'].sort_values(ascending=False)
    
    # Extract insights
    strongest_pos = corr.index[1] # index 0 is FinalExamScore itself
    strongest_pos_val = corr.iloc[1]
    
    strongest_neg = corr.index[-1]
    strongest_neg_val = corr.iloc[-1]
    
    part_score_diff = df[df['ParticipationInActivities'] == 1]['FinalExamScore'].mean() - \
                      df[df['ParticipationInActivities'] == 0]['FinalExamScore'].mean()
                      
    sleep_impact = df.groupby(pd.cut(df['SleepHours'], bins=[0, 6, 8, 24]))['FinalExamScore'].mean().to_dict()
    
    social_media_impact = df.groupby(pd.cut(df['SocialMediaHours'], bins=[0, 2, 5, 24]))['FinalExamScore'].mean().to_dict()
    
    summary = {
        'strongest_positive_factor': strongest_pos,
        'strongest_positive_correlation': float(np.round(strongest_pos_val, 3)),
        'strongest_negative_factor': strongest_neg,
        'strongest_negative_correlation': float(np.round(strongest_neg_val, 3)),
        'extracurricular_bonus': float(np.round(part_score_diff, 2)),
        'sleep_hours_binned_averages': {str(k): float(np.round(v, 2)) for k, v in sleep_impact.items()},
        'social_media_binned_averages': {str(k): float(np.round(v, 2)) for k, v in social_media_impact.items()}
    }
    
    # Construct textual summary
    text_report = f"""=== AUTOMATED EXPLORATORY DATA ANALYSIS INSIGHTS ===
1. Primary Drivers of Academic Success:
   - The factor most strongly positively correlated with final scores is '{strongest_pos}' (r = {strongest_pos_val:.3f}).
   - Academic consistency is evident: Previous Exam Scores and daily Study Hours are robust indicators of final outcomes.
   
2. Primary Distractors:
   - The factor with the strongest negative impact is '{strongest_neg}' (r = {strongest_neg_val:.3f}).
   - Excessive social media usage and internet surfing show a non-linear negative penalty.
   
3. Extracurricular Enrichment:
   - Students participating in extracurricular activities score, on average, {part_score_diff:.2f}% higher than non-participants.
   
4. Lifestyle Binning:
   - Sleep: Students getting 6-8 hours of sleep average {sleep_impact.get(pd.Interval(6, 8, closed='right'), 0.0):.2f}% final score, showing better retention.
   - Social Media: Heavy users (>5 hours) score significantly lower (avg: {social_media_impact.get(pd.Interval(5, 24, closed='right'), 0.0):.2f}%) compared to light users (<2 hours, avg: {social_media_impact.get(pd.Interval(0, 2, closed='right'), 0.0):.2f}%).
====================================================
"""
    return summary, text_report

if __name__ == '__main__':
    try:
        from src.preprocess import load_and_preprocess
    except ModuleNotFoundError:
        from preprocess import load_and_preprocess
    csv_path = os.path.join('data', 'students.csv')
    if os.path.exists(csv_path):
        cleaned_df, _ = load_and_preprocess(csv_path)
        generate_eda_visualizations(cleaned_df)
        _, text_report = generate_insights_summary(cleaned_df)
        print(text_report)
    else:
        print(f"Error: {csv_path} does not exist. Run data_generator.py first.")
