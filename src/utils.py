import os
import json

METRICS_PATH = os.path.join('model', 'metrics.json')

def load_model_metrics():
    """
    Loads model evaluation metrics from model/metrics.json.
    """
    if not os.path.exists(METRICS_PATH):
        return {
            'MAE': 'N/A',
            'RMSE': 'N/A',
            'R2': 'N/A',
            'train_samples': 'N/A',
            'test_samples': 'N/A',
            'best_max_depth': 'N/A',
            'most_influential_feature': 'N/A',
            'least_influential_feature': 'N/A'
        }
    try:
        with open(METRICS_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return {}

def get_category_color(category):
    """
    Returns UI colors for performance categories:
      Excellent -> #50FA7B (Bright Green)
      Good -> #8BE9FD (Cyan)
      Average -> #FFB86C (Orange)
      Needs Improvement -> #FF5555 (Red)
    """
    colors = {
        'Excellent': {
            'background': '#1B5E20',
            'text': '#E8F5E9',
            'border': '#4CAF50',
            'hex': '#50FA7B'
        },
        'Good': {
            'background': '#0D47A1',
            'text': '#E3F2FD',
            'border': '#2196F3',
            'hex': '#8BE9FD'
        },
        'Average': {
            'background': '#E65100',
            'text': '#FFF3E0',
            'border': '#FF9800',
            'hex': '#FFB86C'
        },
        'Needs Improvement': {
            'background': '#B71C1C',
            'text': '#FFEBEE',
            'border': '#F44336',
            'hex': '#FF5555'
        }
    }
    return colors.get(category, {
        'background': '#44475A',
        'text': '#F8F8F2',
        'border': '#6272A4',
        'hex': '#F8F8F2'
    })

def get_premium_css():
    """
    Returns a custom CSS string to inject into Streamlit for advanced UI styling.
    """
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Apply premium font globally */
    html, body, [class*="st-"], .stApp, p, span, label, input, button, select, div, h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', 'Outfit', 'Inter', sans-serif !important;
    }
    
    /* Main Background and Colors */
    .stApp {
        background-color: #0B1120 !important;
        color: #F9FAFB !important;
    }
    
    /* Header background removal */
    header, [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #0B1120 !important;
        border-right: 1px solid #374151 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #CBD5E1 !important;
    }
    [data-testid="stSidebar"] label {
        color: #F9FAFB !important;
        font-weight: 600 !important;
    }
    /* Style navigation radio options */
    [data-testid="stSidebar"] div[data-testid="stRadio"] label p {
        color: #CBD5E1 !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease;
    }
    /* Hover micro-interaction for options */
    [data-testid="stSidebar"] div[data-testid="stRadio"] label:hover p {
        color: #38BDF8 !important;
    }
    /* Style active state in radio buttons */
    [data-testid="stSidebar"] div[data-testid="stRadio"] label[data-selected="true"] p {
        color: #38BDF8 !important;
        font-weight: 700 !important;
    }
    /* Muted purple uppercase for section headers */
    [data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p {
        color: #38BDF8 !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 1.2px !important;
        font-weight: 700 !important;
        margin-bottom: 4px !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #F9FAFB !important;
        font-weight: 700 !important;
    }
    
    /* Custom Card Design - Glassmorphism */
    .metric-card {
        background: #111827 !important;
        border: 1px solid #374151 !important;
        border-radius: 12px !important;
        padding: 24px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.2) !important;
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 20px !important;
    }
    
    .metric-card:hover {
        transform: translateY(-4px) !important;
        border-color: #38BDF8 !important;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 10px 10px -5px rgba(0, 0, 0, 0.3) !important;
    }
    
    .metric-title {
        font-size: 13px !important;
        color: #CBD5E1 !important;
        text-transform: uppercase !important;
        font-weight: 600 !important;
        letter-spacing: 1.2px !important;
        margin-bottom: 8px !important;
    }
    
    .metric-value {
        font-size: 32px !important;
        font-weight: 700 !important;
        color: #38BDF8 !important;
    }
    
    /* Elegant Results Display */
    .result-container {
        padding: 30px !important;
        background: #111827 !important;
        border-radius: 16px !important;
        margin-top: 20px !important;
        text-align: center !important;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4) !important;
        border: 1px solid #374151 !important;
    }
    
    .result-badge {
        display: inline-block !important;
        padding: 8px 16px !important;
        border-radius: 30px !important;
        font-weight: bold !important;
        font-size: 16px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        margin-top: 15px !important;
        border: 2px solid !important;
    }
    
    /* Labels Legibility */
    div[data-testid="stWidgetLabel"] p, label, [data-testid="stWidgetLabel"] label {
        color: #F9FAFB !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        letter-spacing: 0.2px !important;
        margin-bottom: 8px !important;
    }
    
    /* Input Fields Customization (Text & Number Inputs, Dropdowns) */
    div[data-baseweb="input"], 
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"], 
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] div[role="button"],
    div[role="combobox"], 
    div[role="combobox"] > div,
    [data-baseweb="base-input"],
    [data-baseweb="base-input"] > div {
        background-color: #1F2937 !important;
        border-radius: 10px !important;
        border: 1px solid #374151 !important;
        color: #F9FAFB !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-baseweb="input"]:hover, div[data-baseweb="select"]:hover, div[role="combobox"]:hover {
        border-color: #38BDF8 !important;
    }
    
    /* Focus glow on Inputs */
    div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2) !important;
    }
    
    /* Ensure text inside inputs is white */
    div[data-baseweb="input"] input, 
    div[data-baseweb="select"] select,
    div[data-baseweb="select"] div[role="button"] > div,
    div[data-baseweb="select"] span {
        color: #F9FAFB !important;
    }
    
    /* Specific overrides for dropdown control inner text visibility */
    div[data-baseweb="select"] [data-testid="stSelectbox"] div {
        color: #F9FAFB !important;
    }
    
    /* Fallback: if browser background defaults to light/white, allow text color to default to black to prevent invisible numbers */
    input:-webkit-autofill,
    input:-webkit-autofill:hover, 
    input:-webkit-autofill:focus {
        -webkit-text-fill-color: #F9FAFB !important;
        -webkit-box-shadow: 0 0 0px 1000px #1F2937 inset !important;
    }
    
    /* Dropdown Options popover style */
    div[data-baseweb="popover"], ul[role="listbox"], li[role="option"] {
        background-color: #1F2937 !important;
        color: #F9FAFB !important;
        border-color: #374151 !important;
    }
    
    li[role="option"]:hover, li[data-highlighted="true"] {
        background-color: #3B82F6 !important;
        color: #F9FAFB !important;
    }
    
    /* Sliders Customization */
    .stSlider [data-baseweb="slider"] > div {
        background-color: #1F2937 !important;
    }
    
    /* Slider active track */
    .stSlider [data-baseweb="slider"] div[role="presentation"] > div {
        background-color: #3B82F6 !important;
    }
    
    /* Slider Thumb (Handle) */
    .stSlider div[role="slider"] {
        background-color: #38BDF8 !important;
        border: 2px solid #F9FAFB !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.4) !important;
        width: 18px !important;
        height: 18px !important;
        transition: transform 0.2s ease !important;
    }
    
    .stSlider div[role="slider"]:hover {
        transform: scale(1.2) !important;
    }
    
    /* Standard Buttons Customization (e.g. Predict Button) */
    div.stButton > button,
    div[data-testid="stButton"] button,
    button[data-testid="stBaseButton-primary"],
    .stButton button {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: auto !important;
    }
    
    div.stButton > button:hover,
    div[data-testid="stButton"] button:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.4), 0 4px 6px -2px rgba(59, 130, 246, 0.2) !important;
        color: #FFFFFF !important;
    }
    
    div.stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Download Buttons Customization (Success Color gradient - Green) */
    div.stDownloadButton > button,
    div[data-testid="stDownloadButton"] button,
    button[data-testid="stBaseButton-secondary"],
    .stDownloadButton button {
        background: linear-gradient(135deg, #22C55E 0%, #15803D 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px -1px rgba(34, 197, 94, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    div.stDownloadButton > button:hover,
    div[data-testid="stDownloadButton"] button:hover,
    button[data-testid="stBaseButton-secondary"]:hover {
        background: linear-gradient(135deg, #4ADE80 0%, #22C55E 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(34, 197, 94, 0.4), 0 4px 6px -2px rgba(34, 197, 94, 0.2) !important;
        color: #FFFFFF !important;
    }
    
    /* HTML Table Customization */
    table {
        border-collapse: collapse !important;
        width: 100% !important;
        background-color: #111827 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid #374151 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    
    th {
        background-color: #1F2937 !important;
        color: #38BDF8 !important;
        padding: 14px 16px !important;
        text-align: left !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #374151 !important;
        font-size: 13px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    td {
        padding: 14px 16px !important;
        border-bottom: 1px solid #374151 !important;
        color: #CBD5E1 !important;
        font-size: 14px !important;
    }
    
    tr:last-child td {
        border-bottom: none !important;
    }
    
    tr:hover {
        background-color: rgba(56, 189, 248, 0.03) !important;
    }
    
    /* DataFrame Container Rounding */
    div[data-testid="stDataFrame"] {
        border-radius: 12px !important;
        border: 1px solid #374151 !important;
        overflow: hidden !important;
        background-color: #111827 !important;
    }
    
    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: #111827 !important;
        padding: 6px 12px !important;
        border-radius: 10px !important;
        border: 1px solid #374151 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px !important;
        background-color: transparent !important;
        border-radius: 6px !important;
        color: #CBD5E1 !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #F9FAFB !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1F2937 !important;
        color: #38BDF8 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
    }
    
    /* Notification Alerts styling */
    div[data-testid="stAlert"] {
        border-radius: 12px !important;
        border: 1px solid #374151 !important;
        background-color: #111827 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    </style>
    """
