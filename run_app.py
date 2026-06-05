import sys
import os
import streamlit.web.cli as stcli

if __name__ == '__main__':
    # Add src directory to system path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
    
    # Configure command line arguments for streamlit
    sys.argv = [
        "streamlit", 
        "run", 
        "app.py", 
        "--server.headless=true", 
        "--server.port=8501"
    ]
    
    print("Starting Student Performance System Python Server...")
    print("Navigate to http://localhost:8501 in your browser.")
    
    # Launch Streamlit programmatically
    sys.exit(stcli.main())
