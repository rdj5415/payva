#!/usr/bin/env python3
"""Launch script for AuditPulse dashboard.

This module provides a function to run the Streamlit dashboard
with the correct configuration and environment.
"""
import os
import sys
import subprocess
from pathlib import Path


def run_dashboard():
    """Run the Streamlit dashboard."""
    try:
        # Get the directory containing this script
        script_dir = Path(__file__).parent.absolute()
        
        # Add the parent directory to Python path
        parent_dir = script_dir.parent
        sys.path.insert(0, str(parent_dir))
        
        # Set environment variables
        os.environ["STREAMLIT_SERVER_PORT"] = "8501"
        os.environ["STREAMLIT_SERVER_ADDRESS"] = "localhost"
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
        os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        
        # Run Streamlit
        subprocess.run([
            "streamlit",
            "run",
            str(script_dir / "dashboard.py"),
            "--server.port=8501",
            "--server.address=localhost",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ])
        
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error running dashboard: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    run_dashboard() 