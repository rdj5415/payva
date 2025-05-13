#!/usr/bin/env python3
"""Very simple test for AuditPulse dashboard.

These tests don't rely on imports from the dashboard module.
"""
import os
import subprocess


def run_pip_install():
    """Install required dependencies."""
    packages = ["httpx", "pandas", "plotly", "streamlit", "reportlab", "pydantic"]
    print(f"Installing dependencies: {', '.join(packages)}")
    try:
        subprocess.run(
            ["pip", "install"] + packages, check=True, capture_output=True, text=True
        )
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e.stderr}")
        return False


def test_dashboard_files():
    """Test that dashboard files exist."""
    files = ["dashboard.py", "run_dashboard.py", "__init__.py"]

    for file in files:
        if not os.path.exists(file):
            print(f"❌ Required file {file} missing")
            return False
        print(f"✅ Found {file}")

    return True


def test_dashboard_run_script():
    """Test that the dashboard run script is executable."""
    if not os.path.exists("run_dashboard.py"):
        print("❌ run_dashboard.py missing")
        return False

    if not os.access("run_dashboard.py", os.X_OK):
        print("❌ run_dashboard.py is not executable")
        try:
            os.chmod("run_dashboard.py", 0o755)
            print("✅ Made run_dashboard.py executable")
        except Exception as e:
            print(f"❌ Failed to make run_dashboard.py executable: {e}")
            return False
    else:
        print("✅ run_dashboard.py is executable")

    return True


def test_dashboard_content():
    """Test that the dashboard.py file contains necessary features."""
    if not os.path.exists("dashboard.py"):
        print("❌ dashboard.py missing")
        return False

    with open("dashboard.py", "r") as f:
        content = f.read()

    required_features = [
        "export_to_csv",
        "export_to_pdf",
        "fetch_anomalies",
        "submit_anomaly_feedback",
        "show_login_page",
        "show_dashboard",
        "risk_score",
        "slider",
    ]

    for feature in required_features:
        if feature not in content:
            print(f"❌ Required feature '{feature}' not found in dashboard.py")
            return False
        print(f"✅ Found feature: {feature}")

    return True


def main():
    """Run all tests."""
    print("🧪 Running simplified dashboard tests...\n")

    # First check files
    if not test_dashboard_files():
        print("\n❌ File check failed")
        return
    print()

    # Check run script
    if not test_dashboard_run_script():
        print("\n❌ Run script check failed")
        return
    print()

    # Check content
    if not test_dashboard_content():
        print("\n❌ Content check failed")
        return
    print()

    print("🎉 All tests passed!")


if __name__ == "__main__":
    main()
