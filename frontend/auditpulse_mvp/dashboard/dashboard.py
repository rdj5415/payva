"""Streamlit dashboard for AuditPulse MVP.

This module provides the main dashboard interface for AuditPulse,
including authentication, anomaly monitoring, and settings management.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import requests
from typing import List, Dict, Any, Optional
import json

from auditpulse_mvp.config import settings
from auditpulse_mvp.database.models import AnomalyType, FeedbackType, RiskLevel
from .components import (
    render_sidebar,
    render_header,
    render_anomaly_list,
    render_risk_metrics,
    render_risk_settings,
    render_notification_settings,
    render_help,
)


def init_session_state():
    """Initialize session state variables."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user" not in st.session_state:
        st.session_state.user = None
    if "token" not in st.session_state:
        st.session_state.token = None
    if "tenant_id" not in st.session_state:
        st.session_state.tenant_id = None


def login(email: str, password: str) -> bool:
    """Authenticate user and get access token.
    
    Args:
        email: User email
        password: User password
        
    Returns:
        bool: True if authentication successful
    """
    try:
        response = requests.post(
            f"{settings.API_URL}/api/v1/auth/login",
            json={
                "email": email,
                "password": password,
            },
        )
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.token = data["access_token"]
            st.session_state.user = data["user"]
            st.session_state.tenant_id = data["user"]["tenant_id"]
            st.session_state.authenticated = True
            return True
        else:
            st.error("Invalid email or password")
            return False
            
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return False


def register(email: str, password: str, full_name: str) -> bool:
    """Register new user.
    
    Args:
        email: User email
        password: User password
        full_name: User's full name
        
    Returns:
        bool: True if registration successful
    """
    try:
        response = requests.post(
            f"{settings.API_URL}/api/v1/auth/register",
            json={
                "email": email,
                "password": password,
                "full_name": full_name,
            },
        )
        
        if response.status_code == 200:
            st.success("Registration successful! Please log in.")
            return True
        else:
            st.error("Registration failed")
            return False
            
    except Exception as e:
        st.error(f"Registration failed: {str(e)}")
        return False


def get_anomalies(
    start_date: datetime,
    end_date: datetime,
    risk_levels: List[str],
    anomaly_types: List[str],
    resolution_status: str,
) -> List[Dict[str, Any]]:
    """Get anomalies from API with filtering.
    
    Args:
        start_date: Start date for filtering
        end_date: End date for filtering
        risk_levels: List of risk levels to include
        anomaly_types: List of anomaly types to include
        resolution_status: Resolution status filter
        
    Returns:
        List of anomaly dictionaries
    """
    try:
        params = {
            "tenant_id": str(st.session_state.tenant_id),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "risk_levels": ",".join(risk_levels),
            "anomaly_types": ",".join(anomaly_types),
        }
        
        if resolution_status != "All":
            params["is_resolved"] = resolution_status == "Resolved"
        
        response = requests.get(
            f"{settings.API_URL}/api/v1/anomalies",
            headers={"Authorization": f"Bearer {st.session_state.token}"},
            params=params,
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Failed to load anomalies")
            return []
            
    except Exception as e:
        st.error(f"Error loading anomalies: {str(e)}")
        return []


def render_auth_page():
    """Render authentication page."""
    st.title("AuditPulse")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("Login"):
                if login(email, password):
                    st.rerun()
    
    with tab2:
        with st.form("register_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            full_name = st.text_input("Full Name")
            
            if st.form_submit_button("Register"):
                if password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    if register(email, password, full_name):
                        st.rerun()


def render_dashboard():
    """Render main dashboard."""
    # Initialize session state
    init_session_state()
    
    # Check authentication
    if not st.session_state.authenticated:
        render_auth_page()
        return
    
    # Render sidebar
    render_sidebar()
    
    # Get sidebar filters
    date_range = st.session_state.get("date_range", (
        datetime.now() - timedelta(days=7),
        datetime.now(),
    ))
    risk_levels = st.session_state.get("risk_levels", [
        RiskLevel.HIGH.value,
        RiskLevel.MEDIUM.value,
    ])
    anomaly_types = st.session_state.get("anomaly_types", [
        type.value for type in AnomalyType
    ])
    resolution_status = st.session_state.get("resolution_status", "All")
    
    # Get anomalies
    anomalies = get_anomalies(
        date_range[0],
        date_range[1],
        risk_levels,
        anomaly_types,
        resolution_status,
    )
    
    # Render header
    render_header()
    
    # Render main content based on selected page
    page = st.session_state.get("page", "Dashboard")
    
    if page == "Dashboard":
        # Show risk metrics
        render_risk_metrics()
        
        # Show recent anomalies
        st.subheader("Recent Anomalies")
        render_anomaly_list(anomalies[:10])
        
    elif page == "Anomalies":
        # Show all anomalies with filtering
        st.subheader("All Anomalies")
        render_anomaly_list(anomalies)
        
    elif page == "Settings":
        # Show settings tabs
        tab1, tab2 = st.tabs(["Risk Settings", "Notification Settings"])
        
        with tab1:
            render_risk_settings()
            
        with tab2:
            render_notification_settings()
            
    elif page == "Help":
        render_help()


def main():
    """Main entry point for the dashboard."""
    st.set_page_config(
        page_title="AuditPulse",
        page_icon="🔍",
        layout="wide",
    )
    
    render_dashboard()


if __name__ == "__main__":
    main() 