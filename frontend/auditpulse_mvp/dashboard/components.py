"""Dashboard components for AuditPulse AI.

This module provides reusable UI components for the Streamlit dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from auditpulse_mvp.database.models import Transaction, Anomaly, AnomalyType
from auditpulse_mvp.gpt_engine.gpt_engine import get_gpt_engine


def render_anomaly_details(anomaly: Anomaly) -> None:
    """Render detailed view of an anomaly.
    
    Args:
        anomaly: Anomaly to display
    """
    st.subheader("Anomaly Details")
    
    # Basic info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Risk Score", f"{anomaly.risk_score:.2f}")
        st.write(f"**Type:** {anomaly.type}")
        st.write(f"**Created:** {anomaly.created_at}")
    with col2:
        st.metric("Transaction Amount", f"${anomaly.amount:,.2f}")
        st.write(f"**Status:** {anomaly.status}")
        st.write(f"**Resolution:** {anomaly.resolution or 'Pending'}")
    
    # GPT explanation
    st.subheader("AI Analysis")
    gpt_engine = get_gpt_engine()
    explanation = gpt_engine.get_anomaly_explanation(anomaly)
    st.write(explanation)
    
    # Feedback
    st.subheader("Feedback")
    feedback = st.radio(
        "Is this anomaly valid?",
        ["Yes", "No", "Not Sure"],
        horizontal=True
    )
    if feedback != "Not Sure":
        st.text_area("Additional Comments", key=f"feedback_{anomaly.id}")
        if st.button("Submit Feedback", key=f"submit_{anomaly.id}"):
            # TODO: Save feedback
            st.success("Feedback submitted!")


def render_transaction_details(transaction: Transaction) -> None:
    """Render detailed view of a transaction.
    
    Args:
        transaction: Transaction to display
    """
    st.subheader("Transaction Details")
    
    # Basic info
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**ID:** {transaction.transaction_id}")
        st.write(f"**Date:** {transaction.transaction_date}")
        st.write(f"**Category:** {transaction.category}")
    with col2:
        st.write(f"**Amount:** ${transaction.amount:,.2f}")
        st.write(f"**Vendor:** {transaction.merchant_name}")
        st.write(f"**Source:** {transaction.source}")
    
    # Description
    st.write(f"**Description:** {transaction.description}")
    
    # Related anomalies
    if transaction.anomalies:
        st.subheader("Related Anomalies")
        for anomaly in transaction.anomalies:
            with st.expander(f"{anomaly.type} - {anomaly.created_at}"):
                render_anomaly_details(anomaly)


def render_risk_metrics(transactions: List[Transaction], anomalies: List[Anomaly]) -> None:
    """Render risk metrics visualization.
    
    Args:
        transactions: List of transactions
        anomalies: List of anomalies
    """
    st.subheader("Risk Metrics")
    
    # Calculate metrics
    total_amount = sum(t.amount for t in transactions)
    anomaly_amount = sum(a.amount for a in anomalies)
    risk_ratio = anomaly_amount / total_amount if total_amount > 0 else 0
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Risk Ratio",
            f"{risk_ratio:.1%}",
            delta=f"{risk_ratio - 0.1:.1%}"
        )
    with col2:
        st.metric(
            "Anomaly Amount",
            f"${anomaly_amount:,.2f}",
            delta=f"${anomaly_amount - 1000:,.2f}"
        )
    with col3:
        st.metric(
            "Resolution Rate",
            "85%",
            delta="5%"
        )
    
    # Risk trend
    df = pd.DataFrame([
        {
            "date": a.created_at,
            "risk_score": a.risk_score,
            "type": a.type
        }
        for a in anomalies
    ])
    
    if not df.empty:
        fig = px.line(
            df,
            x="date",
            y="risk_score",
            color="type",
            title="Risk Score Trend"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_notification_settings() -> None:
    """Render notification settings form."""
    st.subheader("Notification Settings")
    
    # Email notifications
    st.checkbox("Email Notifications", value=True)
    st.text_input("Email Address")
    
    # Notification types
    st.multiselect(
        "Notify me about",
        ["High Risk Anomalies", "New Anomalies", "Resolved Anomalies"],
        default=["High Risk Anomalies"]
    )
    
    # Frequency
    st.selectbox(
        "Notification Frequency",
        ["Immediate", "Daily Digest", "Weekly Summary"]
    )
    
    if st.button("Save Settings"):
        st.success("Settings saved!")


def render_feedback_form(anomaly: Optional[Anomaly] = None) -> None:
    """Render feedback form.
    
    Args:
        anomaly: Optional anomaly to provide feedback for
    """
    st.subheader("Provide Feedback")
    
    if anomaly:
        st.write(f"Anomaly: {anomaly.type} - {anomaly.created_at}")
    
    # Feedback type
    feedback_type = st.radio(
        "Feedback Type",
        ["Anomaly Accuracy", "System Performance", "Feature Request", "Bug Report"],
        horizontal=True
    )
    
    # Rating
    st.slider("Rating", 1, 5, 3)
    
    # Comments
    st.text_area("Comments")
    
    # Submit
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!") 