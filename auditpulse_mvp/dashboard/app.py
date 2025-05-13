"""AuditPulse AI Dashboard.

This module provides the main Streamlit dashboard for AuditPulse AI.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import asyncio
from typing import List, Dict, Any

from auditpulse_mvp.database.models import Transaction, Anomaly, AnomalyType
from auditpulse_mvp.database.session import get_db
from auditpulse_mvp.rules_engine.rules_engine import RulesEngine
from auditpulse_mvp.ml_engine.ml_engine import get_ml_engine
from auditpulse_mvp.gpt_engine.gpt_engine import get_gpt_engine

# Page config
st.set_page_config(
    page_title="AuditPulse AI Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "selected_tenant" not in st.session_state:
    st.session_state.selected_tenant = None
if "date_range" not in st.session_state:
    st.session_state.date_range = (datetime.now() - timedelta(days=30), datetime.now())


async def get_tenant_data(tenant_id: str) -> Dict[str, Any]:
    """Get data for a specific tenant."""
    async with get_db() as db:
        # Get transactions
        transactions = (
            await db.query(Transaction)
            .filter(
                Transaction.tenant_id == tenant_id,
                Transaction.transaction_date >= st.session_state.date_range[0],
                Transaction.transaction_date <= st.session_state.date_range[1],
            )
            .all()
        )

        # Get anomalies
        anomalies = (
            await db.query(Anomaly)
            .filter(
                Anomaly.tenant_id == tenant_id,
                Anomaly.created_at >= st.session_state.date_range[0],
                Anomaly.created_at <= st.session_state.date_range[1],
            )
            .all()
        )

        return {"transactions": transactions, "anomalies": anomalies}


def render_dashboard():
    """Render the main dashboard."""
    st.title("AuditPulse AI Dashboard")

    # Sidebar
    with st.sidebar:
        st.header("Filters")

        # Tenant selection
        tenants = ["Tenant 1", "Tenant 2", "Tenant 3"]  # TODO: Get from DB
        st.session_state.selected_tenant = st.selectbox(
            "Select Tenant", tenants, index=0
        )

        # Date range
        st.session_state.date_range = st.date_input(
            "Date Range", value=st.session_state.date_range, max_value=datetime.now()
        )

        # Refresh button
        if st.button("Refresh Data"):
            st.experimental_rerun()

    # Main content
    if st.session_state.selected_tenant:
        # Get data
        data = asyncio.run(get_tenant_data(st.session_state.selected_tenant))

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Total Transactions",
                len(data["transactions"]),
                delta=f"{len(data['transactions']) - 100} from last period",
            )
        with col2:
            st.metric(
                "Anomalies Detected",
                len(data["anomalies"]),
                delta=f"{len(data['anomalies']) - 10} from last period",
            )
        with col3:
            st.metric("Risk Score", "Medium", delta="High")
        with col4:
            st.metric("Resolution Rate", "85%", delta="5%")

        # Transaction trends
        st.subheader("Transaction Trends")
        df = pd.DataFrame(
            [
                {"date": t.transaction_date, "amount": t.amount, "category": t.category}
                for t in data["transactions"]
            ]
        )

        fig = px.line(
            df,
            x="date",
            y="amount",
            color="category",
            title="Transaction Amounts Over Time",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Anomaly distribution
        st.subheader("Anomaly Distribution")
        anomaly_counts = (
            pd.DataFrame([{"type": a.type, "count": 1} for a in data["anomalies"]])
            .groupby("type")
            .count()
            .reset_index()
        )

        fig = px.pie(
            anomaly_counts, values="count", names="type", title="Anomaly Types"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Recent anomalies
        st.subheader("Recent Anomalies")
        for anomaly in data["anomalies"][:5]:
            with st.expander(f"{anomaly.type} - {anomaly.created_at}"):
                st.write(f"**Transaction:** {anomaly.transaction_id}")
                st.write(f"**Amount:** ${anomaly.amount:,.2f}")
                st.write(f"**Description:** {anomaly.description}")
                st.write(f"**Risk Score:** {anomaly.risk_score}")

                if st.button("View Details", key=anomaly.id):
                    st.write("Detailed analysis...")  # TODO: Add detailed view


if __name__ == "__main__":
    render_dashboard()
