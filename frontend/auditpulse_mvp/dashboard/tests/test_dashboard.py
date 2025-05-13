"""Tests for the AuditPulse dashboard.

This module contains tests for the Streamlit dashboard components.
"""
import pytest
from datetime import datetime, timedelta
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from auditpulse_mvp.database.models import Transaction, Anomaly, AnomalyType, DataSource
from auditpulse_mvp.dashboard.components import (
    render_anomaly_details,
    render_transaction_details,
    render_risk_metrics,
    render_notification_settings,
    render_feedback_form,
)


@pytest.fixture
def mock_transaction():
    """Create a mock transaction for testing."""
    return Transaction(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        transaction_id="test-txn-001",
        source=DataSource.QUICKBOOKS,
        source_account_id="test-account",
        amount=5000.0,
        currency="USD",
        description="Test transaction",
        category="Office Supplies",
        merchant_name="Test Vendor",
        transaction_date=datetime.now(),
    )


@pytest.fixture
def mock_anomaly():
    """Create a mock anomaly for testing."""
    return Anomaly(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        transaction_id="test-txn-001",
        type=AnomalyType.LARGE_AMOUNT,
        risk_score=0.8,
        amount=5000.0,
        description="Test anomaly",
        status="open",
        created_at=datetime.now(),
    )


@pytest.fixture
def mock_gpt_engine():
    """Create a mock GPT engine for testing."""
    engine = MagicMock()
    engine.get_anomaly_explanation.return_value = "This is a test explanation."
    return engine


def test_render_anomaly_details(mock_anomaly, mock_gpt_engine):
    """Test rendering anomaly details."""
    with patch("auditpulse_mvp.dashboard.components.get_gpt_engine", return_value=mock_gpt_engine):
        render_anomaly_details(mock_anomaly)
        mock_gpt_engine.get_anomaly_explanation.assert_called_once_with(mock_anomaly)


def test_render_transaction_details(mock_transaction, mock_anomaly):
    """Test rendering transaction details."""
    mock_transaction.anomalies = [mock_anomaly]
    render_transaction_details(mock_transaction)


def test_render_risk_metrics(mock_transaction, mock_anomaly):
    """Test rendering risk metrics."""
    transactions = [mock_transaction]
    anomalies = [mock_anomaly]
    render_risk_metrics(transactions, anomalies)


def test_render_notification_settings():
    """Test rendering notification settings."""
    render_notification_settings()


def test_render_feedback_form(mock_anomaly):
    """Test rendering feedback form."""
    render_feedback_form(mock_anomaly)
    render_feedback_form(None)  # Test without anomaly


@pytest.mark.asyncio
async def test_get_tenant_data(mock_transaction, mock_anomaly):
    """Test getting tenant data."""
    # Mock database session
    mock_session = AsyncMock()
    mock_session.query.return_value.filter.return_value.all.return_value = [mock_transaction]
    
    # Mock GPT engine
    mock_gpt_engine = MagicMock()
    mock_gpt_engine.get_anomaly_explanation.return_value = "Test explanation"
    
    with patch("auditpulse_mvp.dashboard.app.get_db", return_value=mock_session), \
         patch("auditpulse_mvp.dashboard.app.get_gpt_engine", return_value=mock_gpt_engine):
        from auditpulse_mvp.dashboard.app import get_tenant_data
        
        data = await get_tenant_data(str(uuid.uuid4()))
        assert "transactions" in data
        assert "anomalies" in data 