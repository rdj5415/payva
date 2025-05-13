"""Tests for the dashboard functionality.

This module contains tests for the Streamlit dashboard,
including data retrieval, filtering, and visualization.
"""

import datetime
import uuid
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy.orm import Session

from auditpulse_mvp.database.models import Anomaly, RiskLevel, Tenant
from auditpulse_mvp.dashboard.dashboard import (
    get_anomalies,
    get_risk_config,
    update_risk_config,
    resolve_anomaly,
)


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = MagicMock(spec=Session)
    return session


@pytest.fixture
def mock_tenant():
    """Create a mock tenant."""
    tenant = MagicMock(spec=Tenant)
    tenant.id = uuid.uuid4()
    tenant.name = "Test Tenant"
    return tenant


@pytest.fixture
def mock_anomalies():
    """Create mock anomalies."""
    anomalies = []
    for i in range(3):
        anomaly = MagicMock(spec=Anomaly)
        anomaly.id = uuid.uuid4()
        anomaly.tenant_id = uuid.uuid4()
        anomaly.transaction_id = uuid.uuid4()
        anomaly.risk_level = RiskLevel.HIGH if i == 0 else RiskLevel.MEDIUM
        anomaly.anomaly_type = "unusual_amount"
        anomaly.score = 0.95 - (i * 0.1)
        anomaly.details = {
            "amount": 1000.0 - (i * 100),
            "date": datetime.datetime.now().isoformat(),
            "description": f"Test transaction {i}",
        }
        anomaly.explanation = f"This is test anomaly {i}"
        anomaly.is_resolved = False
        anomaly.feedback_type = None
        anomaly.resolution_notes = None
        anomaly.resolved_at = None
        anomalies.append(anomaly)
    return anomalies


def test_get_anomalies(
    mock_db_session,
    mock_tenant,
    mock_anomalies,
):
    """Test retrieving anomalies with filters."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.all.return_value = (
        mock_anomalies
    )

    # Get anomalies
    anomalies_df = get_anomalies(
        db=mock_db_session,
        tenant_id=mock_tenant.id,
        start_date=datetime.datetime.now() - datetime.timedelta(days=7),
        end_date=datetime.datetime.now(),
        min_score=0.5,
        risk_levels=[RiskLevel.HIGH, RiskLevel.MEDIUM],
        anomaly_types=["unusual_amount"],
        is_resolved=False,
    )

    # Verify results
    assert isinstance(anomalies_df, pd.DataFrame)
    assert len(anomalies_df) == len(mock_anomalies)
    assert all(
        col in anomalies_df.columns
        for col in [
            "id",
            "risk_level",
            "anomaly_type",
            "score",
            "amount",
            "date",
            "description",
            "explanation",
            "is_resolved",
        ]
    )


def test_get_anomalies_empty(
    mock_db_session,
    mock_tenant,
):
    """Test retrieving anomalies with no results."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.all.return_value = []

    # Get anomalies
    anomalies_df = get_anomalies(
        db=mock_db_session,
        tenant_id=mock_tenant.id,
        start_date=datetime.datetime.now() - datetime.timedelta(days=7),
        end_date=datetime.datetime.now(),
    )

    # Verify results
    assert isinstance(anomalies_df, pd.DataFrame)
    assert len(anomalies_df) == 0


def test_get_risk_config(
    mock_db_session,
    mock_tenant,
):
    """Test retrieving risk configuration."""
    # Setup
    mock_config = {
        "weights": {
            "amount": 0.4,
            "frequency": 0.3,
            "category": 0.3,
        },
        "thresholds": {
            "high_risk": 0.8,
            "medium_risk": 0.5,
            "low_risk": 0.2,
        },
    }
    mock_db_session.query.return_value.filter.return_value.first.return_value = (
        mock_config
    )

    # Get configuration
    config = get_risk_config(mock_db_session, mock_tenant.id)

    # Verify configuration
    assert config == mock_config


def test_update_risk_config(
    mock_db_session,
    mock_tenant,
):
    """Test updating risk configuration."""
    # Setup
    new_config = {
        "weights": {
            "amount": 0.5,
            "frequency": 0.3,
            "category": 0.2,
        },
        "thresholds": {
            "high_risk": 0.85,
            "medium_risk": 0.55,
            "low_risk": 0.25,
        },
    }

    # Update configuration
    update_risk_config(
        db=mock_db_session,
        tenant_id=mock_tenant.id,
        config=new_config,
    )

    # Verify update
    mock_db_session.query.return_value.filter.return_value.update.assert_called_once()
    mock_db_session.commit.assert_called_once()


def test_resolve_anomaly(
    mock_db_session,
    mock_tenant,
    mock_anomalies,
):
    """Test resolving an anomaly."""
    # Setup
    anomaly = mock_anomalies[0]
    mock_db_session.query.return_value.filter.return_value.first.return_value = anomaly

    # Resolve anomaly
    resolve_anomaly(
        db=mock_db_session,
        anomaly_id=anomaly.id,
        feedback_type="true_positive",
        resolution_notes="Test resolution",
    )

    # Verify resolution
    assert anomaly.is_resolved is True
    assert anomaly.feedback_type == "true_positive"
    assert anomaly.resolution_notes == "Test resolution"
    assert anomaly.resolved_at is not None
    mock_db_session.commit.assert_called_once()


def test_resolve_anomaly_not_found(
    mock_db_session,
    mock_tenant,
):
    """Test resolving non-existent anomaly."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.first.return_value = None

    # Attempt to resolve anomaly
    with pytest.raises(ValueError):
        resolve_anomaly(
            db=mock_db_session,
            anomaly_id=uuid.uuid4(),
            feedback_type="true_positive",
            resolution_notes="Test resolution",
        )
