"""Tests for the metrics API endpoints.

This module contains tests for the metrics API endpoints,
including dashboard metrics and risk trends.
"""

import datetime
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from auditpulse_mvp.api.api_v1.endpoints.metrics import router
from auditpulse_mvp.database.models import Anomaly, AnomalyStatus, AnomalyRiskLevel


@pytest.fixture
def app():
    """Create a test FastAPI application."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = MagicMock(spec=Session)
    return session


@pytest.fixture
def mock_anomalies():
    """Create mock anomalies."""
    now = datetime.datetime.now()
    week_ago = now - datetime.timedelta(days=7)

    anomalies = []

    # Create high risk anomalies
    for i in range(5):
        anomaly = MagicMock(spec=Anomaly)
        anomaly.id = uuid.uuid4()
        anomaly.tenant_id = uuid.uuid4()
        anomaly.risk_level = AnomalyRiskLevel.HIGH
        anomaly.status = AnomalyStatus.OPEN
        anomaly.created_at = now - datetime.timedelta(days=i)
        anomaly.resolved_at = None
        anomaly.feedback_type = "true_positive" if i < 3 else None
        anomalies.append(anomaly)

    # Create medium risk anomalies
    for i in range(3):
        anomaly = MagicMock(spec=Anomaly)
        anomaly.id = uuid.uuid4()
        anomaly.tenant_id = uuid.uuid4()
        anomaly.risk_level = AnomalyRiskLevel.MEDIUM
        anomaly.status = AnomalyStatus.RESOLVED
        anomaly.created_at = week_ago - datetime.timedelta(days=i)
        anomaly.resolved_at = now - datetime.timedelta(days=i)
        anomaly.feedback_type = "false_positive"
        anomalies.append(anomaly)

    # Create low risk anomalies
    for i in range(2):
        anomaly = MagicMock(spec=Anomaly)
        anomaly.id = uuid.uuid4()
        anomaly.tenant_id = uuid.uuid4()
        anomaly.risk_level = AnomalyRiskLevel.LOW
        anomaly.status = AnomalyStatus.RESOLVED
        anomaly.created_at = week_ago - datetime.timedelta(days=i)
        anomaly.resolved_at = now - datetime.timedelta(days=i)
        anomaly.feedback_type = "true_positive"
        anomalies.append(anomaly)

    return anomalies


def test_get_metrics_success(
    client,
    mock_db_session,
    mock_anomalies,
):
    """Test successful metrics retrieval."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.scalar.side_effect = [
        10,  # total_anomalies
        5,  # previous_total
        5,  # high_risk_count
        2,  # previous_high_risk
        8,  # total_feedback
        5,  # true_positives
        3,  # previous_true_positives
        5,  # previous_total_feedback
    ]

    mock_db_session.query.return_value.filter.return_value.all.return_value = [
        anomaly
        for anomaly in mock_anomalies
        if anomaly.status == AnomalyStatus.RESOLVED
    ]

    with patch(
        "auditpulse_mvp.api.api_v1.endpoints.metrics.get_db_session",
        return_value=mock_db_session,
    ), patch(
        "auditpulse_mvp.api.api_v1.endpoints.metrics.get_current_user",
        return_value=MagicMock(),
    ):

        # Get metrics
        response = client.get(
            "/api/v1/metrics",
            params={"tenant_id": str(uuid.uuid4())},
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "total_anomalies" in data
        assert "anomaly_change" in data
        assert "high_risk_count" in data
        assert "high_risk_change" in data
        assert "accuracy" in data
        assert "accuracy_change" in data
        assert "avg_response_time" in data
        assert "response_time_change" in data


def test_get_metrics_error(
    client,
    mock_db_session,
):
    """Test metrics retrieval with error."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.scalar.side_effect = (
        Exception("Database error")
    )

    with patch(
        "auditpulse_mvp.api.api_v1.endpoints.metrics.get_db_session",
        return_value=mock_db_session,
    ), patch(
        "auditpulse_mvp.api.api_v1.endpoints.metrics.get_current_user",
        return_value=MagicMock(),
    ):

        # Get metrics
        response = client.get(
            "/api/v1/metrics",
            params={"tenant_id": str(uuid.uuid4())},
        )

        # Verify response
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Failed to get metrics" in data["detail"]


def test_get_risk_metrics_success(
    client,
    mock_db_session,
    mock_anomalies,
):
    """Test successful risk metrics retrieval."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.scalar.side_effect = [
        2,  # high_risk
        1,  # medium_risk
        1,  # low_risk
    ] * 7  # 7 days

    with patch(
        "auditpulse_mvp.api.api_v1.endpoints.metrics.get_db_session",
        return_value=mock_db_session,
    ), patch(
        "auditpulse_mvp.api.api_v1.endpoints.metrics.get_current_user",
        return_value=MagicMock(),
    ):

        # Get risk metrics
        response = client.get(
            "/api/v1/metrics/risk",
            params={
                "tenant_id": str(uuid.uuid4()),
                "period": "7d",
            },
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "risk_trends" in data
        assert len(data["risk_trends"]) == 7

        for trend in data["risk_trends"]:
            assert "timestamp" in trend
            assert "high_risk" in trend
            assert "medium_risk" in trend
            assert "low_risk" in trend


def test_get_risk_metrics_error(
    client,
    mock_db_session,
):
    """Test risk metrics retrieval with error."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.scalar.side_effect = (
        Exception("Database error")
    )

    with patch(
        "auditpulse_mvp.api.api_v1.endpoints.metrics.get_db_session",
        return_value=mock_db_session,
    ), patch(
        "auditpulse_mvp.api.api_v1.endpoints.metrics.get_current_user",
        return_value=MagicMock(),
    ):

        # Get risk metrics
        response = client.get(
            "/api/v1/metrics/risk",
            params={
                "tenant_id": str(uuid.uuid4()),
                "period": "7d",
            },
        )

        # Verify response
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Failed to get risk metrics" in data["detail"]
