"""Tests for the feedback API endpoints.

This module contains tests for the feedback API endpoints,
including feedback submission and statistics retrieval.
"""
import datetime
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from auditpulse_mvp.api.api_v1.endpoints.feedback import router
from auditpulse_mvp.database.models import Anomaly, FeedbackType, User
from auditpulse_mvp.feedback.feedback_service import FeedbackResponse, FeedbackService


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
def mock_current_user():
    """Create a mock current user."""
    user = MagicMock(spec=User)
    user.tenant_id = uuid.uuid4()
    return user


@pytest.fixture
def mock_anomaly():
    """Create a mock anomaly."""
    anomaly = MagicMock(spec=Anomaly)
    anomaly.id = uuid.uuid4()
    anomaly.tenant_id = uuid.uuid4()
    anomaly.is_resolved = False
    anomaly.feedback_type = None
    anomaly.resolution_notes = None
    anomaly.resolved_at = None
    return anomaly


def test_submit_feedback_success(
    client,
    mock_db_session,
    mock_current_user,
    mock_anomaly,
):
    """Test successful feedback submission."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_anomaly
    
    with patch("auditpulse_mvp.api.api_v1.endpoints.feedback.get_db_session", return_value=mock_db_session), \
         patch("auditpulse_mvp.api.api_v1.endpoints.feedback.get_current_user", return_value=mock_current_user), \
         patch("auditpulse_mvp.api.api_v1.endpoints.feedback.FeedbackService") as mock_service:
        
        # Mock service response
        mock_service.return_value.submit_feedback.return_value = FeedbackResponse(
            success=True,
            message="Feedback submitted successfully",
            anomaly_id=mock_anomaly.id,
            feedback_type=FeedbackType.TRUE_POSITIVE,
            timestamp=datetime.datetime.now(),
        )
        
        # Submit feedback
        response = client.post(
            "/api/v1/feedback",
            json={
                "anomaly_id": str(mock_anomaly.id),
                "feedback_type": FeedbackType.TRUE_POSITIVE.value,
                "resolution_notes": "Test feedback",
            },
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert data["anomaly_id"] == str(mock_anomaly.id)
        assert data["feedback_type"] == FeedbackType.TRUE_POSITIVE.value


def test_submit_feedback_not_found(
    client,
    mock_db_session,
    mock_current_user,
):
    """Test feedback submission for non-existent anomaly."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.first.return_value = None
    
    with patch("auditpulse_mvp.api.api_v1.endpoints.feedback.get_db_session", return_value=mock_db_session), \
         patch("auditpulse_mvp.api.api_v1.endpoints.feedback.get_current_user", return_value=mock_current_user), \
         patch("auditpulse_mvp.api.api_v1.endpoints.feedback.FeedbackService") as mock_service:
        
        # Mock service exception
        mock_service.return_value.submit_feedback.side_effect = Exception("Anomaly not found")
        
        # Submit feedback
        response = client.post(
            "/api/v1/feedback",
            json={
                "anomaly_id": str(uuid.uuid4()),
                "feedback_type": FeedbackType.TRUE_POSITIVE.value,
                "resolution_notes": "Test feedback",
            },
        )
        
        # Verify response
        assert response.status_code == 500


def test_get_feedback_stats(
    client,
    mock_db_session,
    mock_current_user,
):
    """Test feedback statistics retrieval."""
    # Setup
    stats = {
        "total_feedback": 1,
        "feedback_types": {
            FeedbackType.TRUE_POSITIVE.value: 1,
            FeedbackType.FALSE_POSITIVE.value: 0,
            FeedbackType.TRUE_NEGATIVE.value: 0,
            FeedbackType.FALSE_NEGATIVE.value: 0,
        },
        "accuracy_metrics": {
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
        },
    }
    
    with patch("auditpulse_mvp.api.api_v1.endpoints.feedback.get_db_session", return_value=mock_db_session), \
         patch("auditpulse_mvp.api.api_v1.endpoints.feedback.get_current_user", return_value=mock_current_user), \
         patch("auditpulse_mvp.api.api_v1.endpoints.feedback.FeedbackService") as mock_service:
        
        # Mock service response
        mock_service.return_value.get_feedback_stats.return_value = stats
        
        # Get statistics
        response = client.get("/api/v1/feedback/stats")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data == stats


def test_get_anomaly_feedback(
    client,
    mock_db_session,
    mock_current_user,
    mock_anomaly,
):
    """Test getting feedback for a specific anomaly."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_anomaly
    
    with patch("auditpulse_mvp.api.api_v1.endpoints.feedback.get_db_session", return_value=mock_db_session), \
         patch("auditpulse_mvp.api.api_v1.endpoints.feedback.get_current_user", return_value=mock_current_user):
        
        # Get feedback
        response = client.get(f"/api/v1/feedback/{mock_anomaly.id}")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["anomaly_id"] == str(mock_anomaly.id)
        assert data["is_resolved"] == mock_anomaly.is_resolved
        assert data["feedback_type"] is None
        assert data["resolution_notes"] is None
        assert data["resolved_at"] is None


def test_get_anomaly_feedback_not_found(
    client,
    mock_db_session,
    mock_current_user,
):
    """Test getting feedback for non-existent anomaly."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.first.return_value = None
    
    with patch("auditpulse_mvp.api.api_v1.endpoints.feedback.get_db_session", return_value=mock_db_session), \
         patch("auditpulse_mvp.api.api_v1.endpoints.feedback.get_current_user", return_value=mock_current_user):
        
        # Get feedback
        response = client.get(f"/api/v1/feedback/{uuid.uuid4()}")
        
        # Verify response
        assert response.status_code == 404 