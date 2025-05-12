"""Tests for the feedback service module.

This module contains tests for the feedback service functionality,
including feedback submission, statistics retrieval, and ML training data updates.
"""
import datetime
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session

from auditpulse_mvp.database.models import Anomaly, FeedbackType, Transaction
from auditpulse_mvp.feedback.feedback_service import FeedbackRequest, FeedbackService


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = MagicMock(spec=Session)
    return session


@pytest.fixture
def mock_ml_engine():
    """Create a mock ML engine."""
    engine = AsyncMock()
    engine.add_training_example = AsyncMock()
    return engine


@pytest.fixture
def mock_risk_engine():
    """Create a mock risk engine."""
    engine = AsyncMock()
    engine.adjust_risk_scoring = AsyncMock()
    return engine


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
    anomaly.transaction = MagicMock(spec=Transaction)
    anomaly.transaction.amount = 100.0
    anomaly.transaction.description = "Test transaction"
    anomaly.transaction.merchant_name = "Test merchant"
    anomaly.transaction.category = "Test category"
    anomaly.transaction.transaction_date = datetime.datetime.now()
    anomaly.anomaly_type.value = "TEST"
    anomaly.score = 0.8
    anomaly.risk_level.value = "HIGH"
    return anomaly


@pytest.mark.asyncio
async def test_submit_feedback_success(
    mock_db_session,
    mock_ml_engine,
    mock_risk_engine,
    mock_anomaly,
):
    """Test successful feedback submission."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_anomaly
    
    with patch("auditpulse_mvp.feedback.feedback_service.get_ml_engine", return_value=mock_ml_engine), \
         patch("auditpulse_mvp.feedback.feedback_service.get_risk_engine", return_value=mock_risk_engine):
        
        service = FeedbackService(mock_db_session)
        
        # Create feedback request
        feedback = FeedbackRequest(
            anomaly_id=mock_anomaly.id,
            feedback_type=FeedbackType.TRUE_POSITIVE,
            resolution_notes="Test feedback",
        )
        
        # Submit feedback
        response = await service.submit_feedback(
            tenant_id=mock_anomaly.tenant_id,
            feedback=feedback,
        )
        
        # Verify response
        assert response.success
        assert response.anomaly_id == mock_anomaly.id
        assert response.feedback_type == feedback.feedback_type
        assert response.timestamp is not None
        
        # Verify anomaly update
        assert mock_anomaly.is_resolved
        assert mock_anomaly.feedback_type == feedback.feedback_type
        assert mock_anomaly.resolution_notes == feedback.resolution_notes
        assert mock_anomaly.resolved_at is not None
        
        # Verify ML training data update
        mock_ml_engine.add_training_example.assert_called_once()
        
        # Verify risk scoring update (should not be called for TRUE_POSITIVE)
        mock_risk_engine.adjust_risk_scoring.assert_not_called()


@pytest.mark.asyncio
async def test_submit_feedback_false_positive(
    mock_db_session,
    mock_ml_engine,
    mock_risk_engine,
    mock_anomaly,
):
    """Test feedback submission for false positive."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_anomaly
    
    with patch("auditpulse_mvp.feedback.feedback_service.get_ml_engine", return_value=mock_ml_engine), \
         patch("auditpulse_mvp.feedback.feedback_service.get_risk_engine", return_value=mock_risk_engine):
        
        service = FeedbackService(mock_db_session)
        
        # Create feedback request
        feedback = FeedbackRequest(
            anomaly_id=mock_anomaly.id,
            feedback_type=FeedbackType.FALSE_POSITIVE,
            resolution_notes="Test feedback",
        )
        
        # Submit feedback
        response = await service.submit_feedback(
            tenant_id=mock_anomaly.tenant_id,
            feedback=feedback,
        )
        
        # Verify response
        assert response.success
        
        # Verify risk scoring update
        mock_risk_engine.adjust_risk_scoring.assert_called_once_with(
            transaction=mock_anomaly.transaction,
            adjustment_factor=0.5,
        )


@pytest.mark.asyncio
async def test_submit_feedback_false_negative(
    mock_db_session,
    mock_ml_engine,
    mock_risk_engine,
    mock_anomaly,
):
    """Test feedback submission for false negative."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_anomaly
    
    with patch("auditpulse_mvp.feedback.feedback_service.get_ml_engine", return_value=mock_ml_engine), \
         patch("auditpulse_mvp.feedback.feedback_service.get_risk_engine", return_value=mock_risk_engine):
        
        service = FeedbackService(mock_db_session)
        
        # Create feedback request
        feedback = FeedbackRequest(
            anomaly_id=mock_anomaly.id,
            feedback_type=FeedbackType.FALSE_NEGATIVE,
            resolution_notes="Test feedback",
        )
        
        # Submit feedback
        response = await service.submit_feedback(
            tenant_id=mock_anomaly.tenant_id,
            feedback=feedback,
        )
        
        # Verify response
        assert response.success
        
        # Verify risk scoring update
        mock_risk_engine.adjust_risk_scoring.assert_called_once_with(
            transaction=mock_anomaly.transaction,
            adjustment_factor=2.0,
        )


@pytest.mark.asyncio
async def test_submit_feedback_not_found(
    mock_db_session,
    mock_ml_engine,
    mock_risk_engine,
):
    """Test feedback submission for non-existent anomaly."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.first.return_value = None
    
    with patch("auditpulse_mvp.feedback.feedback_service.get_ml_engine", return_value=mock_ml_engine), \
         patch("auditpulse_mvp.feedback.feedback_service.get_risk_engine", return_value=mock_risk_engine):
        
        service = FeedbackService(mock_db_session)
        
        # Create feedback request
        feedback = FeedbackRequest(
            anomaly_id=uuid.uuid4(),
            feedback_type=FeedbackType.TRUE_POSITIVE,
            resolution_notes="Test feedback",
        )
        
        # Submit feedback and expect exception
        with pytest.raises(HTTPException) as exc_info:
            await service.submit_feedback(
                tenant_id=uuid.uuid4(),
                feedback=feedback,
            )
        
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_feedback_stats(
    mock_db_session,
    mock_anomaly,
):
    """Test feedback statistics retrieval."""
    # Setup
    mock_anomaly.feedback_type = FeedbackType.TRUE_POSITIVE
    mock_anomaly.is_resolved = True
    mock_anomaly.resolved_at = datetime.datetime.now()
    
    mock_db_session.query.return_value.filter.return_value.all.return_value = [mock_anomaly]
    
    service = FeedbackService(mock_db_session)
    
    # Get statistics
    stats = await service.get_feedback_stats(
        tenant_id=mock_anomaly.tenant_id,
    )
    
    # Verify statistics
    assert stats["total_feedback"] == 1
    assert stats["feedback_types"][FeedbackType.TRUE_POSITIVE] == 1
    assert "accuracy_metrics" in stats
    assert "precision" in stats["accuracy_metrics"]
    assert "recall" in stats["accuracy_metrics"]
    assert "f1_score" in stats["accuracy_metrics"] 