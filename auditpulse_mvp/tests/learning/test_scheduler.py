"""Test cases for the feedback learning scheduler."""
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from auditpulse_mvp.learning.scheduler import (
    FeedbackLearningScheduler,
    DEFAULT_SCHEDULE_HOUR,
    DEFAULT_SCHEDULE_MINUTE,
)


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    return AsyncMock()


@pytest.fixture
def mock_feedback_learner():
    """Create a mock feedback learner."""
    learner = AsyncMock()
    learner.process_recent_feedback = AsyncMock(return_value={"status": "success"})
    return learner


@pytest.mark.asyncio
class TestFeedbackLearningScheduler:
    """Test suite for the FeedbackLearningScheduler class."""
    
    async def test_init(self, mock_db_session):
        """Test that the scheduler initializes correctly."""
        with patch("auditpulse_mvp.learning.scheduler.AsyncIOScheduler") as mock_scheduler_class:
            mock_scheduler = MagicMock()
            mock_scheduler_class.return_value = mock_scheduler
            
            with patch("auditpulse_mvp.learning.scheduler.get_feedback_learner") as mock_get_learner:
                mock_learner = AsyncMock()
                mock_get_learner.return_value = mock_learner
                
                # Act
                scheduler = FeedbackLearningScheduler(db_session=mock_db_session)
                
                # Assert
                assert scheduler.db_session == mock_db_session
                assert scheduler.scheduler == mock_scheduler
                assert scheduler.feedback_learner == mock_learner
                assert scheduler.is_running is False
    
    async def test_start_scheduler(self, mock_db_session, mock_feedback_learner):
        """Test starting the scheduler."""
        # Arrange
        mock_scheduler = MagicMock()
        
        with patch("auditpulse_mvp.learning.scheduler.get_feedback_learner", return_value=mock_feedback_learner):
            scheduler = FeedbackLearningScheduler(db_session=mock_db_session)
            scheduler.scheduler = mock_scheduler
            
            # Act
            await scheduler.start()
            
            # Assert
            assert scheduler.is_running is True
            mock_scheduler.add_job.assert_called_once()
            mock_scheduler.start.assert_called_once()
    
    async def test_stop_scheduler(self, mock_db_session, mock_feedback_learner):
        """Test stopping the scheduler."""
        # Arrange
        mock_scheduler = MagicMock()
        
        with patch("auditpulse_mvp.learning.scheduler.get_feedback_learner", return_value=mock_feedback_learner):
            scheduler = FeedbackLearningScheduler(db_session=mock_db_session)
            scheduler.scheduler = mock_scheduler
            scheduler.is_running = True
            
            # Act
            scheduler.stop()
            
            # Assert
            assert scheduler.is_running is False
            mock_scheduler.shutdown.assert_called_once_with(wait=False)
    
    async def test_run_feedback_learning(self, mock_db_session, mock_feedback_learner):
        """Test running feedback learning process."""
        # Arrange
        tenant_ids = [uuid.uuid4(), uuid.uuid4()]
        
        with patch("auditpulse_mvp.learning.scheduler.get_feedback_learner", return_value=mock_feedback_learner), \
             patch("auditpulse_mvp.learning.scheduler.get_all_tenant_ids", return_value=tenant_ids):
            
            scheduler = FeedbackLearningScheduler(db_session=mock_db_session)
            
            # Act
            await scheduler.run_feedback_learning()
            
            # Assert
            assert mock_feedback_learner.process_recent_feedback.call_count == len(tenant_ids)
            for tenant_id in tenant_ids:
                mock_feedback_learner.process_recent_feedback.assert_any_call(tenant_id, days=30)
    
    async def test_run_feedback_learning_with_error(self, mock_db_session):
        """Test handling errors during feedback learning."""
        # Arrange
        tenant_ids = [uuid.uuid4(), uuid.uuid4()]
        
        mock_learner = AsyncMock()
        # First call succeeds, second call raises an exception
        mock_learner.process_recent_feedback.side_effect = [
            {"status": "success"},
            Exception("Test error"),
        ]
        
        with patch("auditpulse_mvp.learning.scheduler.get_feedback_learner", return_value=mock_learner), \
             patch("auditpulse_mvp.learning.scheduler.get_all_tenant_ids", return_value=tenant_ids), \
             patch("auditpulse_mvp.learning.scheduler.logger") as mock_logger:
            
            scheduler = FeedbackLearningScheduler(db_session=mock_db_session)
            
            # Act
            await scheduler.run_feedback_learning()
            
            # Assert
            assert mock_learner.process_recent_feedback.call_count == len(tenant_ids)
            mock_logger.error.assert_called_once()
    
    async def test_custom_schedule(self, mock_db_session, mock_feedback_learner):
        """Test setting custom schedule."""
        # Arrange
        mock_scheduler = MagicMock()
        custom_hour = 5
        custom_minute = 30
        
        with patch("auditpulse_mvp.learning.scheduler.get_feedback_learner", return_value=mock_feedback_learner):
            scheduler = FeedbackLearningScheduler(
                db_session=mock_db_session, hour=custom_hour, minute=custom_minute
            )
            scheduler.scheduler = mock_scheduler
            
            # Act
            await scheduler.start()
            
            # Assert
            mock_scheduler.add_job.assert_called_once()
            # Check that the custom schedule was used
            args, kwargs = mock_scheduler.add_job.call_args
            assert kwargs["hour"] == custom_hour
            assert kwargs["minute"] == custom_minute
    
    async def test_default_schedule(self, mock_db_session, mock_feedback_learner):
        """Test using default schedule."""
        # Arrange
        mock_scheduler = MagicMock()
        
        with patch("auditpulse_mvp.learning.scheduler.get_feedback_learner", return_value=mock_feedback_learner):
            scheduler = FeedbackLearningScheduler(db_session=mock_db_session)
            scheduler.scheduler = mock_scheduler
            
            # Act
            await scheduler.start()
            
            # Assert
            mock_scheduler.add_job.assert_called_once()
            # Check that the default schedule was used
            args, kwargs = mock_scheduler.add_job.call_args
            assert kwargs["hour"] == DEFAULT_SCHEDULE_HOUR
            assert kwargs["minute"] == DEFAULT_SCHEDULE_MINUTE


@pytest.mark.asyncio
async def test_get_all_tenant_ids(mock_db_session):
    """Test getting all tenant IDs."""
    # Arrange
    tenant1_id = uuid.uuid4()
    tenant2_id = uuid.uuid4()
    
    # Mock the database response
    mock_db_session.execute.return_value.scalars.return_value.all.return_value = [
        MagicMock(id=tenant1_id),
        MagicMock(id=tenant2_id),
    ]
    
    # Act
    with patch("auditpulse_mvp.learning.scheduler.select"), \
         patch("auditpulse_mvp.learning.scheduler.Tenant"):
        
        from auditpulse_mvp.learning.scheduler import get_all_tenant_ids
        result = await get_all_tenant_ids(mock_db_session)
        
        # Assert
        assert len(result) == 2
        assert tenant1_id in result
        assert tenant2_id in result
        mock_db_session.execute.assert_called_once() 