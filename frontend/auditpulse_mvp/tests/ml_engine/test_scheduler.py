"""Test suite for the ML Engine scheduler module.

This module tests the scheduling of ML model retraining tasks.
"""
import asyncio
import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auditpulse_mvp.ml_engine.scheduler import MLScheduler, start_ml_scheduler, stop_ml_scheduler


@pytest.fixture
def mock_ml_engine():
    """Create a mock ML engine."""
    ml_engine = AsyncMock()
    
    # Mock successful retraining
    ml_engine.retrain_all_tenant_models.return_value = {
        "tenant1": {"status": "success", "model_path": "/path/to/model1"},
        "tenant2": {"status": "success", "model_path": "/path/to/model2"},
    }
    
    return ml_engine


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    return AsyncMock()


@pytest.mark.asyncio
class TestMLScheduler:
    """Test the MLScheduler class."""
    
    async def test_init(self, mock_db_session, mock_ml_engine):
        """Test scheduler initialization."""
        scheduler = MLScheduler(db_session=mock_db_session)
        
        # Replace ML engine with mock
        scheduler.ml_engine = mock_ml_engine
        
        assert scheduler.db_session == mock_db_session
        assert scheduler._is_running is False
        assert scheduler._retraining_task is None
    
    async def test_start_stop(self, mock_db_session, mock_ml_engine):
        """Test starting and stopping the scheduler."""
        scheduler = MLScheduler(db_session=mock_db_session)
        scheduler.ml_engine = mock_ml_engine
        
        # Patch the _schedule_daily_retraining method to prevent actual scheduling
        with patch.object(scheduler, '_schedule_daily_retraining', AsyncMock()) as mock_schedule:
            # Start the scheduler
            await scheduler.start()
            
            # Check that the scheduler is running
            assert scheduler._is_running is True
            assert scheduler._retraining_task is not None
            
            # Check that the scheduling method was called
            mock_schedule.assert_called_once()
            
            # Stop the scheduler
            await scheduler.stop()
            
            # Check that the scheduler is stopped
            assert scheduler._is_running is False
            assert scheduler._retraining_task is None
    
    async def test_start_already_running(self, mock_db_session, mock_ml_engine):
        """Test starting the scheduler when it's already running."""
        scheduler = MLScheduler(db_session=mock_db_session)
        scheduler.ml_engine = mock_ml_engine
        scheduler._is_running = True
        
        # Start the scheduler
        await scheduler.start()
        
        # Check that the scheduler state didn't change
        assert scheduler._is_running is True
        assert scheduler._retraining_task is None
    
    async def test_stop_not_running(self, mock_db_session, mock_ml_engine):
        """Test stopping the scheduler when it's not running."""
        scheduler = MLScheduler(db_session=mock_db_session)
        scheduler.ml_engine = mock_ml_engine
        scheduler._is_running = False
        
        # Stop the scheduler
        await scheduler.stop()
        
        # Check that the scheduler state didn't change
        assert scheduler._is_running is False
        assert scheduler._retraining_task is None
    
    async def test_run_retraining(self, mock_db_session, mock_ml_engine):
        """Test running the retraining process."""
        scheduler = MLScheduler(db_session=mock_db_session)
        scheduler.ml_engine = mock_ml_engine
        
        # Run the retraining
        results = await scheduler._run_retraining()
        
        # Check that the ML engine was called
        mock_ml_engine.retrain_all_tenant_models.assert_called_once()
        
        # Check the results
        assert results == mock_ml_engine.retrain_all_tenant_models.return_value
    
    async def test_run_retraining_error(self, mock_db_session, mock_ml_engine):
        """Test handling errors during retraining."""
        scheduler = MLScheduler(db_session=mock_db_session)
        scheduler.ml_engine = mock_ml_engine
        
        # Set up the ML engine to raise an exception
        mock_ml_engine.retrain_all_tenant_models.side_effect = Exception("Test error")
        
        # Run the retraining
        results = await scheduler._run_retraining()
        
        # Check that the error was caught and a result was returned
        assert results["status"] == "error"
        assert "Test error" in results["error"]
    
    async def test_run_immediate_retraining(self, mock_db_session, mock_ml_engine):
        """Test running immediate retraining."""
        scheduler = MLScheduler(db_session=mock_db_session)
        scheduler.ml_engine = mock_ml_engine
        
        # Patch the _run_retraining method
        with patch.object(scheduler, '_run_retraining', AsyncMock()) as mock_run:
            mock_run.return_value = {"status": "success"}
            
            # Run immediate retraining
            results = await scheduler.run_immediate_retraining()
            
            # Check that the retraining method was called
            mock_run.assert_called_once()
            
            # Check the results
            assert results == mock_run.return_value
    
    async def test_schedule_daily_retraining(self, mock_db_session, mock_ml_engine):
        """Test scheduling daily retraining."""
        scheduler = MLScheduler(db_session=mock_db_session)
        scheduler.ml_engine = mock_ml_engine
        scheduler._is_running = True
        
        # Patch asyncio.sleep to avoid waiting
        with patch('asyncio.sleep', AsyncMock()) as mock_sleep:
            # Patch datetime.datetime.now to return a predictable time
            current_time = datetime.datetime(2023, 1, 1, 12, 0, 0)  # Noon
            scheduled_time = datetime.datetime(2023, 1, 1, 2, 0, 0)  # 2 AM tomorrow
            
            with patch('datetime.datetime') as mock_datetime:
                mock_datetime.now.return_value = current_time
                mock_datetime.combine.return_value = scheduled_time
                
                # Patch the _run_retraining method
                with patch.object(scheduler, '_run_retraining', AsyncMock()) as mock_run:
                    # Create a task to run the scheduling
                    task = asyncio.create_task(scheduler._schedule_daily_retraining())
                    
                    # Wait a bit for the task to execute
                    await asyncio.sleep(0.1)
                    
                    # Stop the scheduler to cancel the task
                    scheduler._is_running = False
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    
                    # Check that sleep was called with the expected duration
                    # (should be calculated based on the difference between now and the scheduled time)
                    mock_sleep.assert_called_once()
                    
                    # Check that retraining wasn't called yet (would be called after sleep)
                    mock_run.assert_not_called()


@pytest.mark.asyncio
class TestSchedulerGlobalFunctions:
    """Test the global scheduler functions."""
    
    async def test_start_ml_scheduler(self):
        """Test starting the global ML scheduler."""
        # Patch the MLScheduler class
        with patch('auditpulse_mvp.ml_engine.scheduler.MLScheduler') as mock_scheduler_class:
            # Create a mock scheduler instance
            mock_scheduler = AsyncMock()
            mock_scheduler_class.return_value = mock_scheduler
            
            # Patch the settings
            with patch('auditpulse_mvp.ml_engine.scheduler.settings') as mock_settings:
                mock_settings.enable_ml_scheduler = True
                
                # Start the scheduler
                await start_ml_scheduler()
                
                # Check that the scheduler was started
                mock_scheduler.start.assert_called_once()
    
    async def test_stop_ml_scheduler(self):
        """Test stopping the global ML scheduler."""
        # Patch the global scheduler instance
        mock_scheduler = AsyncMock()
        with patch('auditpulse_mvp.ml_engine.scheduler._ml_scheduler', mock_scheduler):
            # Stop the scheduler
            await stop_ml_scheduler()
            
            # Check that the scheduler was stopped
            mock_scheduler.stop.assert_called_once()
    
    async def test_stop_ml_scheduler_not_initialized(self):
        """Test stopping the global ML scheduler when it's not initialized."""
        # Patch the global scheduler instance to be None
        with patch('auditpulse_mvp.ml_engine.scheduler._ml_scheduler', None):
            # Stop the scheduler (should not raise an exception)
            await stop_ml_scheduler() 