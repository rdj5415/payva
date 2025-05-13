"""Tests for the task manager."""

import asyncio
from datetime import datetime, timedelta
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from auditpulse_mvp.database.models import TaskStatus
from auditpulse_mvp.tasks.task_manager import TaskManager, TaskPriority

@pytest.fixture
async def task_manager(db_session):
    """Create a task manager instance."""
    manager = TaskManager(db_session)
    await manager.start()
    yield manager
    await manager.stop()

@pytest.fixture
def mock_task():
    """Create a mock task handler."""
    async def task_handler(*args, **kwargs):
        return {"result": "success"}
    return task_handler

@pytest.mark.asyncio
async def test_register_task(task_manager, mock_task):
    """Test task registration."""
    # Register task
    task_manager.register_task("test_task", mock_task)
    
    # Verify task is registered
    assert "test_task" in task_manager._task_handlers
    assert task_manager._task_handlers["test_task"]["handler"] == mock_task

@pytest.mark.asyncio
async def test_schedule_task(task_manager, mock_task):
    """Test task scheduling."""
    # Register task
    task_manager.register_task("test_task", mock_task)
    
    # Schedule task
    task_id = await task_manager.schedule_task(
        task_name="test_task",
        priority=TaskPriority.HIGH,
        args=["arg1"],
        kwargs={"kwarg1": "value1"},
    )
    
    # Verify task is scheduled
    assert task_id is not None
    
    # Get task status
    task_info = await task_manager.get_task_status(task_id)
    assert task_info is not None
    assert task_info["name"] == "test_task"
    assert task_info["status"] == TaskStatus.PENDING
    assert task_info["priority"] == TaskPriority.HIGH.name

@pytest.mark.asyncio
async def test_task_execution(task_manager, mock_task):
    """Test task execution."""
    # Register task
    task_manager.register_task("test_task", mock_task)
    
    # Schedule task
    task_id = await task_manager.schedule_task("test_task")
    
    # Wait for task to complete
    await asyncio.sleep(1)
    
    # Get task status
    task_info = await task_manager.get_task_status(task_id)
    assert task_info["status"] == TaskStatus.COMPLETED
    assert task_info["result"] == {"result": "success"}

@pytest.mark.asyncio
async def test_task_retry(task_manager):
    """Test task retry on failure."""
    # Create failing task
    async def failing_task():
        raise ValueError("Task failed")
    
    # Register task
    task_manager.register_task("failing_task", failing_task, max_retries=2)
    
    # Schedule task
    task_id = await task_manager.schedule_task("failing_task")
    
    # Wait for retries
    await asyncio.sleep(5)
    
    # Get task status
    task_info = await task_manager.get_task_status(task_id)
    assert task_info["status"] == TaskStatus.FAILED
    assert task_info["retry_count"] == 2
    assert "Task failed" in task_info["error"]

@pytest.mark.asyncio
async def test_task_stats(task_manager, mock_task):
    """Test task statistics."""
    # Register task
    task_manager.register_task("test_task", mock_task)
    
    # Schedule multiple tasks
    for _ in range(3):
        await task_manager.schedule_task("test_task")
    
    # Wait for tasks to complete
    await asyncio.sleep(1)
    
    # Get stats
    stats = await task_manager.get_task_stats()
    assert stats["total_tasks"] == 3
    assert stats["status_counts"][TaskStatus.COMPLETED] == 3
    assert stats["priority_counts"][TaskPriority.MEDIUM.name] == 3

@pytest.mark.asyncio
async def test_scheduled_task(task_manager, mock_task):
    """Test scheduled task execution."""
    # Register task
    task_manager.register_task("test_task", mock_task)
    
    # Schedule task to run in 1 second
    run_at = datetime.now() + timedelta(seconds=1)
    task_id = await task_manager.schedule_task(
        task_name="test_task",
        run_at=run_at,
    )
    
    # Verify task is scheduled
    task_info = await task_manager.get_task_status(task_id)
    assert task_info["status"] == TaskStatus.PENDING
    
    # Wait for task to run
    await asyncio.sleep(2)
    
    # Verify task completed
    task_info = await task_manager.get_task_status(task_id)
    assert task_info["status"] == TaskStatus.COMPLETED

@pytest.mark.asyncio
async def test_interval_task(task_manager, mock_task):
    """Test interval task execution."""
    # Register task
    task_manager.register_task("test_task", mock_task)
    
    # Schedule task to run every second
    task_id = await task_manager.schedule_task(
        task_name="test_task",
        interval=timedelta(seconds=1),
    )
    
    # Wait for multiple executions
    await asyncio.sleep(3)
    
    # Get stats
    stats = await task_manager.get_task_stats()
    assert stats["status_counts"][TaskStatus.COMPLETED] >= 2

@pytest.mark.asyncio
async def test_cron_task(task_manager, mock_task):
    """Test cron task execution."""
    # Register task
    task_manager.register_task("test_task", mock_task)
    
    # Schedule task to run every minute
    task_id = await task_manager.schedule_task(
        task_name="test_task",
        cron="* * * * *",
    )
    
    # Wait for execution
    await asyncio.sleep(2)
    
    # Get stats
    stats = await task_manager.get_task_stats()
    assert stats["status_counts"][TaskStatus.COMPLETED] >= 1 