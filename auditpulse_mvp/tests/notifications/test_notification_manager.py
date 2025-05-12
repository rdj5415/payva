"""Tests for the notification manager."""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from auditpulse_mvp.notifications.notification_manager import (
    NotificationManager,
    NotificationRequest,
    NotificationRecipient,
    NotificationPriority,
)
from auditpulse_mvp.database.models.notification import Notification
from auditpulse_mvp.tasks.task_manager import TaskManager
from auditpulse_mvp.utils.settings import Settings

@pytest.fixture
def mock_db_session():
    """Mock database session."""
    db = AsyncMock()
    db.add = AsyncMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.query = MagicMock()
    db.query.return_value.filter.return_value.first = AsyncMock()
    db.query.return_value.filter.return_value.all = AsyncMock()
    return db

@pytest.fixture
def mock_task_manager():
    """Mock task manager."""
    task_manager = AsyncMock(spec=TaskManager)
    task_manager.schedule_task = AsyncMock()
    task_manager.get_task_status = AsyncMock()
    return task_manager

@pytest.fixture
def mock_template_manager():
    """Mock template manager."""
    template_manager = AsyncMock()
    template_manager.get_template = AsyncMock()
    template_manager.render_template = AsyncMock()
    return template_manager

@pytest.fixture
def notification_manager(mock_task_manager):
    """Create a notification manager."""
    settings = Settings()
    return NotificationManager(settings, mock_task_manager)

@pytest.mark.asyncio
async def test_send_notification(notification_manager, mock_db_session, mock_task_manager, monkeypatch):
    """Test sending a notification."""
    # Setup
    monkeypatch.setattr("auditpulse_mvp.database.session.get_db", AsyncMock(return_value=mock_db_session))
    
    # Mock notification created in DB
    mock_notification = MagicMock(spec=Notification)
    mock_notification.id = uuid.uuid4()
    mock_db_session.add.return_value = None
    mock_db_session.refresh.side_effect = lambda x: setattr(x, "id", mock_notification.id)
    
    # Create notification request
    recipient = NotificationRecipient(email="test@example.com")
    request = NotificationRequest(
        template_id="test_template",
        recipient=recipient,
        template_data={"name": "Test User"},
        priority=NotificationPriority.HIGH,
        channels=["email"],
    )
    
    # Send notification
    result = await notification_manager.send_notification(
        request=request,
        user_id=uuid.uuid4(),
    )
    
    # Verify
    assert result is not None
    assert "notification_id" in result
    assert result["status"] == "pending"
    assert mock_db_session.add.called
    assert mock_db_session.commit.called
    assert mock_task_manager.schedule_task.called
    
    # Verify task scheduling
    task_name, task_params, task_options = mock_task_manager.schedule_task.call_args[0]
    assert task_name == "process_notification"
    assert task_params["notification_id"] == str(mock_notification.id)

@pytest.mark.asyncio
async def test_send_notification_with_schedule(notification_manager, mock_db_session, mock_task_manager, monkeypatch):
    """Test sending a scheduled notification."""
    # Setup
    monkeypatch.setattr("auditpulse_mvp.database.session.get_db", AsyncMock(return_value=mock_db_session))
    
    # Mock notification created in DB
    mock_notification = MagicMock(spec=Notification)
    mock_notification.id = uuid.uuid4()
    mock_db_session.add.return_value = None
    mock_db_session.refresh.side_effect = lambda x: setattr(x, "id", mock_notification.id)
    
    # Create notification request
    recipient = NotificationRecipient(email="test@example.com")
    request = NotificationRequest(
        template_id="test_template",
        recipient=recipient,
        template_data={"name": "Test User"},
        priority=NotificationPriority.MEDIUM,
    )
    
    # Schedule notification for future
    scheduled_time = datetime.utcnow() + timedelta(hours=1)
    
    # Send notification
    result = await notification_manager.send_notification(
        request=request,
        user_id=uuid.uuid4(),
        scheduled_at=scheduled_time,
    )
    
    # Verify
    assert result is not None
    assert "notification_id" in result
    assert result["status"] == "scheduled"
    assert mock_db_session.add.called
    assert mock_db_session.commit.called
    assert mock_task_manager.schedule_task.called
    
    # Verify task scheduling with scheduled time
    task_name, task_params, task_options = mock_task_manager.schedule_task.call_args[0]
    assert task_name == "process_notification"
    assert task_options.get("scheduled_at") == scheduled_time

@pytest.mark.asyncio
async def test_send_batch_notification(notification_manager, mock_db_session, mock_task_manager, monkeypatch):
    """Test sending batch notifications."""
    # Setup
    monkeypatch.setattr("auditpulse_mvp.database.session.get_db", AsyncMock(return_value=mock_db_session))
    
    # Mock notifications created in DB
    mock_notifications = [MagicMock(spec=Notification) for _ in range(3)]
    for i, notification in enumerate(mock_notifications):
        notification.id = uuid.uuid4()
        
    # Set up mock behavior for db.add and db.refresh
    notification_index = 0
    
    def mock_refresh(notification):
        nonlocal notification_index
        notification.id = mock_notifications[notification_index].id
        notification_index += 1
        
    mock_db_session.add.return_value = None
    mock_db_session.refresh.side_effect = mock_refresh
    
    # Create notification requests
    user_id = uuid.uuid4()
    notifications = []
    
    for i in range(3):
        recipient = NotificationRecipient(email=f"user{i}@example.com")
        request = NotificationRequest(
            template_id="test_template",
            recipient=recipient,
            template_data={"user_id": i},
            priority=NotificationPriority.MEDIUM,
        )
        notifications.append({
            "request": request,
            "user_id": user_id,
        })
    
    # Send batch notification
    result = await notification_manager.send_batch_notification(
        notifications=notifications,
        max_concurrent=2,
    )
    
    # Verify
    assert result is not None
    assert "status" in result
    assert result["status"] == "scheduled"
    assert "notifications" in result
    assert len(result["notifications"]) == 3
    assert mock_db_session.add.call_count == 3
    assert mock_db_session.commit.call_count == 3
    assert mock_task_manager.schedule_task.called
    
    # Verify batch task scheduling
    task_name, task_params, task_options = mock_task_manager.schedule_task.call_args[0]
    assert task_name == "process_batch_notifications"
    assert "notifications" in task_params
    assert len(task_params["notifications"]) == 3
    assert task_params["max_concurrent"] == 2

@pytest.mark.asyncio
async def test_get_notification_status(notification_manager, mock_db_session, mock_task_manager, monkeypatch):
    """Test getting notification status."""
    # Setup
    monkeypatch.setattr("auditpulse_mvp.database.session.get_db", AsyncMock(return_value=mock_db_session))
    
    # Mock notification in DB
    mock_notification = MagicMock(spec=Notification)
    mock_notification.id = uuid.uuid4()
    mock_notification.status = "delivered"
    mock_notification.created_at = datetime.utcnow()
    mock_notification.processed_at = datetime.utcnow()
    mock_notification.to_dict = MagicMock(return_value={
        "id": str(mock_notification.id),
        "status": mock_notification.status,
        "created_at": mock_notification.created_at.isoformat(),
        "processed_at": mock_notification.processed_at.isoformat(),
    })
    
    # Set up mock query result
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_notification
    
    # Mock delivery attempts
    mock_delivery_attempts = [MagicMock() for _ in range(2)]
    for i, attempt in enumerate(mock_delivery_attempts):
        attempt.id = uuid.uuid4()
        attempt.channel = "email" if i == 0 else "slack"
        attempt.status = "delivered"
        attempt.to_dict = MagicMock(return_value={
            "id": str(attempt.id),
            "channel": attempt.channel,
            "status": attempt.status,
        })
        
    mock_db_session.query.return_value.filter.return_value.all.return_value = mock_delivery_attempts
    
    # Get notification status
    result = await notification_manager.get_notification_status(str(mock_notification.id))
    
    # Verify
    assert result is not None
    assert "status" in result
    assert result["status"] == "delivered"
    assert "delivery_attempts" in result
    assert len(result["delivery_attempts"]) == 2 