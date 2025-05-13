"""Tests for notification API endpoints."""

import uuid
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from auditpulse_mvp.main import app
from auditpulse_mvp.api.deps import get_current_user, get_db
from auditpulse_mvp.notifications.notification_manager import (
    NotificationManager,
    NotificationPriority,
)
from auditpulse_mvp.tasks.task_manager import TaskManager
from auditpulse_mvp.database.models.user import User
from auditpulse_mvp.database.models.notification import (
    Notification,
    NotificationTemplate,
)


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_current_user():
    """Mock the current user."""
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.email = "test@example.com"
    user.full_name = "Test User"
    user.is_active = True
    user.is_superuser = False
    user.has_permission = MagicMock(return_value=True)
    user.get_channels_for_notification_type = MagicMock(return_value=["email"])
    user.notification_preferences = {
        "channels": {
            "anomaly_detection": ["email", "slack"],
            "system_alert": ["email"],
        }
    }
    return user


@pytest.fixture
def mock_notification_manager():
    """Mock the notification manager."""
    manager = AsyncMock(spec=NotificationManager)
    manager.send_notification = AsyncMock()
    manager.send_batch_notification = AsyncMock()
    manager.get_notification_status = AsyncMock()
    return manager


@pytest.fixture
def mock_notification():
    """Mock a notification."""
    notification = MagicMock(spec=Notification)
    notification.id = uuid.uuid4()
    notification.template_id = "test_template"
    notification.user_id = uuid.uuid4()
    notification.status = "delivered"
    notification.priority = "high"
    notification.created_at = datetime.utcnow()
    notification.updated_at = datetime.utcnow()
    notification.processed_at = datetime.utcnow()
    notification.scheduled_at = None
    notification.recipient = {"email": "test@example.com"}
    notification.template_data = {"name": "Test User"}
    return notification


@pytest.fixture
def mock_template():
    """Mock a notification template."""
    template = MagicMock(spec=NotificationTemplate)
    template.template_id = "test_template"
    template.name = "Test Template"
    template.subject = "Test Subject"
    template.body = "Hello, {{ name }}!"
    template.html_body = "<p>Hello, {{ name }}!</p>"
    template.description = "A test template"
    template.placeholders = {"name": "User name"}
    template.version = 1
    template.created_at = datetime.utcnow()
    template.updated_at = datetime.utcnow()
    return template


def override_get_current_user():
    """Override the get_current_user dependency."""
    return mock_current_user()


def override_get_db():
    """Override the get_db dependency."""

    async def _get_db():
        yield mock_db_session()

    return _get_db()


def mock_db_session():
    """Mock the database session."""
    db = AsyncMock()
    db.add = AsyncMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.query = MagicMock()
    return db


def override_get_notification_manager():
    """Override the notification manager dependency."""
    return mock_notification_manager()


# Apply the overrides
app.dependency_overrides[get_current_user] = override_get_current_user
app.dependency_overrides[get_db] = override_get_db


# Test cases
def test_create_template(client, mock_current_user, monkeypatch):
    """Test creating a notification template."""
    # Mock template manager
    mock_template_manager = AsyncMock()
    mock_template_manager.create_template = AsyncMock(
        return_value={
            "template_id": "test_template",
            "name": "Test Template",
            "subject": "Test Subject",
            "body": "Hello, {{ name }}!",
            "html_body": "<p>Hello, {{ name }}!</p>",
            "description": "A test template",
            "placeholders": {"name": "User name"},
            "version": 1,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    )

    # Apply patch
    with patch(
        "auditpulse_mvp.notifications.templates.TemplateManager",
        return_value=mock_template_manager,
    ):
        # Send request
        response = client.post(
            "/api/v1/notifications/templates",
            json={
                "template_id": "test_template",
                "name": "Test Template",
                "subject": "Test Subject",
                "body": "Hello, {{ name }}!",
                "html_body": "<p>Hello, {{ name }}!</p>",
                "description": "A test template",
                "placeholders": {"name": "User name"},
            },
        )

        # Check response
        assert response.status_code == 201
        assert response.json()["template_id"] == "test_template"

        # Verify template creation
        assert mock_template_manager.create_template.called


def test_list_templates(client, mock_current_user, monkeypatch):
    """Test listing notification templates."""
    # Mock template manager
    mock_template_manager = AsyncMock()
    mock_template_manager.list_templates = AsyncMock(
        return_value=[
            {
                "template_id": "template1",
                "name": "Template 1",
                "subject": "Subject 1",
                "description": "Description 1",
                "version": 1,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            },
            {
                "template_id": "template2",
                "name": "Template 2",
                "subject": "Subject 2",
                "description": "Description 2",
                "version": 1,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            },
        ]
    )

    # Apply patch
    with patch(
        "auditpulse_mvp.notifications.templates.TemplateManager",
        return_value=mock_template_manager,
    ):
        # Send request
        response = client.get("/api/v1/notifications/templates")

        # Check response
        assert response.status_code == 200
        templates = response.json()
        assert len(templates) == 2
        assert templates[0]["template_id"] == "template1"
        assert templates[1]["template_id"] == "template2"

        # Verify template listing
        assert mock_template_manager.list_templates.called


def test_create_notification(client, mock_current_user, mock_notification, monkeypatch):
    """Test creating a notification."""
    # Mock template manager and notification manager
    mock_template_manager = AsyncMock()
    mock_template_manager.get_template = AsyncMock(
        return_value={
            "template_id": "test_template",
            "name": "Test Template",
            "subject": "Test Subject",
            "body": "Hello, {{ name }}!",
        }
    )

    mock_manager = AsyncMock(spec=NotificationManager)
    mock_manager.send_notification = AsyncMock(
        return_value={
            "notification_id": str(mock_notification.id),
            "status": "pending",
        }
    )

    # Mock database query to get the notification
    mock_db = AsyncMock()
    mock_db.query = MagicMock()
    mock_db.query.return_value.filter.return_value.first = AsyncMock(
        return_value=mock_notification
    )

    # Apply patches
    with patch(
        "auditpulse_mvp.notifications.templates.TemplateManager",
        return_value=mock_template_manager,
    ), patch(
        "auditpulse_mvp.notifications.notification_manager.NotificationManager",
        return_value=mock_manager,
    ), patch(
        "auditpulse_mvp.database.session.get_db", AsyncMock(return_value=mock_db)
    ), patch(
        "auditpulse_mvp.tasks.task_manager.TaskManager.get_instance",
        MagicMock(return_value=AsyncMock()),
    ):

        # Send request
        response = client.post(
            "/api/v1/notifications/notifications",
            json={
                "template_id": "test_template",
                "recipient": {"email": "test@example.com", "channels": ["email"]},
                "template_data": {"name": "Test User"},
                "priority": "HIGH",
            },
        )

        # Check response
        assert response.status_code == 201
        assert response.json()["template_id"] == "test_template"

        # Verify notification creation
        assert mock_manager.send_notification.called


def test_get_notification_preferences(client, mock_current_user):
    """Test getting notification preferences."""
    # Send request
    response = client.get("/api/v1/notifications/notifications/preferences")

    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "notification_preferences" in data
    assert "default_channels" in data
    assert (
        data["notification_preferences"] == mock_current_user.notification_preferences
    )


def test_update_notification_preferences(client, mock_current_user, monkeypatch):
    """Test updating notification preferences."""
    # Mock DB
    mock_db = AsyncMock()
    mock_db.commit = AsyncMock()

    # Apply patch
    with patch(
        "auditpulse_mvp.database.session.get_db", AsyncMock(return_value=mock_db)
    ):
        # Send request
        response = client.put(
            "/api/v1/notifications/notifications/preferences",
            json={
                "channels": {
                    "anomaly_detection": ["email", "slack"],
                    "system_alert": ["email", "sms"],
                    "model_performance": ["email"],
                    "account_security": ["email", "sms"],
                    "scheduled_reports": ["email"],
                }
            },
        )

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "notification_preferences" in data

        # Verify preferences update
        assert mock_db.commit.called
