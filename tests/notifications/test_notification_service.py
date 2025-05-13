"""Tests for the notification service.

This module contains tests for the notification service functionality,
including email and Slack notifications for anomalies and system events.
"""

import datetime
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session

from auditpulse_mvp.database.models import (
    Anomaly,
    NotificationConfig,
    RiskLevel,
    Tenant,
)
from auditpulse_mvp.notifications.notification_service import NotificationService


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
def mock_notification_config():
    """Create a mock notification configuration."""
    config = MagicMock(spec=NotificationConfig)
    config.tenant_id = uuid.uuid4()
    config.email_enabled = True
    config.slack_enabled = True
    config.email_recipients = ["test@example.com"]
    config.slack_webhook_url = "https://hooks.slack.com/services/test"
    config.notify_on_high_risk = True
    config.notify_on_medium_risk = True
    config.notify_on_low_risk = False
    return config


@pytest.fixture
def mock_anomaly():
    """Create a mock anomaly."""
    anomaly = MagicMock(spec=Anomaly)
    anomaly.id = uuid.uuid4()
    anomaly.tenant_id = uuid.uuid4()
    anomaly.transaction_id = uuid.uuid4()
    anomaly.risk_level = RiskLevel.HIGH
    anomaly.anomaly_type = "unusual_amount"
    anomaly.score = 0.95
    anomaly.details = {
        "amount": 1000.0,
        "date": datetime.datetime.now().isoformat(),
        "description": "Test transaction",
    }
    anomaly.explanation = "This is a test anomaly"
    return anomaly


@pytest.mark.asyncio
async def test_send_anomaly_notification_high_risk(
    mock_db_session,
    mock_tenant,
    mock_notification_config,
    mock_anomaly,
):
    """Test sending notification for high-risk anomaly."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.first.return_value = (
        mock_notification_config
    )

    with patch(
        "auditpulse_mvp.notifications.notification_service.NotificationService._send_email"
    ) as mock_send_email, patch(
        "auditpulse_mvp.notifications.notification_service.NotificationService._send_slack"
    ) as mock_send_slack:

        # Initialize service
        service = NotificationService(mock_db_session)

        # Send notification
        await service.send_anomaly_notification(mock_anomaly)

        # Verify email was sent
        mock_send_email.assert_called_once()
        call_args = mock_send_email.call_args[1]
        assert call_args["recipients"] == mock_notification_config.email_recipients
        assert "High Risk Anomaly Detected" in call_args["subject"]

        # Verify Slack message was sent
        mock_send_slack.assert_called_once()
        call_args = mock_send_slack.call_args[1]
        assert call_args["webhook_url"] == mock_notification_config.slack_webhook_url
        assert "High Risk Anomaly" in call_args["message"]


@pytest.mark.asyncio
async def test_send_anomaly_notification_low_risk(
    mock_db_session,
    mock_tenant,
    mock_notification_config,
    mock_anomaly,
):
    """Test sending notification for low-risk anomaly (should not send)."""
    # Setup
    mock_notification_config.notify_on_low_risk = False
    mock_db_session.query.return_value.filter.return_value.first.return_value = (
        mock_notification_config
    )
    mock_anomaly.risk_level = RiskLevel.LOW

    with patch(
        "auditpulse_mvp.notifications.notification_service.NotificationService._send_email"
    ) as mock_send_email, patch(
        "auditpulse_mvp.notifications.notification_service.NotificationService._send_slack"
    ) as mock_send_slack:

        # Initialize service
        service = NotificationService(mock_db_session)

        # Send notification
        await service.send_anomaly_notification(mock_anomaly)

        # Verify no notifications were sent
        mock_send_email.assert_not_called()
        mock_send_slack.assert_not_called()


@pytest.mark.asyncio
async def test_send_system_notification(
    mock_db_session,
    mock_tenant,
    mock_notification_config,
):
    """Test sending system notification."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.first.return_value = (
        mock_notification_config
    )

    with patch(
        "auditpulse_mvp.notifications.notification_service.NotificationService._send_email"
    ) as mock_send_email, patch(
        "auditpulse_mvp.notifications.notification_service.NotificationService._send_slack"
    ) as mock_send_slack:

        # Initialize service
        service = NotificationService(mock_db_session)

        # Send notification
        await service.send_system_notification(
            tenant_id=mock_tenant.id,
            event_type="error",
            message="Test system error",
            details={"error_code": "TEST_ERROR"},
        )

        # Verify email was sent
        mock_send_email.assert_called_once()
        call_args = mock_send_email.call_args[1]
        assert call_args["recipients"] == mock_notification_config.email_recipients
        assert "System Error" in call_args["subject"]

        # Verify Slack message was sent
        mock_send_slack.assert_called_once()
        call_args = mock_send_slack.call_args[1]
        assert call_args["webhook_url"] == mock_notification_config.slack_webhook_url
        assert "System Error" in call_args["message"]


@pytest.mark.asyncio
async def test_get_notification_config(
    mock_db_session,
    mock_tenant,
    mock_notification_config,
):
    """Test retrieving notification configuration."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.first.return_value = (
        mock_notification_config
    )

    # Initialize service
    service = NotificationService(mock_db_session)

    # Get configuration
    config = await service.get_notification_config(mock_tenant.id)

    # Verify configuration
    assert config.tenant_id == mock_notification_config.tenant_id
    assert config.email_enabled == mock_notification_config.email_enabled
    assert config.slack_enabled == mock_notification_config.slack_enabled
    assert config.email_recipients == mock_notification_config.email_recipients
    assert config.slack_webhook_url == mock_notification_config.slack_webhook_url


@pytest.mark.asyncio
async def test_get_notification_config_not_found(
    mock_db_session,
    mock_tenant,
):
    """Test retrieving non-existent notification configuration."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.first.return_value = None

    # Initialize service
    service = NotificationService(mock_db_session)

    # Get configuration
    config = await service.get_notification_config(mock_tenant.id)

    # Verify default configuration
    assert config.tenant_id == mock_tenant.id
    assert config.email_enabled is False
    assert config.slack_enabled is False
    assert config.email_recipients == []
    assert config.slack_webhook_url is None
