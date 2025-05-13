"""Unit tests for the notification service.

This module contains tests for the notification service and providers.
"""
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import Response

from auditpulse_mvp.alerts.base import NotificationPayload, NotificationStatus, NotificationPriority
from auditpulse_mvp.alerts.email import EmailNotificationProvider
from auditpulse_mvp.alerts.slack import SlackNotificationProvider
from auditpulse_mvp.alerts.sms import SMSNotificationProvider
from auditpulse_mvp.alerts.notification_service import NotificationService


@pytest.fixture
def mock_anomaly():
    """Create a mock anomaly for testing."""
    anomaly = MagicMock()
    anomaly.id = uuid.uuid4()
    anomaly.tenant_id = uuid.uuid4()
    anomaly.transaction_id = uuid.uuid4()
    anomaly.description = "Test anomaly description"
    anomaly.risk_score = 75.0
    anomaly.explanation = "This is a test explanation"
    anomaly.notification_sent = False
    anomaly.is_resolved = False
    anomaly.transaction = MagicMock()
    anomaly.transaction.transaction_type = "payment"
    anomaly.anomaly_type = "ml_based"
    anomaly.rule_name = "test_rule"
    return anomaly


@pytest.fixture
def mock_user():
    """Create a mock user for testing."""
    user = MagicMock()
    user.id = uuid.uuid4()
    user.tenant_id = uuid.uuid4()
    user.email = "test@example.com"
    user.email_notifications = True
    user.slack_notifications = True
    user.slack_user_id = "U12345678"
    user.sms_notifications = True
    user.phone_number = "+15551234567"
    user.role = "admin"
    user.is_active = True
    return user


@pytest.fixture
def sample_payload():
    """Create a sample notification payload for testing."""
    return NotificationPayload(
        tenant_id=uuid.uuid4(),
        anomaly_id=uuid.uuid4(),
        transaction_id=uuid.uuid4(),
        subject="Test notification",
        message="Test message",
        risk_level="high",
        risk_score=75.0,
        explanation="This is a test explanation",
        dashboard_url="/dashboard/test",
        priority=NotificationPriority.HIGH,
    )


class TestEmailNotificationProvider:
    """Tests for the email notification provider."""
    
    @pytest.mark.asyncio
    async def test_is_configured(self):
        """Test the is_configured method."""
        # Test with API key
        provider = EmailNotificationProvider(api_key="test_key")
        assert await provider.is_configured() is True
        
        # Test without API key
        provider = EmailNotificationProvider(api_key=None)
        assert await provider.is_configured() is False
    
    @pytest.mark.asyncio
    async def test_send_email_success(self, sample_payload):
        """Test sending an email successfully."""
        provider = EmailNotificationProvider(api_key="test_key")
        
        # Mock the httpx response
        mock_response = Response(200, json={"status": "ok"})
        
        with patch("httpx.AsyncClient.post", return_value=mock_response):
            result = await provider.send("test@example.com", sample_payload)
            
        assert result == NotificationStatus.SENT
    
    @pytest.mark.asyncio
    async def test_send_email_failure(self, sample_payload):
        """Test sending an email with failure."""
        provider = EmailNotificationProvider(api_key="test_key")
        
        # Mock the httpx response
        mock_response = Response(400, json={"error": "Bad request"})
        
        with patch("httpx.AsyncClient.post", return_value=mock_response):
            result = await provider.send("test@example.com", sample_payload)
            
        assert result == NotificationStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_send_email_exception(self, sample_payload):
        """Test sending an email with an exception."""
        provider = EmailNotificationProvider(api_key="test_key")
        
        with patch("httpx.AsyncClient.post", side_effect=Exception("Test error")):
            result = await provider.send("test@example.com", sample_payload)
            
        assert result == NotificationStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_send_email_invalid_recipient(self, sample_payload):
        """Test sending an email with an invalid recipient."""
        provider = EmailNotificationProvider(api_key="test_key")
        
        result = await provider.send("invalid-email", sample_payload)
            
        assert result == NotificationStatus.FAILED


class TestSlackNotificationProvider:
    """Tests for the Slack notification provider."""
    
    @pytest.mark.asyncio
    async def test_is_configured(self):
        """Test the is_configured method."""
        # Test with API key
        provider = SlackNotificationProvider(bot_token="test_token")
        assert await provider.is_configured() is True
        
        # Test without API key
        provider = SlackNotificationProvider(bot_token=None)
        assert await provider.is_configured() is False
    
    @pytest.mark.asyncio
    async def test_send_slack_success(self, sample_payload):
        """Test sending a Slack message successfully."""
        provider = SlackNotificationProvider(bot_token="test_token")
        
        # Mock the httpx response
        mock_response = Response(200, json={"ok": True})
        
        with patch("httpx.AsyncClient.post", return_value=mock_response):
            result = await provider.send("C12345678", sample_payload)
            
        assert result == NotificationStatus.SENT
    
    @pytest.mark.asyncio
    async def test_send_slack_failure(self, sample_payload):
        """Test sending a Slack message with failure."""
        provider = SlackNotificationProvider(bot_token="test_token")
        
        # Mock the httpx response
        mock_response = Response(200, json={"ok": False, "error": "invalid_channel"})
        
        with patch("httpx.AsyncClient.post", return_value=mock_response):
            result = await provider.send("invalid-channel", sample_payload)
            
        assert result == NotificationStatus.FAILED


class TestSMSNotificationProvider:
    """Tests for the SMS notification provider."""
    
    @pytest.mark.asyncio
    async def test_is_configured(self):
        """Test the is_configured method."""
        # Test with credentials
        provider = SMSNotificationProvider(
            account_sid="test_sid", 
            auth_token="test_token"
        )
        assert await provider.is_configured() is True
        
        # Test without credentials
        provider = SMSNotificationProvider(
            account_sid=None, 
            auth_token=None
        )
        assert await provider.is_configured() is False
    
    @pytest.mark.asyncio
    async def test_send_sms_success(self, sample_payload):
        """Test sending an SMS successfully."""
        provider = SMSNotificationProvider(
            account_sid="test_sid", 
            auth_token="test_token"
        )
        
        # Mock the httpx response
        mock_response = Response(201, json={"sid": "SM123456"})
        
        with patch("httpx.AsyncClient.post", return_value=mock_response):
            result = await provider.send("+15551234567", sample_payload)
            
        assert result == NotificationStatus.SENT
    
    @pytest.mark.asyncio
    async def test_send_sms_invalid_phone(self, sample_payload):
        """Test sending an SMS with an invalid phone number."""
        provider = SMSNotificationProvider(
            account_sid="test_sid", 
            auth_token="test_token"
        )
        
        result = await provider.send("invalid-phone", sample_payload)
            
        assert result == NotificationStatus.FAILED


class TestNotificationService:
    """Tests for the notification service."""
    
    @pytest.fixture
    def mock_service(self):
        """Create a mock notification service for testing."""
        email_provider = MagicMock()
        email_provider.is_configured = AsyncMock(return_value=True)
        email_provider.send = AsyncMock(return_value=NotificationStatus.SENT)
        
        slack_provider = MagicMock()
        slack_provider.is_configured = AsyncMock(return_value=True)
        slack_provider.send = AsyncMock(return_value=NotificationStatus.SENT)
        
        sms_provider = MagicMock()
        sms_provider.is_configured = AsyncMock(return_value=True)
        sms_provider.send = AsyncMock(return_value=NotificationStatus.SENT)
        
        service = NotificationService(
            email_provider=email_provider,
            slack_provider=slack_provider,
            sms_provider=sms_provider,
        )
        
        return service
    
    @pytest.mark.asyncio
    async def test_send_anomaly_notification(self, mock_service, mock_anomaly, mock_user):
        """Test sending notifications for an anomaly."""
        # Mock the database session
        db = AsyncMock()
        
        # Mock getting the anomaly
        mock_service._get_anomaly_with_related = AsyncMock(return_value=mock_anomaly)
        
        # Mock getting the notification recipients
        mock_service._get_notification_recipients = AsyncMock(return_value=[mock_user])
        
        # Mock marking the notification as sent
        mock_service._mark_notification_sent = AsyncMock()
        
        # Call the method
        result = await mock_service.send_anomaly_notification(mock_anomaly.id, db)
        
        # Verify the results
        assert result["email"] == NotificationStatus.SENT
        assert result["slack"] == NotificationStatus.SENT
        assert result["sms"] == NotificationStatus.SENT
        
        # Verify the mocks were called
        mock_service._get_anomaly_with_related.assert_called_once_with(mock_anomaly.id, db)
        mock_service._get_notification_recipients.assert_called_once_with(mock_anomaly.tenant_id, db)
        mock_service._mark_notification_sent.assert_called_once_with(mock_anomaly.id, db)
        
        # Verify the providers were called
        mock_service.email_provider.send.assert_called_once()
        mock_service.slack_provider.send.assert_called_once()
        mock_service.sms_provider.send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_anomaly_notification_already_sent(self, mock_service, mock_anomaly):
        """Test sending notifications for an anomaly that was already notified."""
        # Mock the database session
        db = AsyncMock()
        
        # Set the notification_sent flag
        mock_anomaly.notification_sent = True
        
        # Mock getting the anomaly
        mock_service._get_anomaly_with_related = AsyncMock(return_value=mock_anomaly)
        
        # Call the method
        result = await mock_service.send_anomaly_notification(mock_anomaly.id, db, force=False)
        
        # Verify the results (should be sent status even though not actually sent)
        assert result["email"] == NotificationStatus.SENT
        assert result["slack"] == NotificationStatus.SENT
        assert result["sms"] == NotificationStatus.SENT
        
        # Verify the providers were not called
        mock_service.email_provider.send.assert_not_called()
        mock_service.slack_provider.send.assert_not_called()
        mock_service.sms_provider.send.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_create_notification_payload(self, mock_service, mock_anomaly):
        """Test creating a notification payload from an anomaly."""
        # Call the method
        payload = mock_service._create_notification_payload(mock_anomaly)
        
        # Verify the payload
        assert payload.tenant_id == mock_anomaly.tenant_id
        assert payload.anomaly_id == mock_anomaly.id
        assert payload.transaction_id == mock_anomaly.transaction_id
        assert "Unusual Transaction" in payload.subject
        assert payload.risk_score == mock_anomaly.risk_score
        assert payload.explanation == mock_anomaly.explanation
        assert "/dashboard/anomalies/" in payload.dashboard_url
        
        # Verify risk level and priority
        assert payload.risk_level == "high"  # Based on risk_score of 75
        assert payload.priority == NotificationPriority.HIGH  # Based on risk_score of 75
    
    @pytest.mark.asyncio
    async def test_get_risk_level(self, mock_service):
        """Test getting risk level from score."""
        assert mock_service._get_risk_level(0) == "negligible"
        assert mock_service._get_risk_level(15) == "negligible"
        assert mock_service._get_risk_level(25) == "low"
        assert mock_service._get_risk_level(45) == "medium"
        assert mock_service._get_risk_level(65) == "high"
        assert mock_service._get_risk_level(85) == "critical"
        assert mock_service._get_risk_level(100) == "critical"
    
    @pytest.mark.asyncio
    async def test_get_priority_from_risk(self, mock_service):
        """Test getting priority from risk score."""
        assert mock_service._get_priority_from_risk(0) == NotificationPriority.LOW
        assert mock_service._get_priority_from_risk(35) == NotificationPriority.LOW
        assert mock_service._get_priority_from_risk(45) == NotificationPriority.MEDIUM
        assert mock_service._get_priority_from_risk(65) == NotificationPriority.MEDIUM
        assert mock_service._get_priority_from_risk(75) == NotificationPriority.HIGH
        assert mock_service._get_priority_from_risk(85) == NotificationPriority.HIGH
        assert mock_service._get_priority_from_risk(95) == NotificationPriority.CRITICAL
        assert mock_service._get_priority_from_risk(100) == NotificationPriority.CRITICAL 