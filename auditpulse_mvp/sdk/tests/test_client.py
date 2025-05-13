"""Tests for the AuditPulse SDK client.

This module contains tests for the SDK client functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from auditpulse_mvp.sdk import (
    AuditPulseClient,
    AuditPulseError,
    Anomaly,
    AnomalyStatus,
    AnomalyType,
    Feedback,
    FeedbackType,
    Notification,
    NotificationChannel,
    NotificationType,
    PaginatedResponse,
    RiskLevel,
    Transaction,
    User,
)


@pytest.fixture
def client():
    """Create a test client."""
    return AuditPulseClient(
        base_url="https://api.auditpulse.ai", api_key="test_api_key"
    )


@pytest.fixture
def mock_session():
    """Create a mock aiohttp session."""
    session = AsyncMock()
    response = AsyncMock()
    response.json.return_value = {
        "access_token": "test_token",
        "refresh_token": "test_refresh_token",
    }
    response.ok = True
    session.request.return_value.__aenter__.return_value = response
    return session


@pytest.mark.asyncio
async def test_login(client, mock_session):
    """Test login functionality."""
    with patch("aiohttp.ClientSession", return_value=mock_session):
        async with client:
            result = await client.login("test@example.com", "password")
            assert result["access_token"] == "test_token"
            assert result["refresh_token"] == "test_refresh_token"
            assert client.token == "test_token"


@pytest.mark.asyncio
async def test_refresh_token(client, mock_session):
    """Test token refresh functionality."""
    with patch("aiohttp.ClientSession", return_value=mock_session):
        async with client:
            result = await client.refresh_token("test_refresh_token")
            assert result["access_token"] == "test_token"
            assert result["refresh_token"] == "test_refresh_token"
            assert client.token == "test_token"


@pytest.mark.asyncio
async def test_get_transactions(client, mock_session):
    """Test getting transactions."""
    mock_session.request.return_value.__aenter__.return_value.json.return_value = {
        "items": [
            {
                "id": str(uuid4()),
                "tenant_id": str(uuid4()),
                "transaction_id": "txn_001",
                "amount": 1000.0,
                "currency": "USD",
                "description": "Test transaction",
                "category": "Test",
                "merchant_name": "Test Merchant",
                "transaction_date": datetime.now().isoformat(),
                "source": "test",
                "source_account_id": "acc_001",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
        ],
        "total": 1,
        "page": 1,
        "size": 100,
        "pages": 1,
    }

    with patch("aiohttp.ClientSession", return_value=mock_session):
        async with client:
            result = await client.get_transactions(
                tenant_id=str(uuid4()), start_date=datetime.now() - timedelta(days=30)
            )
            assert isinstance(result, PaginatedResponse)
            assert len(result.items) == 1
            assert isinstance(result.items[0], Transaction)


@pytest.mark.asyncio
async def test_get_anomalies(client, mock_session):
    """Test getting anomalies."""
    mock_session.request.return_value.__aenter__.return_value.json.return_value = {
        "items": [
            {
                "id": str(uuid4()),
                "tenant_id": str(uuid4()),
                "transaction_id": "txn_001",
                "type": "large_amount",
                "risk_score": 0.8,
                "risk_level": "high",
                "amount": 1000.0,
                "description": "Test anomaly",
                "status": "open",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
        ],
        "total": 1,
        "page": 1,
        "size": 100,
        "pages": 1,
    }

    with patch("aiohttp.ClientSession", return_value=mock_session):
        async with client:
            result = await client.get_anomalies(tenant_id=str(uuid4()), status="open")
            assert isinstance(result, PaginatedResponse)
            assert len(result.items) == 1
            assert isinstance(result.items[0], Anomaly)


@pytest.mark.asyncio
async def test_update_anomaly(client, mock_session):
    """Test updating an anomaly."""
    mock_session.request.return_value.__aenter__.return_value.json.return_value = {
        "id": str(uuid4()),
        "tenant_id": str(uuid4()),
        "transaction_id": "txn_001",
        "type": "large_amount",
        "risk_score": 0.8,
        "risk_level": "high",
        "amount": 1000.0,
        "description": "Test anomaly",
        "status": "resolved",
        "resolution": "False positive",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    with patch("aiohttp.ClientSession", return_value=mock_session):
        async with client:
            result = await client.update_anomaly(
                anomaly_id=str(uuid4()), status="resolved", resolution="False positive"
            )
            assert isinstance(result, Anomaly)
            assert result.status == AnomalyStatus.RESOLVED
            assert result.resolution == "False positive"


@pytest.mark.asyncio
async def test_submit_feedback(client, mock_session):
    """Test submitting feedback."""
    mock_session.request.return_value.__aenter__.return_value.json.return_value = {
        "type": "false_positive",
        "comment": "Test feedback",
        "rating": 5,
        "created_at": datetime.now().isoformat(),
    }

    with patch("aiohttp.ClientSession", return_value=mock_session):
        async with client:
            result = await client.submit_feedback(
                anomaly_id=str(uuid4()),
                feedback={
                    "type": "false_positive",
                    "comment": "Test feedback",
                    "rating": 5,
                },
            )
            assert isinstance(result, Feedback)
            assert result.type == FeedbackType.FALSE_POSITIVE
            assert result.comment == "Test feedback"
            assert result.rating == 5


@pytest.mark.asyncio
async def test_error_handling(client, mock_session):
    """Test error handling."""
    mock_session.request.return_value.__aenter__.return_value.ok = False
    mock_session.request.return_value.__aenter__.return_value.json.return_value = {
        "code": "not_found",
        "message": "Resource not found",
        "details": {"resource": "transaction"},
    }

    with patch("aiohttp.ClientSession", return_value=mock_session):
        async with client:
            with pytest.raises(AuditPulseError) as exc_info:
                await client.get_transactions(tenant_id="invalid_id")
            assert exc_info.value.code == "not_found"
            assert exc_info.value.message == "Resource not found"
            assert exc_info.value.details == {"resource": "transaction"}
