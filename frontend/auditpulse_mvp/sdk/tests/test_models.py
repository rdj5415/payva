"""Tests for the AuditPulse SDK models.

This module contains tests for the SDK data models.
"""
import pytest
from datetime import datetime
from uuid import uuid4

from auditpulse_mvp.sdk import (
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


def test_transaction_model():
    """Test Transaction model."""
    transaction = Transaction(
        id=uuid4(),
        tenant_id=uuid4(),
        transaction_id="txn_001",
        amount=1000.0,
        currency="USD",
        description="Test transaction",
        category="Test",
        merchant_name="Test Merchant",
        transaction_date=datetime.now(),
        source="test",
        source_account_id="acc_001",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    assert transaction.amount == 1000.0
    assert transaction.currency == "USD"
    assert transaction.description == "Test transaction"
    assert transaction.category == "Test"
    assert transaction.merchant_name == "Test Merchant"
    assert transaction.source == "test"
    assert transaction.source_account_id == "acc_001"


def test_anomaly_model():
    """Test Anomaly model."""
    anomaly = Anomaly(
        id=uuid4(),
        tenant_id=uuid4(),
        transaction_id="txn_001",
        type=AnomalyType.LARGE_AMOUNT,
        risk_score=0.8,
        risk_level=RiskLevel.HIGH,
        amount=1000.0,
        description="Test anomaly",
        status=AnomalyStatus.OPEN,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    assert anomaly.type == AnomalyType.LARGE_AMOUNT
    assert anomaly.risk_score == 0.8
    assert anomaly.risk_level == RiskLevel.HIGH
    assert anomaly.amount == 1000.0
    assert anomaly.description == "Test anomaly"
    assert anomaly.status == AnomalyStatus.OPEN


def test_feedback_model():
    """Test Feedback model."""
    feedback = Feedback(
        type=FeedbackType.FALSE_POSITIVE,
        comment="Test feedback",
        rating=5,
        created_at=datetime.now(),
    )

    assert feedback.type == FeedbackType.FALSE_POSITIVE
    assert feedback.comment == "Test feedback"
    assert feedback.rating == 5


def test_notification_model():
    """Test Notification model."""
    notification = Notification(
        id=uuid4(),
        tenant_id=uuid4(),
        type=NotificationType.ANOMALY_DETECTED,
        channel=NotificationChannel.EMAIL,
        title="Test notification",
        message="Test message",
        data={"key": "value"},
        is_read=False,
        created_at=datetime.now(),
    )

    assert notification.type == NotificationType.ANOMALY_DETECTED
    assert notification.channel == NotificationChannel.EMAIL
    assert notification.title == "Test notification"
    assert notification.message == "Test message"
    assert notification.data == {"key": "value"}
    assert not notification.is_read


def test_paginated_response_model():
    """Test PaginatedResponse model."""
    transactions = [
        Transaction(
            id=uuid4(),
            tenant_id=uuid4(),
            transaction_id="txn_001",
            amount=1000.0,
            currency="USD",
            description="Test transaction",
            category="Test",
            merchant_name="Test Merchant",
            transaction_date=datetime.now(),
            source="test",
            source_account_id="acc_001",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
    ]

    response = PaginatedResponse[Transaction](
        items=transactions,
        total=1,
        page=1,
        size=100,
        pages=1,
    )

    assert len(response.items) == 1
    assert response.total == 1
    assert response.page == 1
    assert response.size == 100
    assert response.pages == 1
    assert isinstance(response.items[0], Transaction)


def test_user_model():
    """Test User model."""
    user = User(
        id=uuid4(),
        email="test@example.com",
        first_name="Test",
        last_name="User",
        role="admin",
        is_active=True,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    assert user.email == "test@example.com"
    assert user.first_name == "Test"
    assert user.last_name == "User"
    assert user.role == "admin"
    assert user.is_active


def test_model_validation():
    """Test model validation."""
    with pytest.raises(ValueError):
        Feedback(
            type=FeedbackType.FALSE_POSITIVE,
            comment="Test feedback",
            rating=6,  # Invalid rating
            created_at=datetime.now(),
        )

    with pytest.raises(ValueError):
        Anomaly(
            id=uuid4(),
            tenant_id=uuid4(),
            transaction_id="txn_001",
            type="invalid_type",  # Invalid type
            risk_score=0.8,
            risk_level=RiskLevel.HIGH,
            amount=1000.0,
            description="Test anomaly",
            status=AnomalyStatus.OPEN,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ) 