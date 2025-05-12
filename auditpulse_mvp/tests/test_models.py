"""Test the database models.

This module tests SQLAlchemy model validation and relationships.
"""
import datetime
import uuid
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.models import (
    Anomaly,
    AnomalyType,
    DataSource,
    FeedbackType,
    Tenant,
    Transaction,
    User,
)


@pytest.mark.asyncio
async def test_transaction_model(db_session: AsyncSession) -> None:
    """Test creating and retrieving a Transaction."""
    # Create a tenant first
    tenant_id = uuid.uuid4()
    tenant = Tenant(
        id=tenant_id,
        name="Test Tenant",
        slug="test-tenant",
    )
    db_session.add(tenant)
    await db_session.flush()

    # Create a transaction
    transaction_id = uuid.uuid4()
    transaction = Transaction(
        id=transaction_id,
        tenant_id=tenant_id,
        transaction_id="txn-123",
        source=DataSource.QUICKBOOKS,
        source_account_id="account-123",
        amount=100.50,
        currency="USD",
        description="Test transaction",
        category="Expenses",
        merchant_name="Test Merchant",
        transaction_date=datetime.datetime.now(),
    )
    db_session.add(transaction)
    await db_session.flush()

    # Query the transaction
    result = await db_session.execute(
        select(Transaction).where(Transaction.id == transaction_id)
    )
    retrieved_transaction = result.scalar_one()

    # Assertions
    assert retrieved_transaction.id == transaction_id
    assert retrieved_transaction.tenant_id == tenant_id
    assert retrieved_transaction.transaction_id == "txn-123"
    assert retrieved_transaction.source == DataSource.QUICKBOOKS
    assert retrieved_transaction.amount == 100.50
    assert retrieved_transaction.currency == "USD"
    assert retrieved_transaction.description == "Test transaction"
    assert retrieved_transaction.is_deleted is False


@pytest.mark.asyncio
async def test_anomaly_model(db_session: AsyncSession) -> None:
    """Test creating and retrieving an Anomaly with its relation to Transaction."""
    # Create a tenant first
    tenant_id = uuid.uuid4()
    tenant = Tenant(
        id=tenant_id,
        name="Test Tenant",
        slug="test-tenant-2",
    )
    db_session.add(tenant)
    await db_session.flush()

    # Create a transaction
    transaction_id = uuid.uuid4()
    transaction = Transaction(
        id=transaction_id,
        tenant_id=tenant_id,
        transaction_id="txn-456",
        source=DataSource.PLAID,
        source_account_id="account-456",
        amount=5000.00,
        currency="USD",
        description="Large test transaction",
        category="Entertainment",
        merchant_name="Test Merchant",
        transaction_date=datetime.datetime.now(),
    )
    db_session.add(transaction)
    await db_session.flush()

    # Create an anomaly
    anomaly_id = uuid.uuid4()
    anomaly = Anomaly(
        id=anomaly_id,
        tenant_id=tenant_id,
        transaction_id=transaction_id,
        anomaly_type=AnomalyType.RULES_BASED,
        rule_name="large_transaction",
        risk_score=85.5,
        confidence_score=95.0,
        description="Transaction amount exceeds threshold of $1,000",
    )
    db_session.add(anomaly)
    await db_session.flush()

    # Query the anomaly with transaction
    result = await db_session.execute(
        select(Anomaly).where(Anomaly.id == anomaly_id)
    )
    retrieved_anomaly = result.scalar_one()

    # Assertions
    assert retrieved_anomaly.id == anomaly_id
    assert retrieved_anomaly.tenant_id == tenant_id
    assert retrieved_anomaly.transaction_id == transaction_id
    assert retrieved_anomaly.anomaly_type == AnomalyType.RULES_BASED
    assert retrieved_anomaly.rule_name == "large_transaction"
    assert retrieved_anomaly.risk_score == 85.5
    assert retrieved_anomaly.confidence_score == 95.0
    assert retrieved_anomaly.resolved is False

    # Test relationship
    assert retrieved_anomaly.transaction.id == transaction_id
    assert retrieved_anomaly.transaction.amount == 5000.00


@pytest.mark.asyncio
async def test_user_model(db_session: AsyncSession) -> None:
    """Test creating and retrieving a User."""
    # Create a tenant first
    tenant_id = uuid.uuid4()
    tenant = Tenant(
        id=tenant_id,
        name="Test Tenant",
        slug="test-tenant-3",
    )
    db_session.add(tenant)
    await db_session.flush()

    # Create a user
    user_id = uuid.uuid4()
    user = User(
        id=user_id,
        tenant_id=tenant_id,
        email="test@example.com",
        hashed_password="hashed_password_here",
        full_name="Test User",
        role="admin",
    )
    db_session.add(user)
    await db_session.flush()

    # Query the user
    result = await db_session.execute(select(User).where(User.id == user_id))
    retrieved_user = result.scalar_one()

    # Assertions
    assert retrieved_user.id == user_id
    assert retrieved_user.tenant_id == tenant_id
    assert retrieved_user.email == "test@example.com"
    assert retrieved_user.full_name == "Test User"
    assert retrieved_user.role == "admin"
    assert retrieved_user.is_active is True


@pytest.mark.asyncio
async def test_tenant_model(db_session: AsyncSession) -> None:
    """Test creating and retrieving a Tenant."""
    # Create a tenant
    tenant_id = uuid.uuid4()
    tenant = Tenant(
        id=tenant_id,
        name="Enterprise Client",
        slug="enterprise-client",
        subscription_tier="enterprise",
        quickbooks_settings={"client_id": "test_client_id"},
        plaid_settings={"client_id": "test_plaid_id"},
    )
    db_session.add(tenant)
    await db_session.flush()

    # Query the tenant
    result = await db_session.execute(select(Tenant).where(Tenant.id == tenant_id))
    retrieved_tenant = result.scalar_one()

    # Assertions
    assert retrieved_tenant.id == tenant_id
    assert retrieved_tenant.name == "Enterprise Client"
    assert retrieved_tenant.slug == "enterprise-client"
    assert retrieved_tenant.subscription_tier == "enterprise"
    assert retrieved_tenant.quickbooks_settings == {"client_id": "test_client_id"}
    assert retrieved_tenant.plaid_settings == {"client_id": "test_plaid_id"}


@pytest.mark.asyncio
async def test_transaction_anomaly_relationship(db_session: AsyncSession) -> None:
    """Test the bidirectional relationship between Transaction and Anomaly."""
    # Create a tenant first
    tenant_id = uuid.uuid4()
    tenant = Tenant(
        id=tenant_id,
        name="Test Tenant",
        slug="test-tenant-4",
    )
    db_session.add(tenant)
    await db_session.flush()

    # Create a transaction
    transaction_id = uuid.uuid4()
    transaction = Transaction(
        id=transaction_id,
        tenant_id=tenant_id,
        transaction_id="txn-789",
        source=DataSource.QUICKBOOKS,
        source_account_id="account-789",
        amount=10000.00,
        currency="USD",
        description="Very large transaction",
        category="Other",
        merchant_name="Unknown Merchant",
        transaction_date=datetime.datetime.now(),
    )
    db_session.add(transaction)

    # Create multiple anomalies
    anomaly1_id = uuid.uuid4()
    anomaly1 = Anomaly(
        id=anomaly1_id,
        tenant_id=tenant_id,
        transaction_id=transaction_id,
        anomaly_type=AnomalyType.RULES_BASED,
        rule_name="large_transaction",
        risk_score=95.0,
        confidence_score=99.0,
        description="Transaction amount far exceeds normal range",
    )

    anomaly2_id = uuid.uuid4()
    anomaly2 = Anomaly(
        id=anomaly2_id,
        tenant_id=tenant_id,
        transaction_id=transaction_id,
        anomaly_type=AnomalyType.ML_BASED,
        risk_score=85.0,
        confidence_score=90.0,
        description="Unusual merchant for this account",
    )

    db_session.add_all([anomaly1, anomaly2])
    await db_session.flush()

    # Query the transaction with its anomalies
    result = await db_session.execute(
        select(Transaction).where(Transaction.id == transaction_id)
    )
    retrieved_transaction = result.scalar_one()

    # Assertions
    assert retrieved_transaction.id == transaction_id
    assert len(retrieved_transaction.anomalies) == 2
    assert any(anomaly.id == anomaly1_id for anomaly in retrieved_transaction.anomalies)
    assert any(anomaly.id == anomaly2_id for anomaly in retrieved_transaction.anomalies)

    # Query one of the anomalies with its transaction
    result = await db_session.execute(
        select(Anomaly).where(Anomaly.id == anomaly1_id)
    )
    retrieved_anomaly = result.scalar_one()

    # Assertions
    assert retrieved_anomaly.transaction.id == transaction_id
    assert retrieved_anomaly.transaction.amount == 10000.00


@pytest.mark.asyncio
async def test_feedback_update(db_session: AsyncSession) -> None:
    """Test updating anomaly with user feedback."""
    # Create a tenant first
    tenant_id = uuid.uuid4()
    tenant = Tenant(
        id=tenant_id,
        name="Test Tenant",
        slug="test-tenant-5",
    )
    db_session.add(tenant)
    await db_session.flush()

    # Create a transaction
    transaction_id = uuid.uuid4()
    transaction = Transaction(
        id=transaction_id,
        tenant_id=tenant_id,
        transaction_id="txn-101112",
        source=DataSource.PLAID,
        source_account_id="account-101112",
        amount=500.00,
        currency="USD",
        description="Medium transaction",
        category="Travel",
        merchant_name="Airline",
        transaction_date=datetime.datetime.now(),
    )
    db_session.add(transaction)

    # Create an anomaly
    anomaly_id = uuid.uuid4()
    anomaly = Anomaly(
        id=anomaly_id,
        tenant_id=tenant_id,
        transaction_id=transaction_id,
        anomaly_type=AnomalyType.ML_BASED,
        risk_score=75.0,
        confidence_score=80.0,
        description="Unusual travel expense",
    )
    db_session.add(anomaly)
    await db_session.flush()

    # Update the anomaly with feedback
    anomaly.feedback = FeedbackType.FALSE_POSITIVE
    anomaly.feedback_notes = "This is a legitimate business expense"
    anomaly.resolved = True
    anomaly.resolved_by = uuid.uuid4()  # Some user ID
    anomaly.resolved_at = datetime.datetime.now()
    await db_session.flush()

    # Query the updated anomaly
    result = await db_session.execute(
        select(Anomaly).where(Anomaly.id == anomaly_id)
    )
    retrieved_anomaly = result.scalar_one()

    # Assertions
    assert retrieved_anomaly.feedback == FeedbackType.FALSE_POSITIVE
    assert retrieved_anomaly.feedback_notes == "This is a legitimate business expense"
    assert retrieved_anomaly.resolved is True
    assert retrieved_anomaly.resolved_by is not None
    assert retrieved_anomaly.resolved_at is not None 