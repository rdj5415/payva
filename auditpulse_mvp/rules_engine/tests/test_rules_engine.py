"""Tests for the rules engine module.

This module contains tests for the rules-based anomaly detection system.
"""

import datetime
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.models import AnomalyType, DataSource, Transaction
from auditpulse_mvp.rules_engine.rules_engine import (
    AmountThresholdRule,
    RulesEngine,
    StatisticalOutlierRule,
    UnapprovedVendorRule,
)


@pytest.fixture
def mock_tenant_id() -> uuid.UUID:
    """Return a mock tenant ID."""
    return uuid.UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def mock_transaction() -> Transaction:
    """Create a mock transaction for testing."""
    return Transaction(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        transaction_id="test-txn-001",
        source=DataSource.QUICKBOOKS,
        source_account_id="test-account",
        amount=5000.0,
        currency="USD",
        description="Test transaction",
        category="Office Supplies",
        merchant_name="Test Vendor",
        transaction_date=datetime.datetime.now(),
    )


@pytest.fixture
def mock_db_session() -> AsyncSession:
    """Create a mock database session."""
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.mark.asyncio
async def test_amount_threshold_rule(mock_transaction, mock_db_session):
    """Test the amount threshold rule."""
    # Create rule with threshold
    rule = AmountThresholdRule(threshold=10000.0)

    # Test below threshold
    score, explanation = await rule.evaluate(mock_transaction, mock_db_session)
    assert score == 0.0
    assert explanation == ""

    # Test above threshold
    mock_transaction.amount = 15000.0
    score, explanation = await rule.evaluate(mock_transaction, mock_db_session)
    assert score == 0.75  # 15000 / (10000 * 2)
    assert "exceeds threshold" in explanation


@pytest.mark.asyncio
async def test_unapproved_vendor_rule(mock_transaction, mock_db_session):
    """Test the unapproved vendor rule."""
    # Create rule with approved vendors
    rule = UnapprovedVendorRule(approved_vendors=["approved vendor"])

    # Test unapproved vendor
    score, explanation = await rule.evaluate(mock_transaction, mock_db_session)
    assert score == 0.8
    assert "not in approved list" in explanation

    # Test approved vendor
    mock_transaction.merchant_name = "approved vendor"
    score, explanation = await rule.evaluate(mock_transaction, mock_db_session)
    assert score == 0.0
    assert explanation == ""

    # Test missing merchant name
    mock_transaction.merchant_name = None
    score, explanation = await rule.evaluate(mock_transaction, mock_db_session)
    assert score == 0.0
    assert explanation == ""


@pytest.mark.asyncio
async def test_statistical_outlier_rule(mock_transaction, mock_db_session):
    """Test the statistical outlier rule."""
    # Create rule
    rule = StatisticalOutlierRule(std_dev_threshold=3.0, lookback_days=30)

    # Mock historical transactions
    historical = [
        Transaction(
            id=uuid.uuid4(),
            tenant_id=mock_transaction.tenant_id,
            amount=1000.0,
            transaction_date=mock_transaction.transaction_date
            - datetime.timedelta(days=i),
        )
        for i in range(1, 6)
    ]

    # Mock database query
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = historical
    mock_db_session.execute.return_value = mock_result

    # Test normal transaction
    score, explanation = await rule.evaluate(mock_transaction, mock_db_session)
    assert score == 0.0
    assert explanation == ""

    # Test outlier transaction
    mock_transaction.amount = 10000.0  # Much higher than mean
    score, explanation = await rule.evaluate(mock_transaction, mock_db_session)
    assert score > 0.0
    assert "σ from mean" in explanation


@pytest.mark.asyncio
async def test_rules_engine_evaluation(mock_transaction, mock_db_session):
    """Test the rules engine evaluation."""
    # Create engine
    engine = RulesEngine(mock_db_session)

    # Add rules
    engine.add_rule(AmountThresholdRule(threshold=10000.0))
    engine.add_rule(UnapprovedVendorRule(approved_vendors=["approved vendor"]))

    # Test evaluation
    result = await engine.evaluate(mock_transaction)

    assert "transaction_id" in result
    assert "score" in result
    assert "flags" in result
    assert len(result["flags"]) > 0


@pytest.mark.asyncio
async def test_rules_engine_anomaly_type(mock_transaction, mock_db_session):
    """Test anomaly type determination."""
    # Create engine
    engine = RulesEngine(mock_db_session)

    # Add rules
    engine.add_rule(AmountThresholdRule(threshold=1000.0))  # Will trigger
    engine.add_rule(
        UnapprovedVendorRule(approved_vendors=["approved vendor"])
    )  # Will trigger

    # Test anomaly type
    anomaly_type = await engine.get_anomaly_type(mock_transaction)
    assert anomaly_type == AnomalyType.LARGE_AMOUNT  # First matching rule type


@pytest.mark.asyncio
async def test_rules_engine_error_handling(mock_transaction, mock_db_session):
    """Test error handling in rules engine."""
    # Create engine
    engine = RulesEngine(mock_db_session)

    # Add rule that will raise an exception
    class ErrorRule(AmountThresholdRule):
        async def evaluate(self, transaction, db):
            raise ValueError("Test error")

    engine.add_rule(ErrorRule(threshold=1000.0))

    # Test evaluation continues despite error
    result = await engine.evaluate(mock_transaction)
    assert "transaction_id" in result
    assert "score" in result
    assert "flags" in result
    assert len(result["flags"]) == 0  # No flags due to error


@pytest.mark.asyncio
async def test_rules_engine_weighted_scores(mock_transaction, mock_db_session):
    """Test weighted score calculation."""
    # Create engine
    engine = RulesEngine(mock_db_session)

    # Add rules with different weights
    engine.add_rule(AmountThresholdRule(threshold=1000.0, weight=2.0))  # Will trigger
    engine.add_rule(
        UnapprovedVendorRule(approved_vendors=["approved vendor"], weight=1.0)
    )  # Will trigger

    # Test evaluation
    result = await engine.evaluate(mock_transaction)

    # Check weighted scores
    assert len(result["flags"]) == 2
    assert result["flags"][0]["weighted_score"] == result["flags"][0]["score"] * 2.0
    assert result["flags"][1]["weighted_score"] == result["flags"][1]["score"] * 1.0
