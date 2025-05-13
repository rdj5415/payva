"""Tests for the rules engine module.

This module contains unit tests for the rules-based anomaly detection,
covering all rule types and edge cases.
"""

import datetime
import uuid
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.models import DataSource, Transaction
from auditpulse_mvp.auditpulse_mvp.rules_engine.rules_engine import (
    LargeAmountRule,
    Rule,
    RuleType,
    RulesEngine,
    StatisticalOutlierRule,
    UnapprovedVendorRule,
)


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
def mock_large_transaction() -> Transaction:
    """Create a mock large transaction for testing."""
    return Transaction(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        transaction_id="test-txn-002",
        source=DataSource.QUICKBOOKS,
        source_account_id="test-account",
        amount=15000.0,
        currency="USD",
        description="Large test transaction",
        category="Equipment",
        merchant_name="Big Purchase Inc",
        transaction_date=datetime.datetime.now(),
    )


@pytest.fixture
def mock_historical_transactions() -> list[Transaction]:
    """Create a list of mock historical transactions for testing."""
    base_date = datetime.datetime.now() - datetime.timedelta(days=30)
    return [
        Transaction(
            id=uuid.uuid4(),
            tenant_id=uuid.uuid4(),
            transaction_id=f"hist-txn-{i}",
            source=DataSource.QUICKBOOKS,
            source_account_id="test-account",
            amount=1000.0 + (i * 100),  # Varying amounts
            currency="USD",
            description=f"Historical transaction {i}",
            category="Office Supplies",
            merchant_name="Regular Vendor",
            transaction_date=base_date + datetime.timedelta(days=i),
        )
        for i in range(10)
    ]


@pytest.fixture
def mock_outlier_transaction() -> Transaction:
    """Create a mock outlier transaction for testing."""
    return Transaction(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        transaction_id="test-txn-003",
        source=DataSource.QUICKBOOKS,
        source_account_id="test-account",
        amount=10000.0,  # Much higher than the historical transactions
        currency="USD",
        description="Potential outlier transaction",
        category="Office Supplies",
        merchant_name="Regular Vendor",
        transaction_date=datetime.datetime.now(),
    )


@pytest.mark.asyncio
async def test_large_amount_rule_triggered(mock_large_transaction):
    """Test that the LargeAmountRule triggers for large amounts."""
    rule = LargeAmountRule(
        name="Test Large Amount", description="Test rule", threshold=10000.0
    )

    triggered, score, reason = await rule.evaluate(mock_large_transaction, {})

    assert triggered is True
    assert 0.0 < score <= 1.0
    assert "exceeds threshold" in reason
    assert "15,000.00" in reason
    assert "10,000.00" in reason


@pytest.mark.asyncio
async def test_large_amount_rule_not_triggered(mock_transaction):
    """Test that the LargeAmountRule doesn't trigger for normal amounts."""
    rule = LargeAmountRule(
        name="Test Large Amount", description="Test rule", threshold=10000.0
    )

    triggered, score, reason = await rule.evaluate(mock_transaction, {})

    assert triggered is False
    assert score == 0.0
    assert reason is None


@pytest.mark.asyncio
async def test_large_amount_rule_boundary():
    """Test the LargeAmountRule at the boundary threshold value."""
    rule = LargeAmountRule(
        name="Test Large Amount", description="Test rule", threshold=10000.0
    )

    # Test exactly at threshold
    txn_at_threshold = Transaction(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        transaction_id="threshold-txn",
        source=DataSource.QUICKBOOKS,
        source_account_id="test-account",
        amount=10000.0,
        transaction_date=datetime.datetime.now(),
    )

    triggered, score, reason = await rule.evaluate(txn_at_threshold, {})

    assert triggered is True
    assert score == 0.5  # 10000 / (10000 * 2) = 0.5
    assert reason is not None

    # Test just below threshold
    txn_below_threshold = Transaction(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        transaction_id="below-threshold-txn",
        source=DataSource.QUICKBOOKS,
        source_account_id="test-account",
        amount=9999.99,
        transaction_date=datetime.datetime.now(),
    )

    triggered, score, reason = await rule.evaluate(txn_below_threshold, {})

    assert triggered is False
    assert score == 0.0
    assert reason is None


@pytest.mark.asyncio
async def test_unapproved_vendor_rule_triggered():
    """Test that the UnapprovedVendorRule triggers for unapproved vendors."""
    rule = UnapprovedVendorRule(
        name="Test Unapproved Vendor",
        description="Test rule",
        approved_vendors={"approved vendor", "another approved vendor"},
    )

    txn = Transaction(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        transaction_id="vendor-test-txn",
        source=DataSource.QUICKBOOKS,
        source_account_id="test-account",
        amount=1000.0,
        merchant_name="Unapproved Vendor",
        transaction_date=datetime.datetime.now(),
    )

    triggered, score, reason = await rule.evaluate(txn, {})

    assert triggered is True
    assert score == 1.0
    assert "not in the approved vendor list" in reason
    assert "Unapproved Vendor" in reason


@pytest.mark.asyncio
async def test_unapproved_vendor_rule_not_triggered():
    """Test that the UnapprovedVendorRule doesn't trigger for approved vendors."""
    rule = UnapprovedVendorRule(
        name="Test Unapproved Vendor",
        description="Test rule",
        approved_vendors={"approved vendor", "regular vendor"},
    )

    txn = Transaction(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        transaction_id="vendor-test-txn",
        source=DataSource.QUICKBOOKS,
        source_account_id="test-account",
        amount=1000.0,
        merchant_name="Approved Vendor",  # Case insensitive check
        transaction_date=datetime.datetime.now(),
    )

    # Make the vendor name match one of the approved vendors (case insensitive)
    txn.merchant_name = "Regular Vendor"

    triggered, score, reason = await rule.evaluate(txn, {})

    assert triggered is False
    assert score == 0.0
    assert reason is None


@pytest.mark.asyncio
async def test_unapproved_vendor_rule_empty_approved_list():
    """Test the UnapprovedVendorRule with an empty approved vendors list."""
    rule = UnapprovedVendorRule(
        name="Test Unapproved Vendor",
        description="Test rule",
        approved_vendors=set(),  # Empty set
    )

    txn = Transaction(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        transaction_id="vendor-test-txn",
        source=DataSource.QUICKBOOKS,
        source_account_id="test-account",
        amount=1000.0,
        merchant_name="Any Vendor",
        transaction_date=datetime.datetime.now(),
    )

    triggered, score, reason = await rule.evaluate(txn, {})

    # Should not trigger if the approved list is empty
    assert triggered is False
    assert score == 0.0
    assert reason is None


@pytest.mark.asyncio
async def test_unapproved_vendor_rule_missing_merchant():
    """Test the UnapprovedVendorRule with a transaction missing merchant name."""
    rule = UnapprovedVendorRule(
        name="Test Unapproved Vendor",
        description="Test rule",
        approved_vendors={"approved vendor"},
    )

    txn = Transaction(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        transaction_id="vendor-test-txn",
        source=DataSource.QUICKBOOKS,
        source_account_id="test-account",
        amount=1000.0,
        merchant_name=None,  # Missing merchant name
        transaction_date=datetime.datetime.now(),
    )

    triggered, score, reason = await rule.evaluate(txn, {})

    # Should not trigger if merchant name is missing
    assert triggered is False
    assert score == 0.0
    assert reason is None


@pytest.mark.asyncio
async def test_statistical_outlier_rule_triggered(
    mock_historical_transactions, mock_outlier_transaction
):
    """Test that the StatisticalOutlierRule triggers for outlier transactions."""
    rule = StatisticalOutlierRule(
        name="Test Statistical Outlier",
        description="Test rule",
        std_dev_threshold=2.0,  # Lower threshold for testing
        min_transactions=5,
    )

    context = {"historical_transactions": mock_historical_transactions}

    triggered, score, reason = await rule.evaluate(mock_outlier_transaction, context)

    assert triggered is True
    assert 0.0 < score <= 1.0
    assert "standard deviations above the mean" in reason
    assert "10,000.00" in reason


@pytest.mark.asyncio
async def test_statistical_outlier_rule_not_triggered(mock_historical_transactions):
    """Test that the StatisticalOutlierRule doesn't trigger for normal transactions."""
    rule = StatisticalOutlierRule(
        name="Test Statistical Outlier",
        description="Test rule",
        std_dev_threshold=3.0,
        min_transactions=5,
    )

    # Create a transaction with an amount similar to the historical ones
    normal_txn = Transaction(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        transaction_id="normal-txn",
        source=DataSource.QUICKBOOKS,
        source_account_id="test-account",
        amount=1500.0,  # Within normal range of historical transactions
        transaction_date=datetime.datetime.now(),
    )

    context = {"historical_transactions": mock_historical_transactions}

    triggered, score, reason = await rule.evaluate(normal_txn, context)

    assert triggered is False
    assert score == 0.0
    assert reason is None


@pytest.mark.asyncio
async def test_statistical_outlier_rule_insufficient_history():
    """Test the StatisticalOutlierRule with insufficient historical data."""
    rule = StatisticalOutlierRule(
        name="Test Statistical Outlier",
        description="Test rule",
        std_dev_threshold=3.0,
        min_transactions=5,
    )

    txn = Transaction(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        transaction_id="test-txn",
        source=DataSource.QUICKBOOKS,
        source_account_id="test-account",
        amount=5000.0,
        transaction_date=datetime.datetime.now(),
    )

    # Not enough historical transactions
    context = {
        "historical_transactions": [
            Transaction(
                id=uuid.uuid4(),
                tenant_id=uuid.uuid4(),
                transaction_id=f"hist-txn-{i}",
                source=DataSource.QUICKBOOKS,
                source_account_id="test-account",
                amount=1000.0,
                transaction_date=datetime.datetime.now() - datetime.timedelta(days=i),
            )
            for i in range(3)  # Only 3 transactions, less than min_transactions=5
        ]
    }

    triggered, score, reason = await rule.evaluate(txn, context)

    # Should not trigger with insufficient history
    assert triggered is False
    assert score == 0.0
    assert reason is None


@pytest.mark.asyncio
async def test_statistical_outlier_rule_zero_std_dev():
    """Test the StatisticalOutlierRule with zero standard deviation in history."""
    rule = StatisticalOutlierRule(
        name="Test Statistical Outlier",
        description="Test rule",
        std_dev_threshold=3.0,
        min_transactions=5,
    )

    txn = Transaction(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        transaction_id="test-txn",
        source=DataSource.QUICKBOOKS,
        source_account_id="test-account",
        amount=5000.0,
        transaction_date=datetime.datetime.now(),
    )

    # All historical transactions have the same amount
    context = {
        "historical_transactions": [
            Transaction(
                id=uuid.uuid4(),
                tenant_id=uuid.uuid4(),
                transaction_id=f"hist-txn-{i}",
                source=DataSource.QUICKBOOKS,
                source_account_id="test-account",
                amount=1000.0,  # Same amount for all
                transaction_date=datetime.datetime.now() - datetime.timedelta(days=i),
            )
            for i in range(10)
        ]
    }

    triggered, score, reason = await rule.evaluate(txn, context)

    # Should not trigger with zero standard deviation
    assert triggered is False
    assert score == 0.0
    assert reason is None


@pytest.mark.asyncio
async def test_rules_engine_initialization(db_session):
    """Test that the RulesEngine initializes with default rules."""
    engine = RulesEngine(db_session)

    # Should have 3 default rules
    assert len(engine.rules) == 3

    # Verify rule types
    rule_types = [type(rule) for rule in engine.rules]
    assert LargeAmountRule in rule_types
    assert UnapprovedVendorRule in rule_types
    assert StatisticalOutlierRule in rule_types


@pytest.mark.asyncio
async def test_rules_engine_update_config(db_session):
    """Test updating rule configurations in the RulesEngine."""
    engine = RulesEngine(db_session)

    # Original threshold for LargeAmountRule
    large_amount_rule = next(r for r in engine.rules if isinstance(r, LargeAmountRule))
    assert large_amount_rule.threshold == 10000.0

    # Update configuration
    engine.update_rule_config(
        [
            {
                "name": "Large Amount",
                "enabled": True,
                "weight": 2.0,
                "threshold": 5000.0,
            }
        ]
    )

    # Verify updates applied
    large_amount_rule = next(r for r in engine.rules if isinstance(r, LargeAmountRule))
    assert large_amount_rule.threshold == 5000.0
    assert large_amount_rule.weight == 2.0
    assert large_amount_rule.enabled is True


@pytest.mark.asyncio
async def test_rules_engine_update_approved_vendors(db_session):
    """Test updating approved vendors in the RulesEngine."""
    engine = RulesEngine(db_session)

    # Original approved vendors for UnapprovedVendorRule
    vendor_rule = next(r for r in engine.rules if isinstance(r, UnapprovedVendorRule))
    assert len(vendor_rule.approved_vendors) == 0  # Empty by default

    # Update configuration with approved vendors
    approved_vendors = ["Vendor A", "Vendor B", "Vendor C"]
    engine.update_rule_config(
        [
            {
                "name": "Unapproved Vendor",
                "approved_vendors": approved_vendors,
            }
        ]
    )

    # Verify updates applied
    vendor_rule = next(r for r in engine.rules if isinstance(r, UnapprovedVendorRule))
    assert vendor_rule.approved_vendors == set(v.lower() for v in approved_vendors)


@pytest.mark.asyncio
async def test_rules_engine_get_historical_transactions(
    db_session, mock_transaction, mock_historical_transactions
):
    """Test retrieving historical transactions in the RulesEngine."""
    tenant_id = uuid.uuid4()

    # Set the same tenant_id and source_account_id for all transactions
    for txn in mock_historical_transactions:
        txn.tenant_id = tenant_id
        txn.source_account_id = "test-account"

    mock_transaction.tenant_id = tenant_id
    mock_transaction.source_account_id = "test-account"

    # Mock the database query
    with patch.object(AsyncSession, "execute") as mock_execute:
        # Mock the result of the query
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_historical_transactions
        mock_execute.return_value = mock_result

        engine = RulesEngine(db_session)

        # Call the method
        result = await engine.get_historical_transactions(
            tenant_id, mock_transaction, days=90
        )

        # Verify the result
        assert result == mock_historical_transactions
        mock_execute.assert_called_once()


@pytest.mark.asyncio
async def test_rules_engine_evaluate_transaction(db_session, mock_large_transaction):
    """Test evaluating transactions in the RulesEngine."""
    tenant_id = uuid.uuid4()
    mock_large_transaction.tenant_id = tenant_id

    engine = RulesEngine(db_session)

    # Mock prepare_context to return an empty context
    with patch.object(RulesEngine, "prepare_context") as mock_prepare_context:
        mock_prepare_context.return_value = {"historical_transactions": []}

        # Evaluate the transaction
        results = await engine.evaluate_transaction(tenant_id, mock_large_transaction)

        # Should trigger the LargeAmountRule
        assert len(results) == 1
        assert results[0]["rule_name"] == "Large Amount"
        assert results[0]["rule_type"] == RuleType.LARGE_AMOUNT
        assert 0.0 < results[0]["score"] <= 1.0
        assert "exceeds threshold" in results[0]["description"]


@pytest.mark.asyncio
async def test_rules_engine_score(db_session, mock_large_transaction):
    """Test scoring transactions in the RulesEngine."""
    tenant_id = uuid.uuid4()
    mock_large_transaction.tenant_id = tenant_id

    engine = RulesEngine(db_session)

    # Mock evaluate_transaction to return a known result
    with patch.object(RulesEngine, "evaluate_transaction") as mock_evaluate:
        mock_evaluate.return_value = [
            {
                "rule_name": "Large Amount",
                "rule_type": RuleType.LARGE_AMOUNT,
                "score": 0.75,
                "description": "Transaction amount exceeds threshold",
                "weight": 1.0,
            }
        ]

        # Get the score
        score = await engine.score(tenant_id, mock_large_transaction)

        # Should return the normalized score
        assert score == 0.75
        mock_evaluate.assert_called_once_with(tenant_id, mock_large_transaction)


@pytest.mark.asyncio
async def test_rules_engine_score_multiple_rules(db_session, mock_large_transaction):
    """Test scoring transactions with multiple triggered rules."""
    tenant_id = uuid.uuid4()
    mock_large_transaction.tenant_id = tenant_id

    engine = RulesEngine(db_session)

    # Mock evaluate_transaction to return multiple results
    with patch.object(RulesEngine, "evaluate_transaction") as mock_evaluate:
        mock_evaluate.return_value = [
            {
                "rule_name": "Large Amount",
                "rule_type": RuleType.LARGE_AMOUNT,
                "score": 0.75,
                "description": "Transaction amount exceeds threshold",
                "weight": 1.0,
            },
            {
                "rule_name": "Statistical Outlier",
                "rule_type": RuleType.STATISTICAL_OUTLIER,
                "score": 0.60,
                "description": "Transaction is a statistical outlier",
                "weight": 2.0,  # Higher weight
            },
        ]

        # Get the score
        score = await engine.score(tenant_id, mock_large_transaction)

        # Calculate expected score: (0.75*1.0 + 0.60*2.0) / (1.0 + 2.0) = 0.65
        expected_score = (0.75 * 1.0 + 0.60 * 2.0) / (1.0 + 2.0)
        assert score == expected_score
        mock_evaluate.assert_called_once_with(tenant_id, mock_large_transaction)


@pytest.mark.asyncio
async def test_rules_engine_score_no_triggered_rules(db_session, mock_transaction):
    """Test scoring transactions with no triggered rules."""
    tenant_id = uuid.uuid4()
    mock_transaction.tenant_id = tenant_id

    engine = RulesEngine(db_session)

    # Mock evaluate_transaction to return an empty list
    with patch.object(RulesEngine, "evaluate_transaction") as mock_evaluate:
        mock_evaluate.return_value = []

        # Get the score
        score = await engine.score(tenant_id, mock_transaction)

        # Should return 0.0 when no rules are triggered
        assert score == 0.0
        mock_evaluate.assert_called_once_with(tenant_id, mock_transaction)


@pytest.mark.asyncio
async def test_rules_engine_flags(db_session, mock_large_transaction):
    """Test getting flags for transactions in the RulesEngine."""
    tenant_id = uuid.uuid4()
    mock_large_transaction.tenant_id = tenant_id

    engine = RulesEngine(db_session)

    # Mock evaluate_transaction to return a known result
    expected_flags = [
        {
            "rule_name": "Large Amount",
            "rule_type": RuleType.LARGE_AMOUNT,
            "score": 0.75,
            "description": "Transaction amount exceeds threshold",
            "weight": 1.0,
        }
    ]

    with patch.object(RulesEngine, "evaluate_transaction") as mock_evaluate:
        mock_evaluate.return_value = expected_flags

        # Get the flags
        flags = await engine.flags(tenant_id, mock_large_transaction)

        # Should return the flags from evaluate_transaction
        assert flags == expected_flags
        mock_evaluate.assert_called_once_with(tenant_id, mock_large_transaction)


@pytest.mark.asyncio
async def test_rule_evaluation_exception_handling(db_session, mock_transaction):
    """Test that exceptions during rule evaluation are handled gracefully."""
    tenant_id = uuid.uuid4()
    mock_transaction.tenant_id = tenant_id

    # Create a custom rule that raises an exception
    class BuggyRule(Rule):
        async def evaluate(self, transaction, context):
            raise ValueError("Simulated rule evaluation error")

    engine = RulesEngine(db_session)

    # Add the buggy rule
    engine.rules.append(
        BuggyRule(name="Buggy Rule", description="A rule that raises an exception")
    )

    # Mock prepare_context to return an empty context
    with patch.object(RulesEngine, "prepare_context") as mock_prepare_context:
        mock_prepare_context.return_value = {}

        # Evaluate the transaction - should not raise an exception
        results = await engine.evaluate_transaction(tenant_id, mock_transaction)

        # Should only include results from non-buggy rules
        for result in results:
            assert result["rule_name"] != "Buggy Rule"
