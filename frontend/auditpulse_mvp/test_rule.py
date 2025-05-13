#!/usr/bin/env python
"""Simple script to test the rules engine directly."""
import asyncio
import uuid
import datetime
import os

# Set environment variables for testing
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["DATABASE_TEST_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["SECRET_KEY"] = "test-secret-key"

from auditpulse_mvp.database.models import Transaction, DataSource
from auditpulse_mvp.auditpulse_mvp.rules_engine.rules_engine import LargeAmountRule


async def test_large_amount_rule():
    """Test the LargeAmountRule."""
    # Create a test transaction
    txn = Transaction(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        transaction_id="test-txn-001",
        source=DataSource.QUICKBOOKS,
        source_account_id="test-account",
        amount=15000.0,
        currency="USD",
        description="Test large transaction",
        merchant_name="Test Vendor",
        transaction_date=datetime.datetime.now(),
    )

    # Create and test a rule
    rule = LargeAmountRule(
        name="Test Large Amount",
        description="Test rule",
        threshold=10000.0,
    )

    triggered, score, reason = await rule.evaluate(txn, {})

    print("Rule triggered:", triggered)
    print("Score:", score)
    print("Reason:", reason)

    assert triggered is True
    assert 0.0 < score <= 1.0
    assert "exceeds threshold" in reason

    print("Test passed!")


if __name__ == "__main__":
    asyncio.run(test_large_amount_rule())
