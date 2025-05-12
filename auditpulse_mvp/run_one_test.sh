#!/bin/bash
# Script to run a single test without using the pytest framework

# Exit on error
set -e

# Enable verbose mode
set -x

# Set environment variables
export DATABASE_URL=sqlite+aiosqlite:///:memory:
export DATABASE_TEST_URL=sqlite+aiosqlite:///:memory:
export SECRET_KEY=test-secret-key
export PYTHONPATH=$(pwd):$PYTHONPATH

# Create a temporary test file
cat > test_one_rule.py << 'EOF'
#!/usr/bin/env python
"""Test a single rule from the rules engine."""
import asyncio
import unittest
import uuid
import datetime

from auditpulse_mvp.database.models import Transaction, DataSource
from auditpulse_mvp.auditpulse_mvp.rules_engine.rules_engine import (
    LargeAmountRule,
    UnapprovedVendorRule,
    StatisticalOutlierRule,
)


class RulesTest(unittest.TestCase):
    """Test cases for rules."""

    def test_large_amount_rule(self):
        """Test LargeAmountRule."""
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

        # Run the evaluation using asyncio
        triggered, score, reason = asyncio.run(rule.evaluate(txn, {}))

        # Verify results
        self.assertTrue(triggered)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIn("exceeds threshold", reason)

    def test_unapproved_vendor_rule(self):
        """Test UnapprovedVendorRule."""
        # Create a test transaction
        txn = Transaction(
            id=uuid.uuid4(),
            tenant_id=uuid.uuid4(),
            transaction_id="test-txn-002",
            source=DataSource.QUICKBOOKS,
            source_account_id="test-account",
            amount=5000.0,
            currency="USD",
            description="Test transaction",
            merchant_name="Unapproved Vendor",
            transaction_date=datetime.datetime.now(),
        )

        # Create and test a rule with approved vendors
        rule = UnapprovedVendorRule(
            name="Test Unapproved Vendor",
            description="Test rule",
            approved_vendors={"approved vendor", "another vendor"},
        )

        # Run the evaluation using asyncio
        triggered, score, reason = asyncio.run(rule.evaluate(txn, {}))

        # Verify results
        self.assertTrue(triggered)
        self.assertEqual(score, 1.0)
        self.assertIn("not in the approved vendor list", reason)

    def test_statistical_outlier(self):
        """Test StatisticalOutlierRule."""
        # Create a test transaction
        txn = Transaction(
            id=uuid.uuid4(),
            tenant_id=uuid.uuid4(),
            transaction_id="test-txn-003",
            source=DataSource.QUICKBOOKS,
            source_account_id="test-account",
            amount=10000.0,
            currency="USD",
            description="Test outlier transaction",
            merchant_name="Test Vendor",
            transaction_date=datetime.datetime.now(),
        )

        # Create historical transactions
        historical_txns = []
        base_date = datetime.datetime.now() - datetime.timedelta(days=30)
        for i in range(10):
            historical_txns.append(
                Transaction(
                    id=uuid.uuid4(),
                    tenant_id=uuid.uuid4(),
                    transaction_id=f"hist-txn-{i}",
                    source=DataSource.QUICKBOOKS,
                    source_account_id="test-account",
                    amount=1000.0 + (i * 100),  # Vary the amounts
                    currency="USD",
                    description=f"Historical transaction {i}",
                    merchant_name="Test Vendor",
                    transaction_date=base_date + datetime.timedelta(days=i),
                )
            )

        # Create and test a rule
        rule = StatisticalOutlierRule(
            name="Test Statistical Outlier",
            description="Test rule",
            std_dev_threshold=2.0,  # Lower threshold for this test
            min_transactions=5,
        )

        # Prepare context with historical transactions
        context = {"historical_transactions": historical_txns}

        # Run the evaluation using asyncio
        triggered, score, reason = asyncio.run(rule.evaluate(txn, context))

        # Verify results
        self.assertTrue(triggered)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIn("standard deviations above the mean", reason)


if __name__ == "__main__":
    unittest.main()
EOF

# Run the test file
python test_one_rule.py

# Clean up
rm test_one_rule.py 