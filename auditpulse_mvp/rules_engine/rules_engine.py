"""Rules engine for transaction anomaly detection.

This module implements rule-based anomaly detection for financial transactions.
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.models import Transaction, AnomalyType

# Configure logging
logger = logging.getLogger(__name__)


class Rule:
    """Base class for transaction rules."""

    def __init__(self, name: str, description: str, weight: float = 1.0):
        """Initialize a rule.

        Args:
            name: Rule name
            description: Rule description
            weight: Rule weight (default: 1.0)
        """
        self.name = name
        self.description = description
        self.weight = weight

    async def evaluate(
        self, transaction: Transaction, db: AsyncSession
    ) -> Tuple[float, str]:
        """Evaluate a transaction against this rule.

        Args:
            transaction: Transaction to evaluate
            db: Database session

        Returns:
            Tuple of (score, explanation)
        """
        raise NotImplementedError


class AmountThresholdRule(Rule):
    """Rule for transactions exceeding amount thresholds."""

    def __init__(self, threshold: float, weight: float = 1.0):
        """Initialize amount threshold rule.

        Args:
            threshold: Amount threshold
            weight: Rule weight
        """
        super().__init__(
            name="Amount Threshold",
            description=f"Transaction amount exceeds {threshold}",
            weight=weight,
        )
        self.threshold = threshold

    async def evaluate(
        self, transaction: Transaction, db: AsyncSession
    ) -> Tuple[float, str]:
        """Evaluate transaction amount against threshold.

        Args:
            transaction: Transaction to evaluate
            db: Database session

        Returns:
            Tuple of (score, explanation)
        """
        if transaction.amount > self.threshold:
            score = min(1.0, transaction.amount / (self.threshold * 2))
            return (
                score,
                f"Transaction amount {transaction.amount} exceeds threshold {self.threshold}",
            )
        return 0.0, ""


class UnapprovedVendorRule(Rule):
    """Rule for transactions with unapproved vendors."""

    def __init__(self, approved_vendors: List[str], weight: float = 1.0):
        """Initialize unapproved vendor rule.

        Args:
            approved_vendors: List of approved vendor names
            weight: Rule weight
        """
        super().__init__(
            name="Unapproved Vendor",
            description="Transaction vendor not in approved list",
            weight=weight,
        )
        self.approved_vendors = set(v.lower() for v in approved_vendors)

    async def evaluate(
        self, transaction: Transaction, db: AsyncSession
    ) -> Tuple[float, str]:
        """Evaluate transaction vendor against approved list.

        Args:
            transaction: Transaction to evaluate
            db: Database session

        Returns:
            Tuple of (score, explanation)
        """
        if not transaction.merchant_name:
            return 0.0, ""

        if transaction.merchant_name.lower() not in self.approved_vendors:
            return 0.8, f"Vendor {transaction.merchant_name} not in approved list"
        return 0.0, ""


class StatisticalOutlierRule(Rule):
    """Rule for transactions that are statistical outliers."""

    def __init__(
        self,
        std_dev_threshold: float = 3.0,
        lookback_days: int = 30,
        weight: float = 1.0,
    ):
        """Initialize statistical outlier rule.

        Args:
            std_dev_threshold: Number of standard deviations for outlier
            lookback_days: Days to look back for statistics
            weight: Rule weight
        """
        super().__init__(
            name="Statistical Outlier",
            description=f"Transaction amount is {std_dev_threshold}σ outlier",
            weight=weight,
        )
        self.std_dev_threshold = std_dev_threshold
        self.lookback_days = lookback_days

    async def evaluate(
        self, transaction: Transaction, db: AsyncSession
    ) -> Tuple[float, str]:
        """Evaluate transaction amount against historical statistics.

        Args:
            transaction: Transaction to evaluate
            db: Database session

        Returns:
            Tuple of (score, explanation)
        """
        # Get historical transactions
        cutoff_date = transaction.transaction_date - timedelta(days=self.lookback_days)
        stmt = select(Transaction).where(
            Transaction.tenant_id == transaction.tenant_id,
            Transaction.transaction_date >= cutoff_date,
            Transaction.transaction_date < transaction.transaction_date,
            Transaction.is_deleted == False,
        )
        result = await db.execute(stmt)
        historical = result.scalars().all()

        if not historical:
            return 0.0, ""

        # Calculate statistics
        amounts = [t.amount for t in historical]
        mean = statistics.mean(amounts)
        std_dev = statistics.stdev(amounts) if len(amounts) > 1 else 0

        if std_dev == 0:
            return 0.0, ""

        # Calculate z-score
        z_score = abs(transaction.amount - mean) / std_dev

        if z_score > self.std_dev_threshold:
            score = min(1.0, z_score / (self.std_dev_threshold * 2))
            return (
                score,
                f"Transaction amount {transaction.amount} is {z_score:.1f}σ from mean {mean:.2f}",
            )
        return 0.0, ""


class RulesEngine:
    """Engine for evaluating transaction rules."""

    def __init__(self, db: AsyncSession):
        """Initialize rules engine.

        Args:
            db: Database session
        """
        self.db = db
        self.rules: List[Rule] = []

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the engine.

        Args:
            rule: Rule to add
        """
        self.rules.append(rule)

    async def evaluate(self, transaction: Transaction) -> Dict[str, Any]:
        """Evaluate a transaction against all rules.

        Args:
            transaction: Transaction to evaluate

        Returns:
            Dictionary with evaluation results
        """
        results = []
        total_score = 0.0
        total_weight = 0.0

        for rule in self.rules:
            try:
                score, explanation = await rule.evaluate(transaction, self.db)
                weighted_score = score * rule.weight

                results.append(
                    {
                        "rule_name": rule.name,
                        "rule_type": rule.name.lower().replace(" ", "_"),
                        "score": score,
                        "weight": rule.weight,
                        "weighted_score": weighted_score,
                        "description": explanation,
                    }
                )

                total_score += weighted_score
                total_weight += rule.weight

            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
                continue

        # Calculate final score
        final_score = total_score / total_weight if total_weight > 0 else 0.0

        return {
            "transaction_id": str(transaction.id),
            "score": final_score,
            "flags": results,
        }

    async def get_anomaly_type(self, transaction: Transaction) -> Optional[AnomalyType]:
        """Determine anomaly type based on rule results."""
        result = await self.evaluate(transaction)
        for flag in result["flags"]:
            if flag["rule_type"] == "amount_threshold":
                return AnomalyType.LARGE_AMOUNT
            elif flag["rule_type"] == "unapproved_vendor":
                return AnomalyType.UNAPPROVED_VENDOR
            elif flag["rule_type"] == "statistical_outlier":
                return AnomalyType.UNUSUAL_AMOUNT
        return None

    async def score(self, transaction: Transaction) -> float:
        """Calculate an overall risk score for a transaction."""
        result = await self.evaluate(transaction)
        score = result.get("score", 0.0)
        if not isinstance(score, float):
            try:
                score = float(score)
            except Exception:
                score = 0.0
        return score
