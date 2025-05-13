"""Rules Engine for AuditPulse MVP.

This module implements the rules-based anomaly detection for transactions:
1. Large transaction amounts (>$10,000)
2. Unapproved vendors
3. Statistical outliers (3σ over a 90-day window)
"""
import datetime
import logging
import statistics
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.models import AnomalyType, Transaction

# Configure logging
logger = logging.getLogger(__name__)


class RuleType(str, Enum):
    """Enum for rule types."""

    LARGE_AMOUNT = "large_amount"
    UNAPPROVED_VENDOR = "unapproved_vendor"
    STATISTICAL_OUTLIER = "statistical_outlier"


@dataclass
class Rule:
    """Base class for rules."""

    name: str
    description: str
    enabled: bool = True
    weight: float = 1.0
    
    async def evaluate(self, 
                       transaction: Transaction, 
                       context: Dict) -> Tuple[bool, float, Optional[str]]:
        """Evaluate the rule for a given transaction.
        
        Args:
            transaction: The transaction to evaluate.
            context: Additional context for rule evaluation.
            
        Returns:
            A tuple of (triggered, score, reason):
            - triggered: True if the rule was triggered, False otherwise.
            - score: A score between 0 and 1, where 1 indicates high risk.
            - reason: A human-readable explanation of why the rule was triggered,
                     or None if the rule was not triggered.
        """
        raise NotImplementedError("Rule subclasses must implement evaluate()")


@dataclass
class LargeAmountRule(Rule):
    """Rule for detecting large transaction amounts."""

    threshold: float = 10000.0
    
    async def evaluate(self, 
                       transaction: Transaction, 
                       context: Dict) -> Tuple[bool, float, Optional[str]]:
        """Evaluate if a transaction has an unusually large amount."""
        if transaction.amount >= self.threshold:
            # The score increases with the amount
            normalized_score = min(1.0, transaction.amount / (self.threshold * 2))
            reason = f"Transaction amount (${transaction.amount:,.2f}) exceeds threshold (${self.threshold:,.2f})"
            return True, normalized_score, reason
        return False, 0.0, None


@dataclass
class UnapprovedVendorRule(Rule):
    """Rule for detecting transactions with unapproved vendors."""

    approved_vendors: Set[str] = field(default_factory=set)
    
    async def evaluate(self, 
                       transaction: Transaction, 
                       context: Dict) -> Tuple[bool, float, Optional[str]]:
        """Evaluate if a transaction's vendor is not in the approved list."""
        if not transaction.merchant_name:
            return False, 0.0, None
            
        # Check if the vendor is in the approved list
        merchant_name = transaction.merchant_name.lower()
        if self.approved_vendors and merchant_name not in self.approved_vendors:
            reason = f"Vendor '{transaction.merchant_name}' is not in the approved vendor list"
            return True, 1.0, reason
        return False, 0.0, None


@dataclass
class StatisticalOutlierRule(Rule):
    """Rule for detecting statistical outliers based on historical transactions."""

    std_dev_threshold: float = 3.0
    window_days: int = 90
    min_transactions: int = 5
    
    async def evaluate(self, 
                       transaction: Transaction, 
                       context: Dict) -> Tuple[bool, float, Optional[str]]:
        """Evaluate if a transaction is a statistical outlier based on historical data."""
        # Get historical transactions from context
        historical_txns = context.get("historical_transactions", [])
        
        if len(historical_txns) < self.min_transactions:
            # Not enough historical data to make a statistical judgment
            return False, 0.0, None
            
        # Calculate mean and standard deviation of historical amounts
        amounts = [txn.amount for txn in historical_txns]
        try:
            mean = statistics.mean(amounts)
            stdev = statistics.stdev(amounts)
        except statistics.StatisticsError:
            logger.warning("Unable to calculate statistics for transaction %s", transaction.id)
            return False, 0.0, None
            
        if stdev == 0:
            # Avoid division by zero
            return False, 0.0, None
            
        # Calculate z-score
        z_score = (transaction.amount - mean) / stdev
        
        # Check if transaction exceeds the threshold
        if z_score > self.std_dev_threshold:
            # Normalize score between 0 and 1, capped at 1.0
            normalized_score = min(1.0, (z_score - self.std_dev_threshold) / 3.0)
            reason = (
                f"Transaction amount (${transaction.amount:,.2f}) is {z_score:.1f} "
                f"standard deviations above the mean (${mean:,.2f})"
            )
            return True, normalized_score, reason
        return False, 0.0, None


class RulesEngine:
    """Engine for evaluating rules against transactions."""

    def __init__(self, db_session: AsyncSession):
        """Initialize the rules engine.
        
        Args:
            db_session: The database session.
        """
        self.db_session = db_session
        self.rules: List[Rule] = []
        self._initialize_default_rules()
        
    def _initialize_default_rules(self):
        """Initialize the default rules."""
        self.rules = [
            LargeAmountRule(
                name="Large Amount",
                description="Flags transactions with amounts exceeding $10,000",
            ),
            UnapprovedVendorRule(
                name="Unapproved Vendor",
                description="Flags transactions from vendors not in the approved list",
            ),
            StatisticalOutlierRule(
                name="Statistical Outlier",
                description="Flags transactions that are statistical outliers (3σ over 90 days)",
            ),
        ]

    def update_rule_config(self, rule_configs: List[Dict]):
        """Update rule configurations.
        
        Args:
            rule_configs: List of rule configuration dictionaries.
        """
        for config in rule_configs:
            rule_name = config.get("name")
            if not rule_name:
                continue
                
            # Find and update the matching rule
            for rule in self.rules:
                if rule.name == rule_name:
                    # Update rule attributes
                    rule.enabled = config.get("enabled", rule.enabled)
                    rule.weight = config.get("weight", rule.weight)
                    
                    # Rule-specific attributes
                    if isinstance(rule, LargeAmountRule):
                        rule.threshold = config.get("threshold", rule.threshold)
                    elif isinstance(rule, UnapprovedVendorRule):
                        if "approved_vendors" in config:
                            rule.approved_vendors = set(config["approved_vendors"])
                    elif isinstance(rule, StatisticalOutlierRule):
                        rule.std_dev_threshold = config.get("std_dev_threshold", rule.std_dev_threshold)
                        rule.window_days = config.get("window_days", rule.window_days)
                        rule.min_transactions = config.get("min_transactions", rule.min_transactions)
                    break
    
    async def get_historical_transactions(
        self, tenant_id: uuid.UUID, transaction: Transaction, days: int = 90
    ) -> List[Transaction]:
        """Get historical transactions for the same account over a time window.
        
        Args:
            tenant_id: The tenant ID.
            transaction: The transaction to get history for.
            days: The number of days to look back.
            
        Returns:
            A list of historical transactions.
        """
        # Calculate the start date for the window
        start_date = transaction.transaction_date - datetime.timedelta(days=days)
        
        # Query for historical transactions in the same account
        stmt = (
            select(Transaction)
            .where(
                and_(
                    Transaction.tenant_id == tenant_id,
                    Transaction.source == transaction.source,
                    Transaction.source_account_id == transaction.source_account_id,
                    Transaction.transaction_date >= start_date,
                    Transaction.transaction_date < transaction.transaction_date,
                    Transaction.is_deleted == False,
                )
            )
            .order_by(Transaction.transaction_date.desc())
        )
        
        result = await self.db_session.execute(stmt)
        return list(result.scalars().all())
    
    async def prepare_context(
        self, tenant_id: uuid.UUID, transaction: Transaction
    ) -> Dict:
        """Prepare the context for rule evaluation.
        
        Args:
            tenant_id: The tenant ID.
            transaction: The transaction to evaluate.
            
        Returns:
            A dictionary containing the evaluation context.
        """
        # Get historical transactions
        historical_txns = await self.get_historical_transactions(
            tenant_id, transaction, days=90
        )
        
        # Prepare context
        context = {
            "tenant_id": tenant_id,
            "historical_transactions": historical_txns,
            # Additional context could be added here
        }
        
        return context
    
    async def evaluate_transaction(
        self, tenant_id: uuid.UUID, transaction: Transaction
    ) -> List[Dict]:
        """Evaluate all rules against a transaction.
        
        Args:
            tenant_id: The tenant ID.
            transaction: The transaction to evaluate.
            
        Returns:
            A list of dictionaries containing the evaluation results for each triggered rule.
        """
        context = await self.prepare_context(tenant_id, transaction)
        results = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
                
            try:
                triggered, score, reason = await rule.evaluate(transaction, context)
                if triggered:
                    results.append({
                        "rule_name": rule.name,
                        "rule_type": self._get_rule_type(rule),
                        "score": score * rule.weight,
                        "description": reason,
                        "weight": rule.weight,
                    })
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
        
        return results
    
    def _get_rule_type(self, rule: Rule) -> RuleType:
        """Get the rule type for a rule instance.
        
        Args:
            rule: The rule instance.
            
        Returns:
            The rule type.
        """
        if isinstance(rule, LargeAmountRule):
            return RuleType.LARGE_AMOUNT
        elif isinstance(rule, UnapprovedVendorRule):
            return RuleType.UNAPPROVED_VENDOR
        elif isinstance(rule, StatisticalOutlierRule):
            return RuleType.STATISTICAL_OUTLIER
        else:
            return RuleType.STATISTICAL_OUTLIER  # Default
    
    async def score(self, tenant_id: uuid.UUID, transaction: Transaction) -> float:
        """Calculate an overall risk score for a transaction.
        
        Args:
            tenant_id: The tenant ID.
            transaction: The transaction to score.
            
        Returns:
            A risk score between 0 and 1, where higher values indicate higher risk.
        """
        results = await self.evaluate_transaction(tenant_id, transaction)
        
        if not results:
            return 0.0
            
        # Calculate weighted average score
        total_weight = sum(result["weight"] for result in results)
        weighted_score = sum(result["score"] for result in results)
        
        if total_weight == 0:
            return 0.0
            
        # Normalize to 0-1 range
        return min(1.0, weighted_score / total_weight)
    
    async def flags(self, tenant_id: uuid.UUID, transaction: Transaction) -> List[Dict]:
        """Get a list of flags for a transaction.
        
        Args:
            tenant_id: The tenant ID.
            transaction: The transaction to check.
            
        Returns:
            A list of dictionaries containing information about each flag.
        """
        return await self.evaluate_transaction(tenant_id, transaction) 