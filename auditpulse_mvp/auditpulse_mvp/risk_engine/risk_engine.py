"""Risk Scoring Engine for AuditPulse MVP.

This module provides functionality to score anomalies by integrating
machine learning predictions with rule-based signals, and generating
human-readable explanations of risk factors.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import math

from pydantic import BaseModel, Field, field_validator

from auditpulse_mvp.utils.settings import settings
from auditpulse_mvp.risk_engine.explanations import get_explanation_provider

logger = logging.getLogger(__name__)


class InvalidRiskConfigurationError(Exception):
    """Exception raised when risk configuration is invalid."""

    pass


class RiskWeights(BaseModel):
    """Model for risk scoring weights."""

    ml_weight: float = Field(0.7, ge=0.0, le=1.0, description="Weight for ML scores")
    rules_weight: float = Field(
        0.3, ge=0.0, le=1.0, description="Weight for rule-based scores"
    )

    @field_validator("rules_weight")
    def validate_weights_sum(cls, v: float, values: Dict[str, float]) -> float:
        """Validate that weights sum to 1.0."""
        ml_weight = values.get("ml_weight", 0.0)
        if ml_weight + v != 1.0:
            raise ValueError("Weights must sum to 1.0")
        return v


class RiskSensitivity(BaseModel):
    """Model for risk sensitivity thresholds."""

    low_threshold: int = Field(30, ge=0, le=100, description="Threshold for low risk")
    medium_threshold: int = Field(
        60, ge=0, le=100, description="Threshold for medium risk"
    )
    high_threshold: int = Field(80, ge=0, le=100, description="Threshold for high risk")

    @field_validator("medium_threshold")
    def validate_medium_threshold(cls, v: int, values: Dict[str, int]) -> int:
        """Validate that medium threshold is greater than low threshold."""
        low = values.get("low_threshold", 0)
        if v <= low:
            raise ValueError("Medium threshold must be greater than low threshold")
        return v

    @field_validator("high_threshold")
    def validate_high_threshold(cls, v: int, values: Dict[str, int]) -> int:
        """Validate that high threshold is greater than medium threshold."""
        medium = values.get("medium_threshold", 0)
        if v <= medium:
            raise ValueError("High threshold must be greater than medium threshold")
        return v


class RiskScoringResult(BaseModel):
    """Model for risk scoring results."""

    anomaly_id: int
    risk_score: int = Field(0, ge=0, le=100)
    risk_level: str
    risk_factors: List[str]
    explanation: Optional[str] = None
    updated_at: Optional[str] = None


class RiskEngine:
    """Engine for scoring anomalies based on ML and rule-based signals."""

    def __init__(
        self,
        weights: Optional[RiskWeights] = None,
        sensitivity: Optional[RiskSensitivity] = None,
    ):
        """Initialize the risk engine with optional custom weights and sensitivity.

        Args:
            weights: Optional custom weights for risk scoring
            sensitivity: Optional custom sensitivity thresholds
        """
        self.weights = weights or RiskWeights()
        self.sensitivity = sensitivity or RiskSensitivity()

    def update_weights(self, weights: Union[RiskWeights, Dict[str, float]]) -> None:
        """Update the risk scoring weights.

        Args:
            weights: New weights for risk scoring

        Raises:
            ValidationError: If the weights are invalid
        """
        if isinstance(weights, dict):
            weights = RiskWeights(**weights)
        self.weights = weights

    def update_sensitivity(
        self, sensitivity: Union[RiskSensitivity, Dict[str, int]]
    ) -> None:
        """Update the risk sensitivity thresholds.

        Args:
            sensitivity: New sensitivity thresholds

        Raises:
            ValidationError: If the sensitivity thresholds are invalid
        """
        if isinstance(sensitivity, dict):
            sensitivity = RiskSensitivity(**sensitivity)
        self.sensitivity = sensitivity

    def normalized_score(self, ml_score: float, rule_score: int) -> int:
        """Calculate a normalized risk score from ML and rule scores.

        Args:
            ml_score: ML model anomaly score (0.0-1.0)
            rule_score: Rule engine score (0-100)

        Returns:
            int: Normalized risk score (0-100)
        """
        # Convert ML score to 0-100 scale and apply weight
        ml_component = ml_score * 100 * self.weights.ml_weight

        # Apply weight to rule score
        rule_component = rule_score * self.weights.rules_weight

        # Combine scores and round to nearest integer
        combined_score = ml_component + rule_component
        return round(combined_score)

    def get_risk_level(self, score: int) -> str:
        """Determine the risk level based on the score and sensitivity thresholds.

        Args:
            score: Normalized risk score (0-100)

        Returns:
            str: Risk level (negligible, low, medium, high)
        """
        if score < self.sensitivity.low_threshold:
            return "negligible"
        elif score < self.sensitivity.medium_threshold:
            return "low"
        elif score < self.sensitivity.high_threshold:
            return "medium"
        else:
            return "high"

    def get_risk_factors(self, anomaly: Dict[str, Any]) -> List[str]:
        """Extract risk factors from an anomaly.

        Args:
            anomaly: Dictionary containing anomaly data

        Returns:
            List[str]: List of risk factors
        """
        factors = []

        # Check ML score
        ml_score = anomaly.get("ml_score", 0.0)
        if ml_score > 0.8:
            factors.append("High ML anomaly score")
        elif ml_score > 0.6:
            factors.append("Elevated ML anomaly score")

        # Check rule triggers
        rule_triggers = anomaly.get("rule_triggers", [])
        if rule_triggers:
            for trigger in rule_triggers:
                # Convert snake_case to human-readable
                readable_trigger = trigger.replace("_", " ").capitalize()
                factors.append(f"Rule trigger: {readable_trigger}")

        # Check transaction amount
        if "transaction_amount" in anomaly:
            amount = anomaly["transaction_amount"]
            # Add factor for large transactions
            if amount > 10000:
                factors.append(f"Large transaction amount: ${amount:.2f}")

        # Check merchant
        if anomaly.get("merchant_name") and "new_merchant" in rule_triggers:
            factors.append(f"New merchant: {anomaly['merchant_name']}")

        # If no factors were identified, add a generic one
        if not factors:
            factors.append("Unusual transaction pattern")

        return factors

    def generate_explanation(self, anomaly: Dict[str, Any]) -> str:
        """Generate a human-readable explanation for the anomaly.

        Args:
            anomaly: Dictionary containing anomaly data

        Returns:
            str: Human-readable explanation
        """
        # Get the explanation provider
        provider = get_explanation_provider()
        return provider.get_explanation(anomaly)

    def score_anomaly(self, anomaly: Dict[str, Any]) -> RiskScoringResult:
        """Score an anomaly based on ML and rule-based signals.

        Args:
            anomaly: Dictionary containing anomaly data

        Returns:
            RiskScoringResult: Scoring result with risk score, level, and factors

        Raises:
            InvalidRiskConfigurationError: If required fields are missing
        """
        # Check for required fields
        if "ml_score" not in anomaly or "rule_score" not in anomaly:
            raise InvalidRiskConfigurationError(
                "Anomaly must contain 'ml_score' and 'rule_score' fields"
            )

        # Calculate normalized risk score
        ml_score = anomaly["ml_score"]
        rule_score = anomaly["rule_score"]
        risk_score = self.normalized_score(ml_score, rule_score)

        # Determine risk level
        risk_level = self.get_risk_level(risk_score)

        # Extract risk factors
        risk_factors = self.get_risk_factors(anomaly)

        # Create result with anomaly ID
        result = RiskScoringResult(
            anomaly_id=anomaly.get("id", 0),
            risk_score=risk_score,
            risk_level=risk_level,
            risk_factors=risk_factors,
        )

        # Generate explanation if requested
        if anomaly.get("generate_explanation", False):
            # Add computed fields to the anomaly for explanation
            explanation_anomaly = dict(anomaly)
            explanation_anomaly.update(
                {
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "risk_factors": risk_factors,
                }
            )
            result.explanation = self.generate_explanation(explanation_anomaly)

        return result

    def batch_score_anomalies(
        self, anomalies: List[Dict[str, Any]]
    ) -> List[RiskScoringResult]:
        """Score multiple anomalies in batch.

        Args:
            anomalies: List of anomaly dictionaries

        Returns:
            List[RiskScoringResult]: List of scoring results
        """
        results = []
        for anomaly in anomalies:
            try:
                result = self.score_anomaly(anomaly)
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Error scoring anomaly {anomaly.get('id', 'unknown')}: {str(e)}"
                )
                # Skip this anomaly and continue with others

        return results

    async def async_score_anomaly(self, anomaly: Dict[str, Any]) -> RiskScoringResult:
        """Asynchronously score an anomaly.

        This is a wrapper around the synchronous method for use in async contexts.

        Args:
            anomaly: Dictionary containing anomaly data

        Returns:
            RiskScoringResult: Scoring result
        """
        return self.score_anomaly(anomaly)

    async def async_batch_score_anomalies(
        self, anomalies: List[Dict[str, Any]]
    ) -> List[RiskScoringResult]:
        """Asynchronously score multiple anomalies in batch.

        Args:
            anomalies: List of anomaly dictionaries

        Returns:
            List[RiskScoringResult]: List of scoring results
        """
        # Create tasks for each anomaly
        tasks = [self.async_score_anomaly(anomaly) for anomaly in anomalies]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        return [r for r in results if not isinstance(r, Exception)]


# Singleton instance
_risk_engine = None


def get_risk_engine() -> RiskEngine:
    """Get or create the global risk engine instance.

    Returns:
        RiskEngine: The risk engine instance
    """
    global _risk_engine

    if _risk_engine is None:
        _risk_engine = RiskEngine()

    return _risk_engine
