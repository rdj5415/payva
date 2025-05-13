"""Risk Engine for AuditPulse MVP.

This module implements a risk scoring decision layer that combines
multiple signals (rules, ML, GPT) to produce a final risk score.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import Depends, HTTPException, status
from pydantic import BaseModel, Field, validator
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.database.models import Anomaly, AnomalyType, Tenant, Transaction
from auditpulse_mvp.gpt_engine.gpt_engine import GPTEngine, get_gpt_engine
from auditpulse_mvp.ml_engine.ml_engine import MLEngine, get_ml_engine
from auditpulse_mvp.rules_engine.rules_engine import RulesEngine
from auditpulse_mvp.utils.settings import settings

# Configure logging
logger = logging.getLogger(__name__)


class RiskWeights(BaseModel):
    """Configuration for risk score component weights."""

    # Weights for each component (0-100)
    rules_weight: float = Field(40.0, ge=0.0, le=100.0)
    ml_weight: float = Field(40.0, ge=0.0, le=100.0)
    gpt_weight: float = Field(20.0, ge=0.0, le=100.0)

    # Thresholds for risk levels
    low_threshold: float = Field(25.0, ge=0.0, le=100.0)
    medium_threshold: float = Field(50.0, ge=0.0, le=100.0)
    high_threshold: float = Field(75.0, ge=0.0, le=100.0)

    @validator("ml_weight", "gpt_weight", "rules_weight")
    def weights_must_sum_to_100(cls, v, values):
        """Validate that the weights sum to 100."""
        if "rules_weight" in values:
            total = values["rules_weight"]
            if "ml_weight" in values:
                total += values["ml_weight"]
            if v is not None:  # Current field being validated
                total += v

            # Allow a small tolerance for floating point arithmetic
            if (
                not (99.99 <= total <= 100.01) and len(values) == 2
            ):  # Only check when all 3 weights are set
                raise ValueError(f"Weights must sum to 100, got {total}")
        return v

    @validator("medium_threshold")
    def medium_threshold_must_be_greater_than_low(cls, v, values):
        """Validate that the medium threshold is greater than the low threshold."""
        if "low_threshold" in values and v <= values["low_threshold"]:
            raise ValueError(
                f"Medium threshold ({v}) must be greater than low threshold ({values['low_threshold']})"
            )
        return v

    @validator("high_threshold")
    def high_threshold_must_be_greater_than_medium(cls, v, values):
        """Validate that the high threshold is greater than the medium threshold."""
        if "medium_threshold" in values and v <= values["medium_threshold"]:
            raise ValueError(
                f"High threshold ({v}) must be greater than medium threshold ({values['medium_threshold']})"
            )
        return v


class RiskConfig(BaseModel):
    """Configuration for risk scoring."""

    # Tenant ID
    tenant_id: uuid.UUID

    # Risk weights
    weights: RiskWeights = Field(default_factory=RiskWeights)

    # GPT confidence threshold (0-1)
    gpt_confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)

    # Minimum rule score to trigger an anomaly (0-1)
    min_rule_score: float = Field(0.3, ge=0.0, le=1.0)

    # Minimum ML score to trigger an anomaly (0-1)
    min_ml_score: float = Field(0.6, ge=0.0, le=1.0)

    # Baseline risk added to all transactions
    baseline_risk: float = Field(0.0, ge=0.0, le=50.0)

    # Whether to use GPT for explanations
    use_gpt: bool = True

    # Whether to use ML for scoring
    use_ml: bool = True


class RiskLevel(str):
    """Risk level enumeration."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskResult(BaseModel):
    """Result of a risk assessment."""

    # Overall risk score (0-100)
    score: float

    # Risk level
    level: str

    # Component scores (0-100)
    rules_score: float
    ml_score: float
    gpt_confidence: Optional[float] = None

    # Flags from rules
    flags: List[Dict[str, Any]] = []

    # Explanation from GPT
    explanation: Optional[str] = None

    # Configuration used
    config: RiskConfig


class RiskEngine:
    """Risk scoring decision engine that combines multiple signals."""

    def __init__(
        self,
        db_session: AsyncSession = Depends(get_db_session),
        rules_engine: Optional[RulesEngine] = None,
        ml_engine: Optional[MLEngine] = None,
        gpt_engine: Optional[GPTEngine] = None,
    ):
        """Initialize the risk engine.

        Args:
            db_session: The database session.
            rules_engine: Optional rules engine instance.
            ml_engine: Optional ML engine instance.
            gpt_engine: Optional GPT engine instance.
        """
        self.db_session = db_session
        self.rules_engine = rules_engine or RulesEngine(db_session)
        self.ml_engine = ml_engine or get_ml_engine()
        self.gpt_engine = gpt_engine or get_gpt_engine()
        self._tenant_configs: Dict[str, RiskConfig] = {}

    async def get_config(self, tenant_id: uuid.UUID) -> RiskConfig:
        """Get the risk configuration for a tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            RiskConfig: The risk configuration.
        """
        # Check cache first
        tenant_key = str(tenant_id)
        if tenant_key in self._tenant_configs:
            return self._tenant_configs[tenant_key]

        # Get tenant from database
        stmt = select(Tenant).where(Tenant.id == tenant_id)
        result = await self.db_session.execute(stmt)
        tenant = result.scalar_one_or_none()

        if not tenant:
            # Return default config for unknown tenant
            config = RiskConfig(tenant_id=tenant_id)
            self._tenant_configs[tenant_key] = config
            return config

        # Extract risk config from tenant settings
        risk_settings = tenant.risk_settings or {}

        # Create weights
        weights_data = risk_settings.get("weights", {})
        weights = RiskWeights(
            rules_weight=weights_data.get("rules_weight", 40.0),
            ml_weight=weights_data.get("ml_weight", 40.0),
            gpt_weight=weights_data.get("gpt_weight", 20.0),
            low_threshold=weights_data.get("low_threshold", 25.0),
            medium_threshold=weights_data.get("medium_threshold", 50.0),
            high_threshold=weights_data.get("high_threshold", 75.0),
        )

        # Create config
        config = RiskConfig(
            tenant_id=tenant_id,
            weights=weights,
            gpt_confidence_threshold=risk_settings.get("gpt_confidence_threshold", 0.5),
            min_rule_score=risk_settings.get("min_rule_score", 0.3),
            min_ml_score=risk_settings.get("min_ml_score", 0.6),
            baseline_risk=risk_settings.get("baseline_risk", 0.0),
            use_gpt=risk_settings.get("use_gpt", True),
            use_ml=risk_settings.get("use_ml", True),
        )

        # Cache config
        self._tenant_configs[tenant_key] = config

        return config

    async def update_config(self, config: RiskConfig) -> bool:
        """Update the risk configuration for a tenant.

        Args:
            config: The risk configuration.

        Returns:
            bool: True if successful, False otherwise.
        """
        tenant_id = config.tenant_id
        tenant_key = str(tenant_id)

        # Update cache
        self._tenant_configs[tenant_key] = config

        # Convert to dict for database
        risk_settings = {
            "weights": {
                "rules_weight": config.weights.rules_weight,
                "ml_weight": config.weights.ml_weight,
                "gpt_weight": config.weights.gpt_weight,
                "low_threshold": config.weights.low_threshold,
                "medium_threshold": config.weights.medium_threshold,
                "high_threshold": config.weights.high_threshold,
            },
            "gpt_confidence_threshold": config.gpt_confidence_threshold,
            "min_rule_score": config.min_rule_score,
            "min_ml_score": config.min_ml_score,
            "baseline_risk": config.baseline_risk,
            "use_gpt": config.use_gpt,
            "use_ml": config.use_ml,
        }

        try:
            # Update tenant in database
            stmt = (
                update(Tenant)
                .where(Tenant.id == tenant_id)
                .values(risk_settings=risk_settings)
            )
            await self.db_session.execute(stmt)
            await self.db_session.commit()

            logger.info(f"Updated risk configuration for tenant {tenant_id}")
            return True

        except Exception as e:
            logger.error(
                f"Error updating risk configuration for tenant {tenant_id}: {e}"
            )
            await self.db_session.rollback()
            return False

    async def evaluate_transaction(
        self, tenant_id: uuid.UUID, transaction: Transaction
    ) -> RiskResult:
        """Evaluate the risk of a transaction.

        Args:
            tenant_id: The tenant ID.
            transaction: The transaction to evaluate.

        Returns:
            RiskResult: The risk evaluation result.
        """
        # Get configuration
        config = await self.get_config(tenant_id)

        # Get rule flags and score
        flags = await self.rules_engine.flags(tenant_id, transaction)
        rules_score = await self.rules_engine.score(tenant_id, transaction)

        # Scale rules score to 0-100
        rules_score_scaled = rules_score * 100

        # Get ML score if enabled
        ml_score_scaled = 0.0
        if config.use_ml:
            try:
                ml_score = await self.ml_engine.score(tenant_id, transaction)
                ml_score_scaled = ml_score * 100
            except Exception as e:
                logger.error(f"Error getting ML score: {e}")
                ml_score_scaled = 0.0

        # Get GPT confidence if enabled and there are flags
        gpt_confidence = None
        explanation = None
        if config.use_gpt and flags:
            try:
                explanation = await self.gpt_engine.generate_explanation(
                    transaction, flags
                )

                # For now, we don't have a true confidence score from GPT
                # We can use a heuristic based on explanation length and specificity
                # This is a placeholder - in production you'd want a more sophisticated measure
                gpt_confidence = 0.7  # Default moderate confidence

            except Exception as e:
                logger.error(f"Error getting GPT explanation: {e}")
                explanation = f"Error generating explanation: {str(e)}"
                gpt_confidence = 0.0

        # Calculate weighted score
        weighted_score = 0.0
        weights_sum = 0.0

        # Add rules component
        weighted_score += rules_score_scaled * (config.weights.rules_weight / 100)
        weights_sum += config.weights.rules_weight

        # Add ML component if enabled
        if config.use_ml:
            weighted_score += ml_score_scaled * (config.weights.ml_weight / 100)
            weights_sum += config.weights.ml_weight

        # Add GPT component if enabled and confidence available
        if config.use_gpt and gpt_confidence is not None:
            gpt_score_scaled = gpt_confidence * 100
            weighted_score += gpt_score_scaled * (config.weights.gpt_weight / 100)
            weights_sum += config.weights.gpt_weight

        # Normalize
        if weights_sum > 0:
            final_score = (weighted_score / weights_sum) * 100
        else:
            final_score = 0.0

        # Add baseline risk
        final_score = min(100.0, final_score + config.baseline_risk)

        # Determine risk level
        risk_level = RiskLevel.LOW
        if final_score >= config.weights.high_threshold:
            risk_level = RiskLevel.HIGH
        elif final_score >= config.weights.medium_threshold:
            risk_level = RiskLevel.MEDIUM

        # Create result
        return RiskResult(
            score=final_score,
            level=risk_level,
            rules_score=rules_score_scaled,
            ml_score=ml_score_scaled,
            gpt_confidence=gpt_confidence,
            flags=flags,
            explanation=explanation,
            config=config,
        )

    async def evaluate_and_store(
        self, tenant_id: uuid.UUID, transaction: Transaction
    ) -> Anomaly:
        """Evaluate a transaction and store an anomaly if risky.

        Args:
            tenant_id: The tenant ID.
            transaction: The transaction to evaluate.

        Returns:
            Optional[Anomaly]: The created anomaly, or None if not risky.

        Raises:
            HTTPException: If there's an error evaluating the transaction.
        """
        try:
            # Evaluate transaction
            result = await self.evaluate_transaction(tenant_id, transaction)

            # Check if transaction is anomalous
            is_anomalous = False
            anomaly_types = []

            # Check rules score
            if result.rules_score >= (result.config.min_rule_score * 100):
                is_anomalous = True
                anomaly_types.append(AnomalyType.RULE_BASED)

            # Check ML score
            if result.ml_score >= (result.config.min_ml_score * 100):
                is_anomalous = True
                anomaly_types.append(AnomalyType.ML_DETECTED)

            # If anomalous, create anomaly
            if is_anomalous:
                # Determine primary anomaly type
                primary_type = (
                    anomaly_types[0] if anomaly_types else AnomalyType.RULE_BASED
                )

                # Create anomaly
                anomaly = Anomaly(
                    tenant_id=tenant_id,
                    transaction_id=transaction.id,
                    anomaly_type=primary_type,
                    score=result.score,
                    flags=result.flags,
                    explanation=result.explanation,
                    is_resolved=False,
                    resolution_notes=None,
                )

                # Store in database
                self.db_session.add(anomaly)
                await self.db_session.commit()

                return anomaly

            return None

        except Exception as e:
            logger.error(f"Error evaluating transaction {transaction.id}: {e}")
            await self.db_session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error evaluating transaction: {str(e)}",
            )


# Singleton instance for dependency injection
risk_engine = RiskEngine()


def get_risk_engine() -> RiskEngine:
    """Dependency function for FastAPI to get the risk engine.

    Returns:
        RiskEngine: The risk engine instance.
    """
    return risk_engine
