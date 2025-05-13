"""Risk Explanation Provider for AuditPulse MVP.

This module provides functionality to generate explanations for anomalies
using machine learning and GPT models. It includes classes for different
explanation providers and utilities for generating contextual risk explanations.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import json

from pydantic import BaseModel

from auditpulse_mvp.utils.settings import settings

logger = logging.getLogger(__name__)


class ExplanationRequest(BaseModel):
    """Model for explanation generation requests."""

    anomaly_id: int
    transaction_amount: float
    transaction_type: str
    merchant_name: str
    transaction_date: str
    ml_score: float
    rule_score: int
    risk_score: int
    risk_level: str
    risk_factors: List[str]

    # Optional context
    category: Optional[str] = None
    description: Optional[str] = None
    rule_triggers: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {k: v for k, v in self.dict().items() if v is not None}


class RiskExplanationProvider:
    """Base class for explanation providers."""

    def get_explanation(self, anomaly: Dict[str, Any]) -> str:
        """Generate an explanation for the provided anomaly.

        Args:
            anomaly: Dictionary containing anomaly data

        Returns:
            str: Human-readable explanation for the anomaly
        """
        try:
            # Convert anomaly dict to ExplanationRequest model for validation
            request = self._prepare_request(anomaly)

            # Generate the explanation based on the validated data
            explanation = self._generate_explanation(request)
            return explanation

        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return self._get_fallback_explanation(anomaly)

    def _prepare_request(self, anomaly: Dict[str, Any]) -> ExplanationRequest:
        """Prepare the explanation request from the anomaly data.

        Args:
            anomaly: Dictionary containing anomaly data

        Returns:
            ExplanationRequest: Validated request model
        """
        # Extract required fields with defaults if missing
        data = {
            "anomaly_id": anomaly.get("id", 0),
            "transaction_amount": anomaly.get("transaction_amount", 0.0),
            "transaction_type": anomaly.get("transaction_type", "unknown"),
            "merchant_name": anomaly.get("merchant_name", "unknown"),
            "transaction_date": anomaly.get("transaction_date", "unknown"),
            "ml_score": anomaly.get("ml_score", 0.0),
            "rule_score": anomaly.get("rule_score", 0),
            "risk_score": anomaly.get("risk_score", 0),
            "risk_level": anomaly.get("risk_level", "unknown"),
            "risk_factors": anomaly.get("risk_factors", []),
            # Optional fields
            "category": anomaly.get("category"),
            "description": anomaly.get("description"),
            "rule_triggers": anomaly.get("rule_triggers"),
        }

        return ExplanationRequest(**data)

    def _generate_explanation(self, request: ExplanationRequest) -> str:
        """Generate the explanation text based on the request.

        This is the base implementation that provides a simple structured explanation.
        Subclasses should override this method to provide more sophisticated explanations.

        Args:
            request: The validated explanation request

        Returns:
            str: Human-readable explanation for the anomaly
        """
        # Basic explanation based on risk level and factors
        risk_level = request.risk_level.capitalize()
        explanation = f"This transaction has been flagged with {risk_level} risk ({request.risk_score}/100)."

        # Add risk factors if available
        if request.risk_factors:
            factors = ", ".join(request.risk_factors)
            explanation += f" Risk factors include: {factors}."

        # Add transaction details
        explanation += f" The {request.transaction_type} transaction of ${request.transaction_amount:.2f}"
        explanation += f" with {request.merchant_name} on {request.transaction_date}"
        explanation += " exhibited unusual characteristics compared to normal patterns."

        return explanation

    def _get_fallback_explanation(self, anomaly: Dict[str, Any]) -> str:
        """Provide a fallback explanation when the main generation fails.

        Args:
            anomaly: Dictionary containing anomaly data

        Returns:
            str: Simple fallback explanation
        """
        return (
            f"Transaction with {anomaly.get('merchant_name', 'unknown merchant')} "
            f"for ${anomaly.get('transaction_amount', 0):.2f} "
            f"has been flagged as unusual based on risk analysis."
        )


class GPTExplanationProvider(RiskExplanationProvider):
    """Explanation provider that uses GPT for generating natural language explanations."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the GPT explanation provider.

        Args:
            api_key: Optional API key for OpenAI. If not provided, uses settings.
        """
        super().__init__()
        self.api_key = api_key or (
            settings.OPENAI_API_KEY.get_secret_value()
            if settings.OPENAI_API_KEY
            else None
        )

        if not self.api_key:
            logger.warning("No OpenAI API key provided, using fallback explanations")

    def _generate_explanation(self, request: ExplanationRequest) -> str:
        """Generate an explanation using GPT.

        Args:
            request: The validated explanation request

        Returns:
            str: Human-readable explanation for the anomaly
        """
        # Check if GPT explanations are enabled
        if not settings.enable_gpt_explanations or not self.api_key:
            return super()._generate_explanation(request)

        try:
            # This would be implemented with actual OpenAI API calls
            # For now, we'll just provide a more detailed template-based explanation
            return self._enhanced_template_explanation(request)
        except Exception as e:
            logger.error(f"Error generating GPT explanation: {str(e)}")
            return super()._generate_explanation(request)

    def _enhanced_template_explanation(self, request: ExplanationRequest) -> str:
        """Generate a more detailed template-based explanation as a fallback.

        Args:
            request: The validated explanation request

        Returns:
            str: Enhanced template-based explanation
        """
        risk_level_descriptions = {
            "negligible": "extremely low",
            "low": "relatively low",
            "medium": "moderate",
            "high": "significant",
        }

        level_desc = risk_level_descriptions.get(request.risk_level.lower(), "unknown")

        # Start with a summary
        explanation = (
            f"This {request.transaction_type} transaction of ${request.transaction_amount:.2f} "
            f"with {request.merchant_name} on {request.transaction_date} "
            f"has been assigned a {level_desc} risk score of {request.risk_score}/100."
        )

        # Add details about the risk factors
        if request.risk_factors:
            explanation += "\n\nThe following risk factors were identified:"
            for factor in request.risk_factors:
                explanation += f"\n- {factor}"

        # Add ML context if the score is significant
        if request.ml_score > 0.5:
            explanation += (
                f"\n\nOur machine learning model detected unusual patterns "
                f"in this transaction (anomaly score: {request.ml_score:.2f})."
            )

        # Add rule context if available
        if request.rule_triggers:
            explanation += "\n\nThe transaction triggered the following rules:"
            for rule in request.rule_triggers:
                explanation += f"\n- {rule.replace('_', ' ').title()}"

        # Add recommendation based on risk level
        if request.risk_level.lower() in ["high", "medium"]:
            explanation += (
                "\n\nRecommendation: This transaction should be reviewed manually."
            )
        else:
            explanation += "\n\nRecommendation: This transaction is likely normal but flagged for awareness."

        return explanation


# Singleton instance
_explanation_provider = None


def get_explanation_provider() -> RiskExplanationProvider:
    """Get or create the global explanation provider instance.

    Returns:
        RiskExplanationProvider: The explanation provider instance
    """
    global _explanation_provider

    if _explanation_provider is None:
        if settings.enable_gpt_explanations:
            _explanation_provider = GPTExplanationProvider()
        else:
            _explanation_provider = RiskExplanationProvider()

    return _explanation_provider
