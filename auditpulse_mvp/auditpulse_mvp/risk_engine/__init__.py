"""Risk Scoring Engine for AuditPulse MVP.

This module provides functionality to score anomalies, integrating
machine learning predictions, rule-based signals, and explanations.
It includes the risk engine, explanation providers, and API endpoints.
"""

from auditpulse_mvp.risk_engine.risk_engine import (
    RiskEngine,
    RiskWeights,
    RiskSensitivity,
    RiskScoringResult,
    get_risk_engine,
    InvalidRiskConfigurationError,
)

from auditpulse_mvp.risk_engine.explanations import (
    RiskExplanationProvider,
    GPTExplanationProvider,
    get_explanation_provider,
)

# Define public API
__all__ = [
    # Risk Engine
    "RiskEngine",
    "RiskWeights",
    "RiskSensitivity",
    "RiskScoringResult",
    "get_risk_engine",
    "InvalidRiskConfigurationError",
    # Explanation Providers
    "RiskExplanationProvider",
    "GPTExplanationProvider",
    "get_explanation_provider",
]
