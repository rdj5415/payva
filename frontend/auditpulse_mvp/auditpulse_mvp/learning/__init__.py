"""Continuous learning module for AuditPulse MVP.

This module provides functions for updating machine learning models and rule
parameters based on user feedback on anomalies.
"""

from .feedback_learning import (
    FeedbackLearner,
    update_thresholds_from_feedback,
    get_feedback_learner,
)

__all__ = [
    "FeedbackLearner",
    "update_thresholds_from_feedback",
    "get_feedback_learner",
]
