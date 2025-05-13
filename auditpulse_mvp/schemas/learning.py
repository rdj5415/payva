"""Schema models for feedback learning and continuous model improvement."""

from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from pydantic import BaseModel, Field


class SensitivityLevel(str, Enum):
    """Sensitivity levels for anomaly detection."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CUSTOM = "custom"


class SensitivityConfiguration(BaseModel):
    """Model for sensitivity configuration settings."""

    sensitivity_level: SensitivityLevel = Field(
        ..., description="Overall sensitivity level"
    )
    risk_engine: Dict[str, Any] = Field(..., description="Risk engine configuration")
    rules: Dict[str, Any] = Field(..., description="Rule-specific configuration")


class SensitivityConfigResponse(BaseModel):
    """Schema for sensitivity configuration response."""

    id: str = Field(
        ..., description="The unique identifier for the sensitivity configuration"
    )
    rule_type: str = Field(
        ...,
        description="The type of rule (e.g., 'amount_threshold', 'frequency_threshold')",
    )
    sensitivity: float = Field(..., description="The sensitivity value (0.0 to 1.0)")
    last_updated: datetime = Field(..., description="The timestamp of the last update")


class LearningStatusResponse(BaseModel):
    """Schema for learning status response."""

    tenant_id: str = Field(..., description="The tenant ID")
    sensitivity_configs: List[SensitivityConfigResponse] = Field(
        ..., description="List of sensitivity configurations"
    )


class FeedbackAnalysisResult(BaseModel):
    """Schema for feedback analysis result."""

    total: int = Field(..., description="Total number of anomalies with feedback")
    confirm: int = Field(..., description="Number of confirmed anomalies")
    dismiss: int = Field(..., description="Number of dismissed anomalies")


class ConfigUpdateResult(BaseModel):
    """Schema for configuration update result."""

    sensitivity_before: float = Field(
        ..., description="The sensitivity value before the update"
    )
    sensitivity_after: float = Field(
        ..., description="The sensitivity value after the update"
    )


class RetrainingStatusResult(BaseModel):
    """Schema for retraining status result."""

    status: str = Field(..., description="The status of the retraining process")
    message: Optional[str] = Field(
        None, description="Additional message about the retraining process"
    )


class SingleTenantLearningResult(BaseModel):
    """Schema for single tenant learning result."""

    status: str = Field(..., description="The status of the learning process")
    tenant_id: str = Field(..., description="The tenant ID")
    feedback_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Analysis of feedback data"
    )
    config_updates: Optional[Dict[str, Any]] = Field(
        None, description="Updates to sensitivity configurations"
    )
    retraining_status: Optional[Dict[str, Any]] = Field(
        None, description="Status of model retraining"
    )
    error: Optional[str] = Field(None, description="Error message, if any")


class MultiTenantLearningResult(BaseModel):
    """Schema for multi-tenant learning result."""

    status: str = Field(..., description="The overall status of the learning process")
    total_tenants: int = Field(..., description="The total number of tenants processed")
    results: List[Dict[str, Any]] = Field(..., description="Results for each tenant")
    error: Optional[str] = Field(None, description="Error message, if any")


class LearningTriggerResponse(BaseModel):
    """Response model for triggering the learning process."""

    status: str = Field(
        ..., description="Status of the learning process (success, skipped, error)"
    )
    processed_count: int = Field(..., description="Number of anomalies processed")
    false_positive_rate: Optional[float] = Field(
        None, description="Rate of false positives in processed anomalies"
    )
    true_positive_rate: Optional[float] = Field(
        None, description="Rate of true positives in processed anomalies"
    )
    config_updated: Optional[bool] = Field(
        None, description="Whether the sensitivity configuration was updated"
    )
    model_retrained: Optional[bool] = Field(
        None, description="Whether the ML model was retrained"
    )
    model_info: Optional[Dict[str, Any]] = Field(
        None, description="Information about the retrained model, if any"
    )
    reason: Optional[str] = Field(None, description="Reason for skipping or error")
    error: Optional[str] = Field(None, description="Error details if status is error")


class ScheduledLearning(BaseModel):
    """Model for scheduled learning configuration."""

    enabled: bool = Field(True, description="Whether scheduled learning is enabled")
    frequency: str = Field(
        "daily", description="Frequency of learning (daily, weekly, etc.)"
    )
    hour: int = Field(3, description="Hour of day to run learning (0-23)")
    minute: int = Field(0, description="Minute of hour to run learning (0-59)")
    next_run: Optional[str] = Field(
        None, description="ISO timestamp of next scheduled run"
    )


class LastLearningRun(BaseModel):
    """Model for information about the last learning run."""

    timestamp: str = Field(..., description="ISO timestamp of when the run occurred")
    status: str = Field(
        ..., description="Status of the learning run (success, skipped, error)"
    )
    processed_count: int = Field(..., description="Number of anomalies processed")
    rules_updated: Optional[int] = Field(None, description="Number of rules updated")
    models_updated: Optional[int] = Field(
        None, description="Number of ML models updated"
    )
    false_positive_rate: Optional[float] = Field(
        None, description="Overall false positive rate"
    )
    true_positive_rate: Optional[float] = Field(
        None, description="Overall true positive rate"
    )
    error: Optional[str] = Field(None, description="Error details if status is error")


class LearningStatus(BaseModel):
    """Response model for learning status."""

    last_learning_run: Optional[LastLearningRun] = Field(
        None, description="Information about the last learning run"
    )
    scheduled_learning: ScheduledLearning = Field(
        ..., description="Scheduled learning configuration"
    )


class FeedbackTypeCounts(BaseModel):
    """Model for counts of different feedback types."""

    false_positive: int = Field(0, description="Number of false positives")
    true_positive: int = Field(0, description="Number of true positives")
    needs_investigation: int = Field(
        0, description="Number of anomalies marked for investigation"
    )
    other: int = Field(0, description="Number of other feedback types")
    total: int = Field(0, description="Total feedback count")
    false_positive_rate: Optional[float] = Field(
        None, description="False positive rate"
    )
    true_positive_rate: Optional[float] = Field(None, description="True positive rate")


class LearningAdjustments(BaseModel):
    """Model for information about learning-based adjustments."""

    rules_adjusted: int = Field(
        0, description="Number of rules adjusted based on feedback"
    )
    ml_models_adjusted: int = Field(
        0, description="Number of ML models adjusted based on feedback"
    )
    last_adjustment_date: Optional[str] = Field(
        None, description="ISO timestamp of last adjustment"
    )


class FeedbackStatistics(BaseModel):
    """Response model for feedback statistics."""

    total_feedback: int = Field(..., description="Total number of feedback items")
    rules_based: FeedbackTypeCounts = Field(
        ..., description="Statistics for rules-based anomalies"
    )
    ml_based: FeedbackTypeCounts = Field(
        ..., description="Statistics for ML-based anomalies"
    )
    combined: FeedbackTypeCounts = Field(
        ..., description="Combined statistics for all anomalies"
    )
    learning_adjustments: LearningAdjustments = Field(
        ..., description="Information about learning-based adjustments"
    )


class LearningScheduleUpdate(BaseModel):
    """Request model for updating the learning schedule."""

    enabled: bool = Field(..., description="Whether scheduled learning is enabled")
    hour: int = Field(
        ..., ge=0, le=23, description="Hour of day to run learning (0-23)"
    )
    minute: int = Field(
        ..., ge=0, le=59, description="Minute of hour to run learning (0-59)"
    )


class SensitivityUpdateResponse(BaseModel):
    """Response model for sensitivity update."""

    status: str = Field(
        ..., description="Status of the update (success, skipped, error)"
    )
    message: Optional[str] = Field(
        None, description="Human-readable message about the update"
    )
    update_applied: bool = Field(..., description="Whether the sensitivity was updated")
    previous_level: Optional[str] = Field(
        None, description="Previous sensitivity level"
    )
    new_level: Optional[str] = Field(None, description="New sensitivity level")
    false_positive_rate: Optional[float] = Field(
        None, description="False positive rate that led to the update"
    )
    true_positive_rate: Optional[float] = Field(
        None, description="True positive rate that led to the update"
    )
    reason: Optional[str] = Field(None, description="Reason for skipping or error")
