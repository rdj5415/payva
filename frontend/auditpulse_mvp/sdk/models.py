"""Pydantic models for the AuditPulse SDK.

This module provides Pydantic models for data validation and serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected."""

    LARGE_AMOUNT = "large_amount"
    UNUSUAL_PATTERN = "unusual_pattern"
    DUPLICATE = "duplicate"
    UNAPPROVED_VENDOR = "unapproved_vendor"
    SUSPICIOUS_TIMING = "suspicious_timing"
    CATEGORY_MISMATCH = "category_mismatch"
    ROUND_AMOUNT = "round_amount"
    HIGH_FREQUENCY = "high_frequency"


class AnomalyStatus(str, Enum):
    """Status of an anomaly."""

    OPEN = "open"
    IN_REVIEW = "in_review"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"
    IGNORED = "ignored"


class RiskLevel(str, Enum):
    """Risk levels for anomalies."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationType(str, Enum):
    """Types of notifications."""

    ANOMALY_DETECTED = "anomaly_detected"
    ANOMALY_RESOLVED = "anomaly_resolved"
    RISK_THRESHOLD = "risk_threshold"
    SYSTEM_ALERT = "system_alert"
    USER_ACTION = "user_action"


class NotificationChannel(str, Enum):
    """Channels for sending notifications."""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"


class FeedbackType(str, Enum):
    """Types of feedback for anomalies."""

    VALID = "valid"
    FALSE_POSITIVE = "false_positive"
    NEEDS_REVIEW = "needs_review"
    SUGGESTION = "suggestion"


T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response model."""

    items: List[T]
    total: int
    page: int
    size: int
    pages: int


class Transaction(BaseModel):
    """Transaction model."""

    id: UUID
    tenant_id: UUID
    transaction_id: str
    amount: float
    currency: str
    description: str
    category: str
    merchant_name: str
    transaction_date: datetime
    source: str
    source_account_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        """Pydantic config."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class Anomaly(BaseModel):
    """Anomaly model."""

    id: UUID
    tenant_id: UUID
    transaction_id: str
    type: AnomalyType
    risk_score: float
    risk_level: RiskLevel
    amount: float
    description: str
    status: AnomalyStatus
    resolution: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        """Pydantic config."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class User(BaseModel):
    """User model."""

    id: UUID
    email: str
    first_name: str
    last_name: str
    role: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        """Pydantic config."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class Tenant(BaseModel):
    """Tenant model."""

    id: UUID
    name: str
    domain: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        """Pydantic config."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class RiskScore(BaseModel):
    """Risk score model."""

    score: float = Field(ge=0.0, le=1.0)
    level: RiskLevel
    factors: List[Dict[str, Any]]
    created_at: datetime

    class Config:
        """Pydantic config."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class Feedback(BaseModel):
    """Feedback model."""

    type: FeedbackType
    comment: Optional[str]
    rating: Optional[int] = Field(ge=1, le=5)
    created_at: datetime

    class Config:
        """Pydantic config."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class Notification(BaseModel):
    """Notification model."""

    id: UUID
    tenant_id: UUID
    type: NotificationType
    channel: NotificationChannel
    title: str
    message: str
    data: Dict[str, Any]
    is_read: bool
    created_at: datetime

    class Config:
        """Pydantic config."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class NotificationSettings(BaseModel):
    """Notification settings model."""

    channels: List[NotificationChannel]
    types: List[NotificationType]
    email: Optional[str]
    slack_webhook: Optional[str]
    sms_number: Optional[str]
    webhook_url: Optional[str]
    frequency: str
    created_at: datetime
    updated_at: datetime

    class Config:
        """Pydantic config."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
