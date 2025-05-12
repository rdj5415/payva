"""Database models for AuditPulse MVP.

This module defines the SQLAlchemy ORM models for the application.
"""
import enum
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
import uuid

from auditpulse_mvp.database.base import Base


class DataSource(enum.Enum):
    """Enum for data sources."""

    QUICKBOOKS = "quickbooks"
    PLAID = "plaid"
    MANUAL = "manual"
    NETSUITE = "netsuite"
    CSV = "csv"


class AnomalyType(enum.Enum):
    """Enum for anomaly types."""

    # More specific anomaly types for better risk assessment
    LARGE_AMOUNT = "large_amount"
    UNUSUAL_VENDOR = "unusual_vendor"
    DUPLICATE_TRANSACTION = "duplicate_transaction"
    STATISTICAL_OUTLIER = "statistical_outlier"
    UNAUTHORIZED_APPROVER = "unauthorized_approver"
    WEEKEND_TRANSACTION = "weekend_transaction"
    ROUND_NUMBER = "round_number"
    OTHER = "other"


class FeedbackType(enum.Enum):
    """Enum for anomaly feedback types."""

    TRUE_POSITIVE = "true_positive"  # This was a real anomaly
    FALSE_POSITIVE = "false_positive"  # This was not a real anomaly
    IGNORE = "ignore"  # Not sure or want to ignore


class Transaction(Base):
    """Transaction model for storing financial transactions from various sources."""

    __tablename__ = "transactions"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Multi-tenant support
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )

    # Basic transaction information
    transaction_id: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[DataSource] = mapped_column(Enum(DataSource), nullable=False)
    source_account_id: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Ensure unique transactions per tenant and source
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "transaction_id", "source", name="uq_transaction_per_tenant_source"
        ),
    )

    # Transaction details
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(10), nullable=False, default="USD")
    description: Mapped[str] = mapped_column(Text, nullable=True)
    category: Mapped[str] = mapped_column(String(255), nullable=True)
    merchant_name: Mapped[str] = mapped_column(String(255), nullable=True)
    
    # Transaction dates
    transaction_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    posting_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # JSON data for source-specific fields
    raw_data: Mapped[Dict] = mapped_column(JSONB, nullable=True)

    # Tracking fields
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    
    # Relationships
    anomalies: Mapped[List["Anomaly"]] = relationship(
        "Anomaly", back_populates="transaction", cascade="all, delete-orphan"
    )


class Anomaly(Base):
    """Anomaly model for storing detected financial transaction anomalies."""

    __tablename__ = "anomalies"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Multi-tenant support
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )

    # Foreign key to Transaction
    transaction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("transactions.id", ondelete="CASCADE"), nullable=False
    )
    
    # Anomaly details
    anomaly_type: Mapped[AnomalyType] = mapped_column(Enum(AnomalyType), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Confidence from detection mechanism (ML or rule)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    
    # ML-specific score (0-1)
    ml_score: Mapped[float] = mapped_column(Float, nullable=True)
    
    # Risk assessment
    risk_score: Mapped[int] = mapped_column(Integer, nullable=True)  # 0-100 normalized score
    risk_level: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # negligible, low, medium, high
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # GPT explanation of risk
    is_flagged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Rule-specific score (0-100)
    rule_score: Mapped[int] = mapped_column(Integer, nullable=True)
    
    # Detection metadata (includes rule name, scoring components, etc.)
    detection_metadata: Mapped[Dict] = mapped_column(JSONB, nullable=True)
    
    # User feedback and resolution
    is_resolved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    feedback: Mapped[Optional[FeedbackType]] = mapped_column(
        Enum(FeedbackType), nullable=True
    )
    feedback_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    resolved_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Tracking fields
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    
    # Notification status
    notification_sent: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    notification_sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    transaction: Mapped["Transaction"] = relationship(
        "Transaction", back_populates="anomalies"
    )


class User(Base):
    """User model for authentication and authorization."""

    __tablename__ = "users"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    
    # Multi-tenant support
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    
    # User information
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(
        String(50), nullable=False, default="viewer"
    )  # admin, auditor, viewer
    
    # Account status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Authentication related
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Notification preferences
    email_notifications: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    slack_notifications: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    sms_notifications: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    phone_number: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    slack_user_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Tracking fields
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Tenant(Base):
    """Tenant model for multi-tenancy support."""

    __tablename__ = "tenants"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    
    # Tenant information
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    
    # Tenant status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    subscription_tier: Mapped[str] = mapped_column(
        String(50), nullable=False, default="standard"
    )  # standard, pro, enterprise
    
    # Tenant configurations and metadata
    metadata: Mapped[Dict] = mapped_column(JSONB, nullable=True)
    
    # Integration settings
    quickbooks_settings: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    plaid_settings: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    
    # Feature flags (tenant specific overrides)
    enable_ml_engine: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    enable_gpt_explanations: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    enable_demo_mode: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Tracking fields
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class AuditLog(Base):
    """Audit log model for tracking important actions in the system."""

    __tablename__ = "audit_logs"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    
    # Multi-tenant support
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    
    # Audit information
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    action: Mapped[str] = mapped_column(String(255), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(255), nullable=False)
    resource_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    details: Mapped[Dict] = mapped_column(JSONB, nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Tracking fields
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    ) 