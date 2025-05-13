"""Database models for AuditPulse MVP."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, List, Union, cast
from uuid import UUID, uuid4

from sqlalchemy import (
    Column,
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Integer,
    String,
    JSON,
    Float,
    Boolean,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.ext.declarative import declarative_base

from auditpulse_mvp.database.base_class import Base

Base = declarative_base()


class TaskStatus(str, Enum):
    """Task status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class TaskLog(Base):
    """Background task execution log."""

    __tablename__ = "task_logs"

    id: Mapped[UUID] = mapped_column(PGUUID(), primary_key=True, default=uuid4)
    tenant_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(), ForeignKey("tenants.id"), nullable=True
    )
    task_name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Relationships
    tenant: Mapped[Optional["Tenant"]] = relationship(
        "Tenant", back_populates="task_logs"
    )

    def __repr__(self) -> str:
        return f"<TaskLog {self.task_name} ({self.id})>"


class ModelVersion(Base):
    """ML model version information."""

    __tablename__ = "model_versions"

    id: Mapped[UUID] = mapped_column(PGUUID(), primary_key=True, default=uuid4)
    model_type: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[str] = mapped_column(String, nullable=False)
    model_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    activated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    deactivated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    performances: Mapped[List["ModelPerformance"]] = relationship(
        "ModelPerformance", back_populates="model_version"
    )

    def __repr__(self) -> str:
        return f"<ModelVersion {self.model_type} v{self.version} ({self.id})>"


class ModelPerformance(Base):
    """ML model performance metrics."""

    __tablename__ = "model_performances"

    id: Mapped[UUID] = mapped_column(PGUUID(), primary_key=True, default=uuid4)
    model_version_id: Mapped[UUID] = mapped_column(
        PGUUID(), ForeignKey("model_versions.id"), nullable=False
    )
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    dataset_size: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluation_time: Mapped[float] = mapped_column(Float, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    # Relationships
    model_version: Mapped["ModelVersion"] = relationship(
        "ModelVersion", back_populates="performances"
    )

    def __repr__(self) -> str:
        return f"<ModelPerformance {self.model_version.model_type} v{self.model_version.version} ({self.id})>"


class FinancialInstitution(Base):
    """Financial institution connected via Plaid."""

    __tablename__ = "financial_institutions"

    id: Mapped[UUID] = mapped_column(PGUUID(), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        PGUUID(), ForeignKey("users.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    plaid_access_token: Mapped[str] = mapped_column(String, nullable=False)
    plaid_item_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    plaid_institution_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    last_updated: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="financial_institutions")
    accounts: Mapped[List["FinancialAccount"]] = relationship(
        "FinancialAccount", back_populates="institution"
    )

    def __repr__(self):
        return f"<FinancialInstitution {self.name} ({self.id})>"


class FinancialAccount(Base):
    """Financial account from a financial institution."""

    __tablename__ = "financial_accounts"

    id: Mapped[UUID] = mapped_column(PGUUID(), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        PGUUID(), ForeignKey("users.id"), nullable=False, index=True
    )
    institution_id: Mapped[UUID] = mapped_column(
        PGUUID(), ForeignKey("financial_institutions.id"), nullable=False
    )
    plaid_account_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    official_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    type: Mapped[str] = mapped_column(String, nullable=False)
    subtype: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    mask: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    balances: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    last_updated: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="financial_accounts")
    institution: Mapped["FinancialInstitution"] = relationship(
        "FinancialInstitution", back_populates="accounts"
    )
    transactions: Mapped[List["FinancialTransaction"]] = relationship(
        "FinancialTransaction", back_populates="account"
    )

    def __repr__(self):
        return f"<FinancialAccount {self.name} ({self.id})>"


class FinancialTransaction(Base):
    """Financial transaction from a financial account."""

    __tablename__ = "financial_transactions"

    id: Mapped[UUID] = mapped_column(PGUUID(), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        PGUUID(), ForeignKey("users.id"), nullable=False, index=True
    )
    account_id: Mapped[UUID] = mapped_column(
        PGUUID(), ForeignKey("financial_accounts.id"), nullable=False
    )
    transaction_id: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    merchant_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    pending: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    category: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    category_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    transaction_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    payment_channel: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    location: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    last_updated: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="financial_transactions")
    account: Mapped["FinancialAccount"] = relationship(
        "FinancialAccount", back_populates="financial_transactions"
    )
    anomalies: Mapped[List["Anomaly"]] = relationship(
        "Anomaly", back_populates="transaction"
    )

    def __repr__(self):
        return f"<FinancialTransaction {self.name} ({self.id})>"


class UserRole(str, SQLEnum):
    """User role enum."""

    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class NotificationChannel(str, SQLEnum):
    """Notification channel enum."""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"


class NotificationStatus(str, SQLEnum):
    """Notification status enum."""

    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"


class User(Base):
    """User information."""

    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(PGUUID(), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(
        PGUUID(), ForeignKey("tenants.id"), nullable=False, index=True
    )
    email: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    role: Mapped[UserRole] = mapped_column(
        SQLEnum(UserRole), nullable=False, default=UserRole.USER
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="users")
    financial_institutions: Mapped[List["FinancialInstitution"]] = relationship(
        "FinancialInstitution", back_populates="user"
    )
    financial_accounts: Mapped[List["FinancialAccount"]] = relationship(
        "FinancialAccount", back_populates="user"
    )
    financial_transactions: Mapped[List["FinancialTransaction"]] = relationship(
        "FinancialTransaction", back_populates="user"
    )
    transactions: Mapped[List["Transaction"]] = relationship(
        "Transaction", back_populates="user"
    )

    def __repr__(self) -> str:
        return f"<User {self.email} ({self.id})>"

    def get_permissions(self) -> Dict[str, List[str]]:
        """Get user permissions based on role."""
        base_permissions = {
            "read": ["transactions", "accounts", "institutions", "anomalies"],
            "write": [],
            "admin": [],
        }

        if self.role == UserRole.ADMIN:
            base_permissions["write"].extend(
                ["transactions", "accounts", "institutions", "anomalies"]
            )
            base_permissions["admin"].extend(["users", "settings", "models"])
        elif self.role == UserRole.USER:
            base_permissions["write"].extend(
                ["transactions", "accounts", "institutions"]
            )

        return base_permissions


class AnomalyType(str, SQLEnum):
    """Anomaly type enum."""

    LARGE_AMOUNT = "large_amount"
    UNUSUAL_PATTERN = "unusual_pattern"
    UNAPPROVED_VENDOR = "unapproved_vendor"
    DUPLICATE_TRANSACTION = "duplicate_transaction"
    SUSPICIOUS_TIMING = "suspicious_timing"
    UNUSUAL_LOCATION = "unusual_location"
    UNUSUAL_CATEGORY = "unusual_category"
    UNUSUAL_FREQUENCY = "unusual_frequency"
    UNUSUAL_AMOUNT = "unusual_amount"
    UNUSUAL_MERCHANT = "unusual_merchant"


class FeedbackType(str, Enum):
    """Feedback type enumeration."""

    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    TRUE_NEGATIVE = "true_negative"
    INCONCLUSIVE = "inconclusive"
    IGNORE = "ignore"
    NEEDS_INVESTIGATION = "needs_investigation"


class RiskLevel(str, Enum):
    """Risk level enumeration."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyStatus(str, SQLEnum):
    """Anomaly status enum."""

    NEW = "new"
    IN_REVIEW = "in_review"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"
    CONFIRMED = "confirmed"
    IGNORED = "ignored"


class AnomalyRiskLevel(str, SQLEnum):
    """Anomaly risk level enum."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Notification(Base):
    """System notification records."""

    __tablename__ = "notifications"

    id: Mapped[UUID] = mapped_column(PGUUID(), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(
        PGUUID(), ForeignKey("tenants.id"), nullable=False
    )
    user_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(), ForeignKey("users.id"), nullable=True
    )
    channel: Mapped[NotificationChannel] = mapped_column(
        Enum(NotificationChannel), nullable=False
    )
    status: Mapped[NotificationStatus] = mapped_column(
        Enum(NotificationStatus), nullable=False, default=NotificationStatus.PENDING
    )
    title: Mapped[str] = mapped_column(String, nullable=False)
    message: Mapped[str] = mapped_column(String, nullable=False)
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="notifications")
    user: Mapped[Optional["User"]] = relationship(
        "User", back_populates="notifications"
    )

    def __repr__(self) -> str:
        return f"<Notification {self.id} ({self.channel})>"


class ErrorLog(Base):
    """System error logging."""

    __tablename__ = "error_logs"

    id: Mapped[UUID] = mapped_column(PGUUID(), primary_key=True, default=uuid4)
    tenant_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(), ForeignKey("tenants.id"), nullable=True
    )
    user_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(), ForeignKey("users.id"), nullable=True
    )
    error_type: Mapped[str] = mapped_column(String, nullable=False)
    message: Mapped[str] = mapped_column(String, nullable=False)
    stack_trace: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    # Relationships
    tenant: Mapped[Optional["Tenant"]] = relationship(
        "Tenant", back_populates="error_logs"
    )
    user: Mapped[Optional["User"]] = relationship("User", back_populates="error_logs")

    def __repr__(self) -> str:
        return f"<ErrorLog {self.error_type} ({self.id})>"


class SystemMetric(Base):
    """System performance metrics."""

    __tablename__ = "system_metrics"

    id: Mapped[UUID] = mapped_column(PGUUID(), primary_key=True, default=uuid4)
    tenant_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(), ForeignKey("tenants.id"), nullable=True
    )
    metric_type: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    # Relationships
    tenant: Mapped[Optional["Tenant"]] = relationship(
        "Tenant", back_populates="system_metrics"
    )

    def __repr__(self) -> str:
        return f"<SystemMetric {self.metric_type} ({self.id})>"


class Transaction(Base):
    """Top-level transaction model for business logic."""

    __tablename__ = "transactions"

    id: Mapped[UUID] = mapped_column(PGUUID(), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        PGUUID(), ForeignKey("users.id"), nullable=False, index=True
    )
    account_id: Mapped[UUID] = mapped_column(
        PGUUID(), ForeignKey("financial_accounts.id"), nullable=False
    )
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    merchant_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    pending: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    category: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    category_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    transaction_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    payment_channel: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    location: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    last_updated: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="transactions")
    account: Mapped["FinancialAccount"] = relationship(
        "FinancialAccount", back_populates="transactions"
    )
    anomalies: Mapped[List["Anomaly"]] = relationship(
        "Anomaly", back_populates="transaction"
    )

    def __repr__(self) -> str:
        return f"<Transaction {self.name} ({self.id})>"


class SensitivityConfig(Base):
    """Configuration for anomaly detection sensitivity."""

    __tablename__ = "sensitivity_configs"

    id: Mapped[UUID] = mapped_column(PGUUID(), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(
        PGUUID(), ForeignKey("tenants.id"), nullable=False
    )
    anomaly_type: Mapped[AnomalyType] = mapped_column(Enum(AnomalyType), nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    tenant: Mapped["Tenant"] = relationship(
        "Tenant", back_populates="sensitivity_configs"
    )

    def __repr__(self) -> str:
        return f"<SensitivityConfig {self.anomaly_type} for tenant {self.tenant_id}>"


class Anomaly(Base):
    """Anomaly detection record."""

    __tablename__ = "anomalies"

    id: Mapped[UUID] = mapped_column(PGUUID(), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(
        PGUUID(), ForeignKey("tenants.id"), nullable=False
    )
    transaction_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(), ForeignKey("transactions.id"), nullable=True
    )
    anomaly_type: Mapped[AnomalyType] = mapped_column(Enum(AnomalyType), nullable=False)
    status: Mapped[AnomalyStatus] = mapped_column(
        Enum(AnomalyStatus), nullable=False, default=AnomalyStatus.PENDING
    )
    risk_level: Mapped[AnomalyRiskLevel] = mapped_column(
        Enum(AnomalyRiskLevel), nullable=False
    )
    score: Mapped[float] = mapped_column(Float, nullable=False)
    explanation: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="anomalies")
    transaction: Mapped[Optional["Transaction"]] = relationship(
        "Transaction", back_populates="anomalies"
    )

    def __repr__(self) -> str:
        return f"<Anomaly {self.anomaly_type} ({self.id})>"


class Tenant(Base):
    """Tenant information."""

    __tablename__ = "tenants"

    id: Mapped[UUID] = mapped_column(PGUUID(), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    api_key: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    # Relationships
    users: Mapped[List["User"]] = relationship("User", back_populates="tenant")
    anomalies: Mapped[List["Anomaly"]] = relationship(
        "Anomaly", back_populates="tenant"
    )
    configurations: Mapped[List["TenantConfiguration"]] = relationship(
        "TenantConfiguration", back_populates="tenant"
    )
    notifications: Mapped[List["Notification"]] = relationship(
        "Notification", back_populates="tenant"
    )
    error_logs: Mapped[List["ErrorLog"]] = relationship(
        "ErrorLog", back_populates="tenant"
    )
    system_metrics: Mapped[List["SystemMetric"]] = relationship(
        "SystemMetric", back_populates="tenant"
    )
    task_logs: Mapped[List["TaskLog"]] = relationship(
        "TaskLog", back_populates="tenant"
    )


class TenantConfiguration(Base):
    """Tenant-specific configuration settings."""

    __tablename__ = "tenant_configurations"

    id: Mapped[UUID] = mapped_column(PGUUID(), primary_key=True, default=uuid4)
    tenant_id: Mapped[UUID] = mapped_column(
        PGUUID(), ForeignKey("tenants.id"), nullable=False
    )
    key: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="configurations")

    def __repr__(self) -> str:
        return f"<TenantConfiguration {self.key} for tenant {self.tenant_id}>"


class DataSource(str, SQLEnum):
    """Data source enum."""

    PLAID = "plaid"
    QUICKBOOKS = "quickbooks"
    MANUAL = "manual"
