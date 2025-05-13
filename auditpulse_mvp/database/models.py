"""Database models for AuditPulse MVP."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, List
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
from sqlalchemy.orm import relationship
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
    """Task execution log."""

    __tablename__ = "task_logs"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, index=True, nullable=False)
    task_name = Column(String, nullable=False)
    status = Column(SQLEnum(TaskStatus), nullable=False)
    priority = Column(Integer, nullable=False)
    args = Column(JSON)
    kwargs = Column(JSON)
    result = Column(JSON)
    error = Column(String)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    retry_count = Column(Integer, default=0)

    def __repr__(self):
        return f"<TaskLog {self.task_id} ({self.status})>"


class ModelVersion(Base):
    """Model version information."""

    __tablename__ = "model_versions"

    id = Column(UUID(), primary_key=True, default=uuid4)
    model_type = Column(String, nullable=False, index=True)
    version = Column(String, nullable=False)
    model_data = Column(JSON, nullable=False)
    metadata = Column(JSON, nullable=False, default=dict)
    is_active = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    activated_at = Column(DateTime, nullable=True)
    deactivated_at = Column(DateTime, nullable=True)

    # Relationships
    performance_metrics = relationship(
        "ModelPerformance", back_populates="model_version"
    )

    class Config:
        """Pydantic config."""

        orm_mode = True


class ModelPerformance(Base):
    """Model performance metrics."""

    __tablename__ = "model_performance"

    id = Column(UUID(), primary_key=True, default=uuid4)
    model_type = Column(String, nullable=False, index=True)
    version = Column(String, nullable=False)
    metrics = Column(JSON, nullable=False)
    dataset_size = Column(Integer, nullable=False)
    evaluation_time = Column(Float, nullable=False)
    recorded_at = Column(DateTime, nullable=False, default=datetime.now)

    # Relationships
    model_version = relationship("ModelVersion", back_populates="performance_metrics")

    class Config:
        """Pydantic config."""

        orm_mode = True


class FinancialInstitution(Base):
    """Financial institution connected via Plaid."""

    __tablename__ = "financial_institutions"

    id = Column(PGUUID(), primary_key=True, default=uuid4)
    user_id = Column(
        PGUUID(), ForeignKey("users.id"), nullable=False, index=True
    )
    name = Column(String, nullable=False)
    plaid_access_token = Column(String, nullable=False)
    plaid_item_id = Column(String, nullable=False, index=True)
    plaid_institution_id = Column(String, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="financial_institutions")
    accounts = relationship("FinancialAccount", back_populates="institution")

    def __repr__(self):
        return f"<FinancialInstitution {self.name} ({self.id})>"


class FinancialAccount(Base):
    """Financial account from a financial institution."""

    __tablename__ = "financial_accounts"

    id = Column(PGUUID(), primary_key=True, default=uuid4)
    user_id = Column(
        PGUUID(), ForeignKey("users.id"), nullable=False, index=True
    )
    institution_id = Column(
        PGUUID(), ForeignKey("financial_institutions.id"), nullable=False
    )
    plaid_account_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    official_name = Column(String, nullable=True)
    type = Column(String, nullable=False)
    subtype = Column(String, nullable=True)
    mask = Column(String, nullable=True)
    balances = Column(JSON, nullable=False, default=dict)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="financial_accounts")
    institution = relationship("FinancialInstitution", back_populates="accounts")
    transactions = relationship("FinancialTransaction", back_populates="account")

    def __repr__(self):
        return f"<FinancialAccount {self.name} ({self.id})>"


class FinancialTransaction(Base):
    """Financial transaction from a financial account."""

    __tablename__ = "financial_transactions"

    id = Column(PGUUID(), primary_key=True, default=uuid4)
    user_id = Column(
        PGUUID(), ForeignKey("users.id"), nullable=False, index=True
    )
    account_id = Column(String, nullable=False, index=True)
    transaction_id = Column(String, nullable=False, unique=True, index=True)
    amount = Column(Float, nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    name = Column(String, nullable=False)
    merchant_name = Column(String, nullable=True)
    pending = Column(Boolean, nullable=False, default=False)
    category = Column(JSON, nullable=True)
    category_id = Column(String, nullable=True)
    transaction_type = Column(String, nullable=True)
    payment_channel = Column(String, nullable=True)
    location = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="financial_transactions")
    account = relationship(
        "FinancialAccount",
        back_populates="transactions",
        foreign_keys=[account_id],
        primaryjoin="FinancialAccount.plaid_account_id == FinancialTransaction.account_id",
    )

    def __repr__(self):
        return f"<FinancialTransaction {self.name} ({self.id})>"


class User(Base):
    __tablename__ = "users"
    # ... existing code ...


# Add relationships after User class definition
User.financial_institutions = relationship(
    "FinancialInstitution", back_populates="user"
)
User.financial_accounts = relationship("FinancialAccount", back_populates="user")
User.financial_transactions = relationship(
    "FinancialTransaction", back_populates="user"
)
# ... existing code ...


class AnomalyType(str, Enum):
    """Anomaly type enumeration."""

    TRANSACTION = "transaction"
    ACCOUNT = "account"
    BEHAVIORAL = "behavioral"
    SYSTEM = "system"
    ML_BASED = "ml_based"
    RULES_BASED = "rules_based"
    STATISTICAL_OUTLIER = "statistical_outlier"
    RULE_BASED = "rule_based"
    ML_DETECTED = "ml_detected"


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


class AnomalyStatus(str, Enum):
    """Anomaly status enumeration."""

    NEW = "new"
    IN_REVIEW = "in_review"
    RESOLVED = "resolved"
    IGNORED = "ignored"


class AnomalyRiskLevel(str, Enum):
    """Anomaly risk level enumeration."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationChannel(str, Enum):
    """Notification channel enumeration."""

    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"


class NotificationStatus(str, Enum):
    """Notification status enumeration."""

    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    DELIVERED = "delivered"


class UserRole(str, Enum):
    """User role enumeration."""

    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class DataSource(str, Enum):
    """Data source enumeration."""

    PLAID = "plaid"
    QUICKBOOKS = "quickbooks"
    MANUAL = "manual"


class Anomaly(Base):
    """Anomaly detection result."""

    __tablename__ = "anomalies"

    id = Column(PGUUID(), primary_key=True, default=uuid4)
    tenant_id = Column(
        PGUUID(), ForeignKey("tenants.id"), nullable=False, index=True
    )
    transaction_id = Column(
        PGUUID(), ForeignKey("financial_transactions.id"), nullable=True
    )
    anomaly_type = Column(SQLEnum(AnomalyType), nullable=False)
    status = Column(SQLEnum(AnomalyStatus), nullable=False, default=AnomalyStatus.NEW)
    risk_level = Column(SQLEnum(AnomalyRiskLevel), nullable=False)
    score = Column(Float, nullable=False)
    explanation = Column(String, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

    # Relationships
    tenant = relationship("Tenant", back_populates="anomalies")
    transaction = relationship("FinancialTransaction", back_populates="anomalies")


class Tenant(Base):
    """Tenant information."""

    __tablename__ = "tenants"

    id = Column(PGUUID(), primary_key=True, default=uuid4)
    name = Column(String, nullable=False)
    api_key = Column(String, nullable=False, unique=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    users = relationship("User", back_populates="tenant")
    anomalies = relationship("Anomaly", back_populates="tenant")
    configurations = relationship("TenantConfiguration", back_populates="tenant")


class TenantConfiguration(Base):
    """Tenant configuration settings."""

    __tablename__ = "tenant_configurations"

    id = Column(PGUUID(), primary_key=True, default=uuid4)
    tenant_id = Column(PGUUID(), ForeignKey("tenants.id"), nullable=False)
    key = Column(String, nullable=False)
    value = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    tenant = relationship("Tenant", back_populates="configurations")


class Notification(Base):
    """Notification record."""

    __tablename__ = "notifications"

    id = Column(PGUUID(), primary_key=True, default=uuid4)
    tenant_id = Column(PGUUID(), ForeignKey("tenants.id"), nullable=False)
    user_id = Column(PGUUID(), ForeignKey("users.id"), nullable=False)
    channel = Column(SQLEnum(NotificationChannel), nullable=False)
    status = Column(
        SQLEnum(NotificationStatus), nullable=False, default=NotificationStatus.PENDING
    )
    subject = Column(String, nullable=False)
    message = Column(String, nullable=False)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    sent_at = Column(DateTime, nullable=True)
    delivered_at = Column(DateTime, nullable=True)

    # Relationships
    tenant = relationship("Tenant")
    user = relationship("User")


class ErrorLog(Base):
    """System error log."""

    __tablename__ = "error_logs"

    id = Column(PGUUID(), primary_key=True, default=uuid4)
    tenant_id = Column(PGUUID(), ForeignKey("tenants.id"), nullable=True)
    error_type = Column(String, nullable=False)
    message = Column(String, nullable=False)
    stack_trace = Column(String, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    tenant = relationship("Tenant")


class SystemMetric(Base):
    """System performance metric."""

    __tablename__ = "system_metrics"

    id = Column(PGUUID(), primary_key=True, default=uuid4)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    metadata = Column(JSON, nullable=True)
    recorded_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class Transaction(Base):
    """Top-level transaction model for business logic."""

    __tablename__ = "transactions"

    id = Column(PGUUID(), primary_key=True, default=uuid4)
    user_id = Column(
        PGUUID(), ForeignKey("users.id"), nullable=False, index=True
    )
    account_id = Column(
        PGUUID(), ForeignKey("financial_accounts.id"), nullable=False
    )
    amount = Column(Float, nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    name = Column(String, nullable=False)
    merchant_name = Column(String, nullable=True)
    pending = Column(Boolean, nullable=False, default=False)
    category = Column(JSON, nullable=True)
    category_id = Column(String, nullable=True)
    transaction_type = Column(String, nullable=True)
    payment_channel = Column(String, nullable=True)
    location = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="transactions")
    account = relationship("FinancialAccount", back_populates="transactions")
    anomalies = relationship("Anomaly", back_populates="transaction")

    def __repr__(self):
        return f"<Transaction {self.name} ({self.id})>"


class SensitivityConfig(Base):
    __tablename__ = "sensitivity_configs"
    id = Column(PGUUID(), primary_key=True, default=uuid4)
    tenant_id = Column(PGUUID(), ForeignKey("tenants.id"), nullable=False)
    config = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    tenant = relationship("Tenant")
