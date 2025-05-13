"""Database models for AuditPulse MVP."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, List
from uuid import UUID, uuid4

from sqlalchemy import Column, DateTime, Enum as SQLEnum, ForeignKey, Integer, String, JSON, Float, Boolean
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from auditpulse_mvp.database.base_class import Base

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
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_type = Column(String, nullable=False, index=True)
    version = Column(String, nullable=False)
    model_data = Column(JSON, nullable=False)
    metadata = Column(JSON, nullable=False, default=dict)
    is_active = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    activated_at = Column(DateTime, nullable=True)
    deactivated_at = Column(DateTime, nullable=True)
    
    # Relationships
    performance_metrics = relationship("ModelPerformance", back_populates="model_version")
    
    class Config:
        """Pydantic config."""
        
        orm_mode = True
        
class ModelPerformance(Base):
    """Model performance metrics."""
    
    __tablename__ = "model_performance"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
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
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
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
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    institution_id = Column(PGUUID(as_uuid=True), ForeignKey("financial_institutions.id"), nullable=False)
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
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
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
    account = relationship("FinancialAccount", back_populates="transactions", foreign_keys=[account_id], primaryjoin="FinancialAccount.plaid_account_id == FinancialTransaction.account_id")
    
    def __repr__(self):
        return f"<FinancialTransaction {self.name} ({self.id})>"
        
# Add relationships to User class
User.financial_institutions = relationship("FinancialInstitution", back_populates="user")
User.financial_accounts = relationship("FinancialAccount", back_populates="user")
User.financial_transactions = relationship("FinancialTransaction", back_populates="user") 