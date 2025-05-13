"""Pydantic schemas for admin API endpoints.

This module defines the request and response schemas for the admin API.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, EmailStr, Field, UUID4, validator

from auditpulse_mvp.database.models import AnomalyType, FeedbackType


# Tenant schemas
class TenantBase(BaseModel):
    """Base schema for tenants."""

    name: str = Field(..., min_length=2, max_length=100)
    slug: str = Field(..., min_length=2, max_length=50, pattern=r"^[a-z0-9-]+$")
    description: Optional[str] = Field(None, max_length=500)


class TenantCreate(TenantBase):
    """Schema for tenant creation."""

    is_active: bool = True
    settings: Optional[Dict[str, Any]] = None
    risk_settings: Optional[Dict[str, Any]] = None

    @validator("slug")
    def validate_slug(cls, v):
        """Validate that the slug is lowercase and contains only letters, numbers, and hyphens."""
        if not all(c.isalnum() or c == "-" for c in v):
            raise ValueError("Slug must contain only letters, numbers, and hyphens")
        if not v[0].isalnum():
            raise ValueError("Slug must start with a letter or number")
        if not v[-1].isalnum():
            raise ValueError("Slug must end with a letter or number")
        return v


class TenantUpdate(BaseModel):
    """Schema for tenant updates."""

    name: Optional[str] = Field(None, min_length=2, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    is_active: Optional[bool] = None
    settings: Optional[Dict[str, Any]] = None
    risk_settings: Optional[Dict[str, Any]] = None


class TenantRead(TenantBase):
    """Schema for tenant responses."""

    id: UUID4
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    settings: Optional[Dict[str, Any]] = None
    risk_settings: Optional[Dict[str, Any]] = None

    class Config:
        """Pydantic config."""

        orm_mode = True


# User schemas
class UserBase(BaseModel):
    """Base schema for users."""

    email: EmailStr = Field(..., max_length=100)
    full_name: str = Field(..., min_length=2, max_length=100)
    is_admin: bool = False
    is_superuser: bool = False


class UserCreate(UserBase):
    """Schema for user creation."""

    password: str = Field(..., min_length=8, max_length=100)
    is_active: bool = True
    notification_preferences: Optional[Dict[str, Any]] = None


class UserUpdate(BaseModel):
    """Schema for user updates."""

    email: Optional[EmailStr] = Field(None, max_length=100)
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    password: Optional[str] = Field(None, min_length=8, max_length=100)
    is_admin: Optional[bool] = None
    is_active: Optional[bool] = None
    notification_preferences: Optional[Dict[str, Any]] = None


class UserRead(UserBase):
    """Schema for user responses."""

    id: UUID4
    tenant_id: UUID4
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    notification_preferences: Optional[Dict[str, Any]] = None

    class Config:
        """Pydantic config."""

        orm_mode = True


# System status schemas
class SystemServiceStatus(BaseModel):
    """Status of a system service."""

    api: str
    database: str
    task_queue: str
    scheduler: str
    cache: str


class JobInfo(BaseModel):
    """Information about a scheduled job."""

    id: str
    name: str
    next_run_time: Optional[datetime] = None
    trigger: str


class SystemStatusResponse(BaseModel):
    """Response model for system status endpoint."""

    status: str
    uptime_seconds: int
    service_status: SystemServiceStatus
    jobs: List[JobInfo]
    resource_usage: Dict[str, Any]


class SystemTaskResponse(BaseModel):
    """Response model for system task execution endpoint."""

    task_id: str
    status: str
    message: str
    started_at: datetime
    details: Optional[Dict[str, Any]] = None


class TransactionStatsResponse(BaseModel):
    """Response model for transaction statistics endpoint."""

    total_count: int
    total_amount: float
    average_amount: float
    sources: Dict[str, int]


class AnomalyStatsResponse(BaseModel):
    """Response model for anomaly statistics endpoint."""

    total_count: int
    by_type: Dict[str, int]
    resolution_status: Dict[str, int]
    feedback_types: Dict[str, int]
    average_score: float


class ModelStatsResponse(BaseModel):
    """Response model for model statistics endpoint."""

    total_versions: int
    active_versions: int
    model_types: List[str]
    versions_by_type: Dict[str, int]
    latest_created: Optional[datetime] = None
    average_accuracy: Optional[float] = None


class ModelActivationRequest(BaseModel):
    """Request model for activating a model version."""

    version_id: UUID4 = Field(..., description="ID of the model version to activate")


class ModelDeactivationRequest(BaseModel):
    """Request model for deactivating a model version."""

    version_id: UUID4 = Field(..., description="ID of the model version to deactivate")


class ModelExportRequest(BaseModel):
    """Request model for exporting a model's configuration."""

    version_id: UUID4 = Field(..., description="ID of the model version to export")
    include_metrics: bool = Field(
        True, description="Whether to include performance metrics in the export"
    )


class ModelRenameRequest(BaseModel):
    """Request model for renaming a model version."""

    version_id: UUID4 = Field(..., description="ID of the model version to rename")
    new_version: str = Field(
        ..., description="New version name", min_length=1, max_length=50
    )

    @validator("new_version")
    def validate_version_name(cls, v):
        """Validate that the version name is valid."""
        if not v or not v.strip():
            raise ValueError("Version name cannot be empty")

        forbidden_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|", " "]
        for char in forbidden_chars:
            if char in v:
                raise ValueError(f"Version name cannot contain character: {char}")

        return v


# Transaction admin schemas
class TransactionFilter(BaseModel):
    """Schema for transaction filtering."""

    tenant_id: UUID4
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    skip: int = 0
    limit: int = 100


class TransactionStats(BaseModel):
    """Schema for transaction statistics."""

    total_count: int
    total_amount: float
    average_amount: float
    sources: Dict[str, int]


# Anomaly admin schemas
class AnomalyFilter(BaseModel):
    """Schema for anomaly filtering."""

    tenant_id: UUID4
    is_resolved: Optional[bool] = None
    anomaly_type: Optional[str] = None
    min_score: Optional[float] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    skip: int = 0
    limit: int = 100


class AnomalyResolve(BaseModel):
    """Schema for anomaly resolution."""

    resolution_notes: str = Field(..., min_length=3, max_length=500)
    feedback_type: str


class AnomalyStats(BaseModel):
    """Schema for anomaly statistics."""

    total_count: int
    resolved_count: int
    unresolved_count: int
    resolution_rate: float
    by_type: Dict[str, int]
    by_feedback: Dict[str, int]
