"""Admin API for system and tenant management.

This module defines the FastAPI endpoints for administrative functions. This includes
system status retrieval, tenant management, and user management.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import UUID4, EmailStr

from auditpulse_mvp.api.dependencies import (
    get_current_superuser,
    get_current_user,
    get_db,
)
from auditpulse_mvp.database.session import get_db_session
from auditpulse_mvp.database.models import User, Tenant
from auditpulse_mvp.schemas.user import UserCreate, UserResponse
from auditpulse_mvp.schemas.tenant import TenantCreate, TenantResponse, TenantUpdate
from auditpulse_mvp.admin.system_admin import SystemAdmin
from auditpulse_mvp.admin.tenant_admin import TenantAdmin
from auditpulse_mvp.admin.models_admin import TransactionAdmin, AnomalyAdmin, ModelAdmin
from auditpulse_mvp.schemas.admin import (
    SystemTaskResponse,
    SystemStatusResponse,
    TransactionStatsResponse,
    AnomalyStatsResponse,
)
from auditpulse_mvp.schemas.model import (
    ModelVersionResponse,
    ModelVersionCreate,
    ModelPerformanceResponse,
    ModelValidationRequest,
    ModelValidationResponse,
)


# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/admin", tags=["admin"])


# System management endpoints (superuser only)
@router.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """Get system status information.

    Args:
        db: Database session.
        current_user: Current superuser.

    Returns:
        System status data.
    """
    system_admin = SystemAdmin(db)
    return await system_admin.get_status()


@router.post("/system/tasks/{task_name}", response_model=SystemTaskResponse)
async def run_system_task(
    task_name: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """Run a system maintenance task.

    Args:
        task_name: Name of the task to run.
        db: Database session.
        current_user: Current superuser.

    Returns:
        Task execution result.
    """
    system_admin = SystemAdmin(db)
    return await system_admin.run_task(task_name)


# Tenant management endpoints (superuser only)
@router.get("/tenants", response_model=List[TenantResponse])
async def get_tenants(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """Get all tenants with pagination.

    Args:
        skip: Number of records to skip.
        limit: Maximum number of records to return.
        db: Database session.
        current_user: Current superuser.

    Returns:
        List of tenants.
    """
    tenant_admin = TenantAdmin(db)
    return await tenant_admin.get_all(skip=skip, limit=limit)


@router.post(
    "/tenants", response_model=TenantResponse, status_code=status.HTTP_201_CREATED
)
async def create_tenant(
    tenant_data: TenantCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """Create a new tenant.

    Args:
        tenant_data: Tenant data.
        db: Database session.
        current_user: Current superuser.

    Returns:
        Created tenant.
    """
    tenant_admin = TenantAdmin(db)
    return await tenant_admin.create(tenant_data)


@router.get("/tenants/{tenant_id}", response_model=TenantResponse)
async def get_tenant(
    tenant_id: UUID4,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """Get a tenant by ID.

    Args:
        tenant_id: Tenant ID.
        db: Database session.
        current_user: Current superuser.

    Returns:
        Tenant details.
    """
    tenant_admin = TenantAdmin(db)
    tenant = await tenant_admin.get_by_id(tenant_id)

    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )

    return tenant


@router.put("/tenants/{tenant_id}", response_model=TenantResponse)
async def update_tenant(
    tenant_id: UUID4,
    tenant_data: TenantUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """Update a tenant.

    Args:
        tenant_id: Tenant ID.
        tenant_data: Updated tenant data.
        db: Database session.
        current_user: Current superuser.

    Returns:
        Updated tenant.
    """
    tenant_admin = TenantAdmin(db)
    tenant = await tenant_admin.update(tenant_id, tenant_data)

    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )

    return tenant


@router.delete("/tenants/{tenant_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tenant(
    tenant_id: UUID4,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """Delete a tenant.

    Args:
        tenant_id: Tenant ID.
        db: Database session.
        current_user: Current superuser.
    """
    tenant_admin = TenantAdmin(db)
    result = await tenant_admin.delete(tenant_id)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )


# User management for tenants (superuser only)
@router.post(
    "/tenants/{tenant_id}/users",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_tenant_user(
    tenant_id: UUID4,
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """Create a user for a tenant.

    Args:
        tenant_id: Tenant ID.
        user_data: User data.
        db: Database session.
        current_user: Current superuser.

    Returns:
        Created user.
    """
    tenant_admin = TenantAdmin(db)
    return await tenant_admin.create_user(tenant_id, user_data)


@router.get("/tenants/{tenant_id}/users", response_model=List[UserResponse])
async def get_tenant_users(
    tenant_id: UUID4,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """Get all users for a tenant.

    Args:
        tenant_id: Tenant ID.
        skip: Number of records to skip.
        limit: Maximum number of records to return.
        db: Database session.
        current_user: Current superuser.

    Returns:
        List of users.
    """
    tenant_admin = TenantAdmin(db)
    return await tenant_admin.get_users(tenant_id, skip=skip, limit=limit)


# Transaction stats endpoint
@router.get("/transactions/stats", response_model=TransactionStatsResponse)
async def get_transaction_stats(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get transaction statistics for the current user's tenant.

    Args:
        start_date: Optional start date for filtering.
        end_date: Optional end date for filtering.
        db: Database session.
        current_user: Current user.

    Returns:
        Transaction statistics.
    """
    if not current_user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is not associated with a tenant",
        )

    transaction_admin = TransactionAdmin(db)
    stats = await transaction_admin.get_stats(
        tenant_id=current_user.tenant_id,
        start_date=start_date,
        end_date=end_date,
    )

    return stats


# Anomaly stats endpoint
@router.get("/anomalies/stats", response_model=AnomalyStatsResponse)
async def get_anomaly_stats(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get anomaly statistics for the current user's tenant.

    Args:
        start_date: Optional start date for filtering.
        end_date: Optional end date for filtering.
        db: Database session.
        current_user: Current user.

    Returns:
        Anomaly statistics.
    """
    if not current_user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is not associated with a tenant",
        )

    anomaly_admin = AnomalyAdmin(db)
    stats = await anomaly_admin.get_stats(
        tenant_id=current_user.tenant_id,
        start_date=start_date,
        end_date=end_date,
    )

    return stats


# Model management endpoints
@router.get("/models/types", response_model=List[str])
async def get_model_types(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """Get all model types in the system.

    Args:
        db: Database session.
        current_user: Current superuser.

    Returns:
        List of model types.
    """
    model_admin = ModelAdmin(db)
    return await model_admin.get_all_model_types()


@router.get("/models/{model_type}/versions", response_model=List[ModelVersionResponse])
async def get_model_versions(
    model_type: str = Path(..., description="Type of model"),
    include_inactive: bool = Query(
        True, description="Whether to include inactive versions"
    ),
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get all versions of a model type.

    Args:
        model_type: Type of model.
        include_inactive: Whether to include inactive versions.
        skip: Number of records to skip.
        limit: Maximum number of records to return.
        db: Database session.
        current_user: Current user.

    Returns:
        List of model versions.
    """
    model_admin = ModelAdmin(db)
    versions = await model_admin.get_all_versions(
        model_type=model_type,
        skip=skip,
        limit=limit,
        include_inactive=include_inactive,
    )

    return versions


@router.get("/models/active", response_model=List[ModelVersionResponse])
async def get_active_model_versions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get all active model versions.

    Args:
        db: Database session.
        current_user: Current user.

    Returns:
        List of active model versions.
    """
    model_admin = ModelAdmin(db)
    return await model_admin.get_active_versions()


@router.get("/models/versions/{version_id}", response_model=ModelVersionResponse)
async def get_model_version(
    version_id: UUID4,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a model version by ID.

    Args:
        version_id: Model version ID.
        db: Database session.
        current_user: Current user.

    Returns:
        Model version details.
    """
    model_admin = ModelAdmin(db)
    version = await model_admin.get_version_by_id(version_id)

    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model version not found",
        )

    return version


@router.put(
    "/models/versions/{version_id}/activate", response_model=ModelVersionResponse
)
async def activate_model_version(
    version_id: UUID4,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """Activate a model version.

    Args:
        version_id: Model version ID.
        db: Database session.
        current_user: Current superuser.

    Returns:
        Activated model version.
    """
    model_admin = ModelAdmin(db)
    version = await model_admin.activate_version(version_id)

    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model version not found",
        )

    return version


@router.put(
    "/models/versions/{version_id}/deactivate", response_model=ModelVersionResponse
)
async def deactivate_model_version(
    version_id: UUID4,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """Deactivate a model version.

    Args:
        version_id: Model version ID.
        db: Database session.
        current_user: Current superuser.

    Returns:
        Deactivated model version.
    """
    model_admin = ModelAdmin(db)
    version = await model_admin.deactivate_version(version_id)

    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model version not found",
        )

    return version


@router.get(
    "/models/{model_type}/performance", response_model=List[ModelPerformanceResponse]
)
async def get_model_performance(
    model_type: str = Path(..., description="Type of model"),
    version: Optional[str] = Query(None, description="Optional version identifier"),
    limit: int = Query(10, description="Maximum number of records to return"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get performance history for a model.

    Args:
        model_type: Type of model.
        version: Optional version identifier.
        limit: Maximum number of records to return.
        db: Database session.
        current_user: Current user.

    Returns:
        List of performance records.
    """
    model_admin = ModelAdmin(db)
    performance = await model_admin.get_performance_history(
        model_type=model_type,
        version=version,
        limit=limit,
    )

    return performance


@router.get("/models/versions/{version_id}/config", response_model=Dict[str, Any])
async def export_model_config(
    version_id: UUID4,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Export a model's configuration and metadata.

    Args:
        version_id: Model version ID.
        db: Database session.
        current_user: Current user.

    Returns:
        Model configuration.
    """
    model_admin = ModelAdmin(db)
    config = await model_admin.export_model_config(version_id)

    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model version not found",
        )

    return config


@router.put("/models/versions/{version_id}/rename", response_model=ModelVersionResponse)
async def rename_model_version(
    version_id: UUID4,
    new_version: str = Query(..., description="New version name"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """Rename a model version.

    Args:
        version_id: Model version ID.
        new_version: New version name.
        db: Database session.
        current_user: Current superuser.

    Returns:
        Updated model version.
    """
    model_admin = ModelAdmin(db)
    version = await model_admin.rename_model_version(version_id, new_version)

    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model version not found or version name already exists",
        )

    return version


# Error monitoring endpoints
@router.get("/system/errors", response_model=List[Dict[str, Any]])
async def get_error_logs(
    limit: int = Query(100, description="Maximum number of records to return"),
    skip: int = Query(0, description="Number of records to skip"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    component: Optional[str] = Query(None, description="Filter by component"),
    error_type: Optional[str] = Query(None, description="Filter by error type"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """Get system error logs with filtering and pagination.

    Args:
        limit: Maximum number of records to return.
        skip: Number of records to skip.
        start_date: Optional start date for filtering.
        end_date: Optional end date for filtering.
        component: Optional component for filtering.
        error_type: Optional error type for filtering.
        db: Database session.
        current_user: Current superuser.

    Returns:
        List of error logs.
    """
    system_admin = SystemAdmin(db)
    return await system_admin.get_error_logs(
        limit=limit,
        skip=skip,
        start_date=start_date,
        end_date=end_date,
        component=component,
        error_type=error_type,
    )


@router.get("/system/health", response_model=Dict[str, Any])
async def get_system_health(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """Get comprehensive system health check results.

    Args:
        db: Database session.
        current_user: Current superuser.

    Returns:
        Health check results.
    """
    system_admin = SystemAdmin(db)
    return await system_admin.perform_health_check()


@router.get("/system/metrics", response_model=List[Dict[str, Any]])
async def get_system_metrics(
    metric_name: Optional[str] = Query(None, description="Filter by metric name"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    limit: int = Query(100, description="Maximum number of records to return"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """Get system metrics with filtering.

    Args:
        metric_name: Optional metric name for filtering.
        start_date: Optional start date for filtering.
        end_date: Optional end date for filtering.
        limit: Maximum number of records to return.
        db: Database session.
        current_user: Current superuser.

    Returns:
        List of system metrics.
    """
    system_admin = SystemAdmin(db)
    return await system_admin.get_system_metrics(
        metric_name=metric_name,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )


@router.post("/system/metrics", status_code=status.HTTP_201_CREATED)
async def record_system_metric(
    name: str = Query(..., description="Metric name"),
    value: Any = Body(..., description="Metric value"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """Record a system metric.

    Args:
        name: Metric name.
        value: Metric value.
        db: Database session.
        current_user: Current superuser.

    Returns:
        Created system metric.
    """
    system_admin = SystemAdmin(db)
    await system_admin.record_metric(name=name, value=value)
    return {"status": "success", "message": f"Metric '{name}' recorded successfully"}


@router.post("/system/alerts", status_code=status.HTTP_200_OK)
async def send_system_alert(
    alert_type: str = Query(..., description="Type of alert"),
    message: str = Body(..., description="Alert message"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser),
):
    """Send a system alert to administrators.

    Args:
        alert_type: Type of alert.
        message: Alert message.
        db: Database session.
        current_user: Current superuser.

    Returns:
        Alert sending result.
    """
    system_admin = SystemAdmin(db)
    result = await system_admin.send_system_alerts(
        alert_type=alert_type, message=message
    )

    if result:
        return {"status": "success", "message": "Alert sent successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send alert",
        )
