"""API endpoints for continuous learning functionality.

This module provides API endpoints for working with the continuous learning
features, including manual triggering of retraining and viewing feedback statistics.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.api.deps import (
    get_current_user,
    get_current_tenant,
    require_admin,
    log_audit_action,
    AuditAction,
)
from auditpulse_mvp.database.models import User, Tenant
from auditpulse_mvp.database.session import get_db
from auditpulse_mvp.learning.feedback_learning import (
    update_thresholds_from_feedback,
    get_learning_statistics,
    FeedbackLearner,
    get_feedback_learner,
)
from auditpulse_mvp.learning.scheduler import (
    get_feedback_learning_scheduler,
    FeedbackLearningScheduler,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class LearningStats(BaseModel):
    """Learning statistics model."""
    
    total_feedback: int
    rules_based: Dict[str, Any]
    ml_based: Dict[str, Any]
    combined: Dict[str, Any]
    learning_adjustments: Dict[str, Any]


class LearningResult(BaseModel):
    """Result from triggering learning process."""
    
    status: str
    processed_count: int
    rules_updated: Optional[int] = None
    ml_models_updated: Optional[int] = None
    false_positive_rate: Optional[float] = None
    true_positive_rate: Optional[float] = None
    error: Optional[str] = None
    no_action_reason: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


@router.get(
    "/statistics",
    response_model=LearningStats,
    summary="Get learning statistics for the tenant",
)
async def get_statistics(
    days: int = Query(90, ge=1, le=365, description="Number of days to look back"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    current_tenant: Tenant = Depends(get_current_tenant),
) -> LearningStats:
    """Get statistics about feedback and learning effectiveness.
    
    Args:
        days: Number of days to look back
        db: Database session
        current_user: Current authenticated user
        current_tenant: Current tenant
        
    Returns:
        LearningStats: Statistics about feedback and learning effectiveness
    """
    # Log audit action
    await log_audit_action(
        db=db,
        action=AuditAction(
            action="get_learning_statistics",
            user_id=current_user.id,
            tenant_id=current_tenant.id,
            resource_type="learning",
            details={"days": days},
        ),
    )
    
    # Get learning statistics
    stats = await get_learning_statistics(current_tenant.id, db, days_lookback=days)
    
    return LearningStats(**stats)


@router.post(
    "/process-feedback",
    response_model=LearningResult,
    summary="Trigger learning process based on feedback",
)
async def trigger_learning(
    days: int = Query(30, ge=1, le=365, description="Number of days of feedback to consider"),
    min_feedback: int = Query(5, ge=1, description="Minimum feedback count required to make adjustments"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
    current_tenant: Tenant = Depends(get_current_tenant),
) -> LearningResult:
    """Trigger learning process based on feedback.
    
    This endpoint processes recent feedback to adjust ML thresholds and rule parameters.
    
    Args:
        days: Number of days of feedback to consider
        min_feedback: Minimum feedback count required to make adjustments
        db: Database session
        current_user: Current authenticated user (must be admin)
        current_tenant: Current tenant
        
    Returns:
        LearningResult: Results of the learning process
    """
    # Log audit action
    await log_audit_action(
        db=db,
        action=AuditAction(
            action="trigger_learning",
            user_id=current_user.id,
            tenant_id=current_tenant.id,
            resource_type="learning",
            details={"days": days, "min_feedback": min_feedback},
        ),
    )
    
    # Trigger learning process
    result = await update_thresholds_from_feedback(
        tenant_id=current_tenant.id,
        db=db,
        days_lookback=days,
        min_feedback_count=min_feedback,
    )
    
    if result:
        # Transform result into response model
        response = {
            "status": result.get("status", "unknown"),
            "processed_count": result.get("processed_count", 0),
            "rules_updated": result.get("rules_updated", 0),
            "ml_models_updated": result.get("ml_models_updated", 0),
            "false_positive_rate": result.get("false_positive_rate"),
            "true_positive_rate": result.get("true_positive_rate"),
            "error": result.get("error"),
            "no_action_reason": result.get("no_action_reason"),
            "timestamp": datetime.utcnow(),
        }
        
        return LearningResult(**response)
    else:
        # Return error if process failed
        return LearningResult(
            status="error",
            processed_count=0,
            error="Learning process failed",
            timestamp=datetime.utcnow(),
        )


@router.post(
    "/schedule",
    summary="Schedule or reschedule the learning process",
)
async def schedule_learning(
    enable: bool = Query(True, description="Enable or disable scheduled learning"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
    current_tenant: Tenant = Depends(get_current_tenant),
    scheduler: FeedbackLearningScheduler = Depends(get_feedback_learning_scheduler),
) -> Dict[str, Any]:
    """Schedule or reschedule the learning process.
    
    Args:
        enable: Enable or disable scheduled learning
        db: Database session
        current_user: Current authenticated user (must be admin)
        current_tenant: Current tenant
        scheduler: Feedback learning scheduler
        
    Returns:
        Dict with status information
    """
    # Log audit action
    await log_audit_action(
        db=db,
        action=AuditAction(
            action="schedule_learning",
            user_id=current_user.id,
            tenant_id=current_tenant.id,
            resource_type="learning",
            details={"enable": enable},
        ),
    )
    
    if enable:
        # Start scheduler if not already running
        if not scheduler._is_running:
            await scheduler.start()
            
        return {
            "status": "scheduled",
            "message": "Learning process scheduled",
            "is_running": scheduler._is_running,
        }
    else:
        # Stop scheduler if running
        if scheduler._is_running:
            await scheduler.stop()
            
        return {
            "status": "unscheduled",
            "message": "Learning process unscheduled",
            "is_running": scheduler._is_running,
        } 