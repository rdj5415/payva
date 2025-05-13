"""API endpoints for feedback learning and continuous model improvement."""

import logging
import uuid
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, status
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.api.api_v1.deps import get_db_session, get_current_tenant
from auditpulse_mvp.database.models import Tenant
from auditpulse_mvp.schemas.learning import (
    LearningTriggerResponse,
    LearningStatus,
    FeedbackStatistics,
    LearningScheduleUpdate,
    SensitivityUpdateResponse,
)
from auditpulse_mvp.learning.feedback_learning import (
    get_feedback_learner,
    update_thresholds_from_feedback,
    get_learning_statistics,
)
from auditpulse_mvp.database.configuration import (
    get_tenant_configuration,
    update_tenant_configuration,
)
from auditpulse_mvp.api.api_v1.endpoints.config import (
    update_sensitivity,
    SensitivityUpdateRequest,
)

# Set up logging
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()


@router.post(
    "/trigger",
    response_model=LearningTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_learning_now(
    db: AsyncSession = Depends(get_db_session),
    current_tenant: Tenant = Depends(get_current_tenant),
    days: int = Query(30, description="Number of days of feedback to consider"),
) -> Dict[str, Any]:
    """
    Trigger immediate feedback learning for the current tenant.

    This endpoint initiates a process that analyzes recent user feedback
    to adjust anomaly detection parameters.
    """
    try:
        logger.info(
            f"Triggering immediate feedback learning for tenant {current_tenant.id}"
        )

        # Get the feedback learner
        learner = await get_feedback_learner(db_session=db)

        # Process recent feedback
        result = await learner.process_recent_feedback(current_tenant.id, days=days)

        # Log the result
        if result["status"] == "success":
            logger.info(
                f"Feedback learning completed successfully for tenant {current_tenant.id}: "
                f"processed {result['processed_count']} anomalies"
            )
        else:
            logger.info(
                f"Feedback learning skipped for tenant {current_tenant.id}: {result.get('reason', 'unknown reason')}"
            )

        # Store the result in tenant configuration
        await update_tenant_configuration(
            db,
            current_tenant.id,
            "last_learning_run",
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": result["status"],
                **{
                    k: v
                    for k, v in result.items()
                    if k != "status" and not isinstance(v, dict)
                },
            },
        )

        return result
    except Exception as e:
        logger.error(
            f"Failed to trigger feedback learning for tenant {current_tenant.id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger feedback learning: {str(e)}",
        )


@router.get("/status", response_model=LearningStatus)
async def get_learning_status(
    db: AsyncSession = Depends(get_db_session),
    current_tenant: Tenant = Depends(get_current_tenant),
) -> Dict[str, Any]:
    """
    Get the current feedback learning status and schedule for the tenant.

    This endpoint returns information about the last learning run and the
    scheduled learning configuration.
    """
    try:
        # Get tenant configuration
        config = await get_tenant_configuration(
            db, current_tenant.id, "learning_config"
        )

        if not config:
            # Return default configuration if none exists
            return {
                "last_learning_run": None,
                "scheduled_learning": {
                    "enabled": True,
                    "frequency": "daily",
                    "hour": 3,  # Default: 3 AM
                    "minute": 0,
                    "next_run": None,
                },
            }

        return config
    except Exception as e:
        logger.error(
            f"Failed to get learning status for tenant {current_tenant.id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get learning status: {str(e)}",
        )


@router.get("/statistics", response_model=FeedbackStatistics)
async def get_feedback_statistics(
    days_lookback: int = Query(
        90, description="Number of days to look back for statistics"
    ),
    db: AsyncSession = Depends(get_db_session),
    current_tenant: Tenant = Depends(get_current_tenant),
) -> Dict[str, Any]:
    """
    Get statistics about feedback and learning effectiveness.

    This endpoint provides metrics on feedback patterns, false positive rates,
    and learning adjustments over time.
    """
    try:
        statistics = await get_learning_statistics(
            current_tenant.id,
            db,
            days_lookback=days_lookback,
        )
        return statistics
    except Exception as e:
        logger.error(
            f"Failed to get feedback statistics for tenant {current_tenant.id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get feedback statistics: {str(e)}",
        )


@router.post("/schedule", response_model=Dict[str, Any])
async def update_learning_schedule(
    update_data: LearningScheduleUpdate = Body(...),
    db: AsyncSession = Depends(get_db_session),
    current_tenant: Tenant = Depends(get_current_tenant),
) -> Dict[str, Any]:
    """
    Update the feedback learning schedule for the tenant.

    This endpoint allows configuring when automatic feedback learning runs.
    """
    try:
        # Update the tenant configuration
        await update_tenant_configuration(
            db,
            current_tenant.id,
            "learning_schedule",
            update_data.dict(),
        )

        # Return success response
        return {
            "status": "success",
            "message": "Learning schedule updated successfully",
            "update": update_data.dict(),
        }
    except Exception as e:
        logger.error(
            f"Failed to update learning schedule for tenant {current_tenant.id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update learning schedule: {str(e)}",
        )


@router.post("/sensitivity", response_model=SensitivityUpdateResponse)
async def update_sensitivity_from_feedback(
    db: AsyncSession = Depends(get_db_session),
    current_tenant: Tenant = Depends(get_current_tenant),
) -> Dict[str, Any]:
    """
    Update sensitivity configuration based on feedback patterns.

    This endpoint analyzes recent feedback and adjusts sensitivity settings
    accordingly, without requiring a full learning process.
    """
    try:
        # Get the feedback learner
        learner = await get_feedback_learner(db_session=db)

        # Gather feedback data
        feedback_data = await learner.gather_feedback_data(current_tenant.id, days=30)

        if not feedback_data["anomalies"]:
            return {
                "status": "skipped",
                "reason": "No recent feedback found",
                "update_applied": False,
            }

        # Analyze feedback
        feedback_analysis = learner.analyze_feedback(feedback_data)

        # Update sensitivity configuration
        updated_config = await learner.update_sensitivity_config(
            current_tenant.id, feedback_analysis
        )

        return {
            "status": "success",
            "message": "Sensitivity configuration updated successfully",
            "update_applied": True,
            "previous_level": feedback_data.get("previous_level", "unknown"),
            "new_level": updated_config.sensitivity_level.value,
            "false_positive_rate": feedback_analysis["false_positive_rate"],
            "true_positive_rate": feedback_analysis["true_positive_rate"],
        }
    except Exception as e:
        logger.error(
            f"Failed to update sensitivity from feedback for tenant {current_tenant.id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update sensitivity from feedback: {str(e)}",
        )
