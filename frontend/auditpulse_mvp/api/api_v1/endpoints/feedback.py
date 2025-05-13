"""Feedback API endpoints for AuditPulse MVP.

This module provides API endpoints for submitting and retrieving feedback
on anomalies.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import UUID4
from sqlalchemy.orm import Session

from auditpulse_mvp.api.api_v1.auth import get_current_user
from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.feedback.feedback_service import (
    FeedbackRequest,
    FeedbackResponse,
    FeedbackService,
)

# Create router
router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    db: Session = Depends(get_db_session),
    current_user=Depends(get_current_user),
) -> FeedbackResponse:
    """Submit feedback for an anomaly.

    Args:
        feedback: Feedback request.
        db: Database session.
        current_user: Current user.

    Returns:
        Feedback response.

    Raises:
        HTTPException: If feedback submission fails.
    """
    # Initialize service
    service = FeedbackService(db)

    # Submit feedback
    return await service.submit_feedback(
        tenant_id=current_user.tenant_id,
        feedback=feedback,
    )


@router.get("/feedback/stats")
async def get_feedback_stats(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db_session),
    current_user=Depends(get_current_user),
) -> dict:
    """Get feedback statistics.

    Args:
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        db: Database session.
        current_user: Current user.

    Returns:
        Dictionary containing feedback statistics.

    Raises:
        HTTPException: If statistics retrieval fails.
    """
    # Initialize service
    service = FeedbackService(db)

    # Get statistics
    return await service.get_feedback_stats(
        tenant_id=current_user.tenant_id,
        start_date=start_date,
        end_date=end_date,
    )


@router.get("/feedback/{anomaly_id}")
async def get_anomaly_feedback(
    anomaly_id: UUID4,
    db: Session = Depends(get_db_session),
    current_user=Depends(get_current_user),
) -> dict:
    """Get feedback for a specific anomaly.

    Args:
        anomaly_id: Anomaly ID.
        db: Database session.
        current_user: Current user.

    Returns:
        Dictionary containing anomaly feedback.

    Raises:
        HTTPException: If anomaly not found or feedback retrieval fails.
    """
    # Get anomaly
    anomaly = (
        db.query(Anomaly)
        .filter(
            Anomaly.id == anomaly_id,
            Anomaly.tenant_id == current_user.tenant_id,
        )
        .first()
    )

    if not anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")

    # Return feedback
    return {
        "anomaly_id": str(anomaly.id),
        "is_resolved": anomaly.is_resolved,
        "feedback_type": anomaly.feedback_type.value if anomaly.feedback_type else None,
        "resolution_notes": anomaly.resolution_notes,
        "resolved_at": anomaly.resolved_at,
    }
