"""Anomaly detection API endpoints.

This module provides API endpoints for retrieving and managing detected anomalies.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status, Path, Body
from pydantic import BaseModel, Field
from sqlalchemy import select, update, and_, or_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from auditpulse_mvp.api.deps import (
    get_current_user,
    require_admin,
    require_auditor,
    get_current_tenant,
    log_audit_action,
    AuditAction,
)
from auditpulse_mvp.database.models import User, Tenant, Anomaly, Transaction
from auditpulse_mvp.database.session import get_db
from auditpulse_mvp.utils.settings import settings

# Import for immediate learning trigger
from auditpulse_mvp.learning.feedback_learning import update_thresholds_from_feedback

logger = logging.getLogger(__name__)

router = APIRouter()


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected."""

    RULES_BASED = "rules_based"
    ML_BASED = "ml_based"
    COMBINED = "combined"


class AnomalyStatus(str, Enum):
    """Status of an anomaly."""

    NEW = "new"
    REVIEWED = "reviewed"
    DISMISSED = "dismissed"
    CONFIRMED = "confirmed"
    RESOLVED = "resolved"


class FeedbackType(str, Enum):
    """Type of feedback for an anomaly."""

    FALSE_POSITIVE = "false_positive"
    TRUE_POSITIVE = "true_positive"
    NEEDS_INVESTIGATION = "needs_investigation"
    OTHER = "other"


class AnomalyFilter(BaseModel):
    """Filter model for anomaly queries."""

    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_risk_score: Optional[float] = None
    max_risk_score: Optional[float] = None
    types: Optional[List[AnomalyType]] = None
    statuses: Optional[List[AnomalyStatus]] = None
    transaction_types: Optional[List[str]] = None
    search_term: Optional[str] = None


class TransactionModel(BaseModel):
    """Model for transaction data in anomaly responses."""

    id: UUID
    date: datetime
    amount: float
    description: str
    type: str
    account_id: Optional[str] = None
    counterparty: Optional[str] = None
    category: Optional[str] = None
    memo: Optional[str] = None


class AnomalyFeedback(BaseModel):
    """Model for submitting feedback on an anomaly."""

    feedback_type: FeedbackType
    notes: Optional[str] = None
    should_notify: bool = Field(
        False, description="Whether to send notifications about this feedback"
    )


class AnomalyResponse(BaseModel):
    """Response model for a single anomaly."""

    id: UUID
    tenant_id: UUID
    transaction_id: UUID
    anomaly_type: str
    rule_name: Optional[str] = None
    description: str
    risk_score: float
    risk_level: str
    explanation: Optional[str] = None
    is_resolved: bool
    created_at: datetime
    updated_at: datetime
    feedback: Optional[str] = None
    feedback_notes: Optional[str] = None
    transaction: Optional[TransactionModel] = None
    detection_metadata: Optional[Dict[str, Any]] = None


class AnomalyListResponse(BaseModel):
    """Response model for a list of anomalies."""

    items: List[AnomalyResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


def get_risk_level(risk_score: float) -> str:
    """Get the risk level from a risk score.

    Args:
        risk_score: Risk score between 0 and 100

    Returns:
        str: Risk level (negligible, low, medium, high, critical)
    """
    if risk_score < 20:
        return "negligible"
    elif risk_score < 40:
        return "low"
    elif risk_score < 60:
        return "medium"
    elif risk_score < 80:
        return "high"
    else:
        return "critical"


@router.get(
    "",
    response_model=AnomalyListResponse,
    summary="List anomalies with filtering options",
)
async def list_anomalies(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    min_risk_score: Optional[float] = Query(
        None, ge=0, le=100, description="Minimum risk score"
    ),
    max_risk_score: Optional[float] = Query(
        None, ge=0, le=100, description="Maximum risk score"
    ),
    anomaly_types: Optional[List[AnomalyType]] = Query(
        None, description="Filter by anomaly types"
    ),
    statuses: Optional[List[AnomalyStatus]] = Query(
        None, description="Filter by statuses"
    ),
    transaction_types: Optional[List[str]] = Query(
        None, description="Filter by transaction types"
    ),
    search_term: Optional[str] = Query(
        None, description="Search in description or transaction details"
    ),
    include_transactions: bool = Query(
        False, description="Include transaction details in response"
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    current_tenant: Tenant = Depends(get_current_tenant),
) -> AnomalyListResponse:
    """List anomalies with various filtering options.

    Args:
        page: Page number (1-based)
        page_size: Items per page
        start_date: Filter by start date
        end_date: Filter by end date
        min_risk_score: Minimum risk score
        max_risk_score: Maximum risk score
        anomaly_types: Filter by anomaly types
        statuses: Filter by statuses
        transaction_types: Filter by transaction types
        search_term: Search in description or transaction details
        include_transactions: Include transaction details in response
        db: Database session
        current_user: Current authenticated user
        current_tenant: Current tenant

    Returns:
        AnomalyListResponse: List of anomalies matching the filters
    """
    # Build query
    query = select(Anomaly).where(Anomaly.tenant_id == current_tenant.id)

    # Apply filters
    if start_date:
        query = query.where(Anomaly.created_at >= start_date)

    if end_date:
        query = query.where(Anomaly.created_at <= end_date)

    if min_risk_score is not None:
        query = query.where(Anomaly.risk_score >= min_risk_score)

    if max_risk_score is not None:
        query = query.where(Anomaly.risk_score <= max_risk_score)

    if anomaly_types:
        query = query.where(Anomaly.anomaly_type.in_([t.value for t in anomaly_types]))

    if statuses:
        # Map statuses to database fields
        status_conditions = []
        for status in statuses:
            if status == AnomalyStatus.NEW:
                status_conditions.append(
                    and_(Anomaly.feedback.is_(None), Anomaly.is_resolved == False)
                )
            elif status == AnomalyStatus.REVIEWED:
                status_conditions.append(
                    and_(Anomaly.feedback.isnot(None), Anomaly.is_resolved == False)
                )
            elif status == AnomalyStatus.DISMISSED:
                status_conditions.append(
                    and_(
                        Anomaly.feedback == "false_positive",
                        Anomaly.is_resolved == True,
                    )
                )
            elif status == AnomalyStatus.CONFIRMED:
                status_conditions.append(
                    and_(
                        Anomaly.feedback == "true_positive",
                        Anomaly.is_resolved == False,
                    )
                )
            elif status == AnomalyStatus.RESOLVED:
                status_conditions.append(Anomaly.is_resolved == True)

        if status_conditions:
            query = query.where(or_(*status_conditions))

    # Include transaction data if requested
    if include_transactions:
        query = query.options(selectinload(Anomaly.transaction))

    # Apply transaction type filter if provided
    if transaction_types and include_transactions:
        query = query.join(Anomaly.transaction).where(
            Transaction.transaction_type.in_(transaction_types)
        )

    # Apply search term if provided
    if search_term:
        search_pattern = f"%{search_term}%"
        if include_transactions:
            query = query.join(Anomaly.transaction).where(
                or_(
                    Anomaly.description.ilike(search_pattern),
                    Transaction.description.ilike(search_pattern),
                    Transaction.counterparty.ilike(search_pattern),
                    Transaction.memo.ilike(search_pattern),
                )
            )
        else:
            query = query.where(Anomaly.description.ilike(search_pattern))

    # Order by risk score (highest first) and created date (newest first)
    query = query.order_by(desc(Anomaly.risk_score), desc(Anomaly.created_at))

    # Count total results
    count_query = select(Anomaly.id).where(query.whereclause)
    result = await db.execute(count_query)
    total_count = len(result.all())

    # Apply pagination
    query = query.offset((page - 1) * page_size).limit(page_size)

    # Execute query
    result = await db.execute(query)
    anomalies = result.scalars().all()

    # Convert to response model
    items = []
    for anomaly in anomalies:
        transaction_model = None
        if include_transactions and anomaly.transaction:
            txn = anomaly.transaction
            transaction_model = TransactionModel(
                id=txn.id,
                date=txn.transaction_date,
                amount=txn.amount,
                description=txn.description,
                type=txn.transaction_type,
                account_id=txn.account_id,
                counterparty=txn.counterparty,
                category=txn.category,
                memo=txn.memo,
            )

        items.append(
            AnomalyResponse(
                id=anomaly.id,
                tenant_id=anomaly.tenant_id,
                transaction_id=anomaly.transaction_id,
                anomaly_type=anomaly.anomaly_type,
                rule_name=getattr(anomaly, "rule_name", None),
                description=anomaly.description,
                risk_score=anomaly.risk_score,
                risk_level=get_risk_level(anomaly.risk_score),
                explanation=anomaly.explanation,
                is_resolved=anomaly.is_resolved,
                created_at=anomaly.created_at,
                updated_at=anomaly.updated_at,
                feedback=anomaly.feedback,
                feedback_notes=anomaly.feedback_notes,
                transaction=transaction_model,
                detection_metadata=anomaly.detection_metadata,
            )
        )

    # Log audit action
    await log_audit_action(
        db=db,
        action=AuditAction(
            action="list_anomalies",
            user_id=current_user.id,
            tenant_id=current_tenant.id,
            resource_type="anomaly",
            details={
                "page": page,
                "page_size": page_size,
                "filters": {
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                    "min_risk_score": min_risk_score,
                    "max_risk_score": max_risk_score,
                    "anomaly_types": (
                        [t.value for t in anomaly_types] if anomaly_types else None
                    ),
                    "statuses": [s.value for s in statuses] if statuses else None,
                    "transaction_types": transaction_types,
                    "search_term": search_term,
                },
                "result_count": len(items),
            },
        ),
    )

    return AnomalyListResponse(
        items=items,
        total=total_count,
        page=page,
        page_size=page_size,
        has_more=total_count > page * page_size,
    )


@router.get(
    "/{anomaly_id}",
    response_model=AnomalyResponse,
    summary="Get a single anomaly by ID",
)
async def get_anomaly(
    anomaly_id: UUID = Path(..., description="Anomaly ID"),
    include_transaction: bool = Query(
        True, description="Include transaction details in response"
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    current_tenant: Tenant = Depends(get_current_tenant),
) -> AnomalyResponse:
    """Get a single anomaly by ID.

    Args:
        anomaly_id: Anomaly ID
        include_transaction: Include transaction details in response
        db: Database session
        current_user: Current authenticated user
        current_tenant: Current tenant

    Returns:
        AnomalyResponse: Anomaly details

    Raises:
        HTTPException: If anomaly not found or not accessible
    """
    # Build query
    query = select(Anomaly).where(
        Anomaly.id == anomaly_id,
        Anomaly.tenant_id == current_tenant.id,
    )

    # Include transaction data if requested
    if include_transaction:
        query = query.options(selectinload(Anomaly.transaction))

    # Execute query
    result = await db.execute(query)
    anomaly = result.scalar_one_or_none()

    if not anomaly:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Anomaly not found",
        )

    # Convert to response model
    transaction_model = None
    if include_transaction and anomaly.transaction:
        txn = anomaly.transaction
        transaction_model = TransactionModel(
            id=txn.id,
            date=txn.transaction_date,
            amount=txn.amount,
            description=txn.description,
            type=txn.transaction_type,
            account_id=txn.account_id,
            counterparty=txn.counterparty,
            category=txn.category,
            memo=txn.memo,
        )

    # Log audit action
    await log_audit_action(
        db=db,
        action=AuditAction(
            action="get_anomaly",
            user_id=current_user.id,
            tenant_id=current_tenant.id,
            resource_type="anomaly",
            resource_id=str(anomaly_id),
        ),
    )

    return AnomalyResponse(
        id=anomaly.id,
        tenant_id=anomaly.tenant_id,
        transaction_id=anomaly.transaction_id,
        anomaly_type=anomaly.anomaly_type,
        rule_name=getattr(anomaly, "rule_name", None),
        description=anomaly.description,
        risk_score=anomaly.risk_score,
        risk_level=get_risk_level(anomaly.risk_score),
        explanation=anomaly.explanation,
        is_resolved=anomaly.is_resolved,
        created_at=anomaly.created_at,
        updated_at=anomaly.updated_at,
        feedback=anomaly.feedback,
        feedback_notes=anomaly.feedback_notes,
        transaction=transaction_model,
        detection_metadata=anomaly.detection_metadata,
    )


@router.post(
    "/{anomaly_id}/feedback",
    response_model=AnomalyResponse,
    summary="Provide feedback on an anomaly",
)
async def add_anomaly_feedback(
    anomaly_id: UUID = Path(..., description="Anomaly ID"),
    feedback: AnomalyFeedback = Body(..., description="Feedback details"),
    send_notifications: bool = Query(
        False, description="Send notifications about this feedback"
    ),
    trigger_learning: bool = Query(
        False, description="Trigger immediate learning based on feedback"
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    current_tenant: Tenant = Depends(get_current_tenant),
) -> AnomalyResponse:
    """Provide feedback on an anomaly.

    Args:
        anomaly_id: Anomaly ID
        feedback: Feedback details
        send_notifications: Send notifications about this feedback
        trigger_learning: Trigger immediate learning based on feedback
        db: Database session
        current_user: Current authenticated user
        current_tenant: Current tenant

    Returns:
        AnomalyResponse: Updated anomaly details

    Raises:
        HTTPException: If anomaly not found or not accessible
    """
    # Check if anomaly exists and belongs to this tenant
    query = (
        select(Anomaly)
        .where(
            Anomaly.id == anomaly_id,
            Anomaly.tenant_id == current_tenant.id,
        )
        .options(selectinload(Anomaly.transaction))
    )

    result = await db.execute(query)
    anomaly = result.scalar_one_or_none()

    if not anomaly:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Anomaly not found",
        )

    # Update anomaly with feedback
    update_values = {
        "feedback": feedback.feedback_type.value,
        "feedback_notes": feedback.notes,
        "updated_at": datetime.utcnow(),
    }

    # Automatically resolve if false positive
    if feedback.feedback_type == FeedbackType.FALSE_POSITIVE:
        update_values["is_resolved"] = True
        update_values["resolved_by"] = current_user.id
        update_values["resolved_at"] = datetime.utcnow()

    stmt = update(Anomaly).where(Anomaly.id == anomaly_id).values(**update_values)

    await db.execute(stmt)
    await db.commit()

    # Refresh anomaly from database
    await db.refresh(anomaly)

    # Send notifications if requested
    if send_notifications or feedback.should_notify:
        # This would call the notification service to send notifications
        # For now, just log it
        logger.info(
            f"Notification about anomaly feedback would be sent: "
            f"Anomaly {anomaly_id}, feedback: {feedback.feedback_type.value}"
        )

        # In a real implementation, this would call the notification service:
        # from auditpulse_mvp.alerts.notification_service import send_anomaly_notification
        # await send_anomaly_notification(anomaly_id)

    # Log audit action
    await log_audit_action(
        db=db,
        action=AuditAction(
            action="add_anomaly_feedback",
            user_id=current_user.id,
            tenant_id=current_tenant.id,
            resource_type="anomaly",
            resource_id=str(anomaly_id),
            details={
                "feedback_type": feedback.feedback_type.value,
                "notes": feedback.notes,
                "send_notifications": send_notifications or feedback.should_notify,
                "trigger_learning": trigger_learning,
            },
        ),
    )

    # Determine if we should trigger immediate learning
    should_trigger_learning = trigger_learning

    if feedback.feedback_type == FeedbackType.FALSE_POSITIVE and not trigger_learning:
        # Check if we've received enough false positive feedback of the same type
        # to warrant an immediate learning update
        rule_name = (
            anomaly.detection_metadata.get("rule_name")
            if anomaly.detection_metadata
            else None
        )

        if rule_name or anomaly.anomaly_type == AnomalyType.ML_BASED:
            # Count recent false positives for the same rule or ML
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)

            if rule_name:
                # Count false positives for the specific rule
                fp_query = (
                    select(func.count())
                    .select_from(Anomaly)
                    .where(
                        Anomaly.tenant_id == current_tenant.id,
                        Anomaly.feedback == FeedbackType.FALSE_POSITIVE.value,
                        Anomaly.updated_at >= thirty_days_ago,
                        Anomaly.detection_metadata.has_key("rule_name"),
                        Anomaly.detection_metadata["rule_name"].astext == rule_name,
                    )
                )
            else:
                # Count ML-based false positives
                fp_query = (
                    select(func.count())
                    .select_from(Anomaly)
                    .where(
                        Anomaly.tenant_id == current_tenant.id,
                        Anomaly.feedback == FeedbackType.FALSE_POSITIVE.value,
                        Anomaly.updated_at >= thirty_days_ago,
                        Anomaly.anomaly_type == AnomalyType.ML_BASED,
                    )
                )

            fp_result = await db.execute(fp_query)
            fp_count = fp_result.scalar_one()

            # If we have enough false positives, trigger learning
            threshold = settings.false_positive_threshold or 5
            if fp_count >= threshold:
                should_trigger_learning = True
                logger.info(
                    f"Triggering immediate learning due to {fp_count} false positives "
                    f"for {'rule ' + rule_name if rule_name else 'ML model'}"
                )

    # Trigger learning if requested or threshold reached
    if should_trigger_learning:
        try:
            # Run learning asynchronously - don't wait for completion
            asyncio.create_task(update_thresholds_from_feedback(current_tenant.id, db))
            logger.info(f"Triggered learning process for tenant {current_tenant.id}")
        except Exception as e:
            # Log but don't fail the request
            logger.error(f"Failed to trigger learning process: {e}")

    # Convert to response model
    transaction_model = None
    if anomaly.transaction:
        txn = anomaly.transaction
        transaction_model = TransactionModel(
            id=txn.id,
            date=txn.transaction_date,
            amount=txn.amount,
            description=txn.description,
            type=txn.transaction_type,
            account_id=txn.account_id,
            counterparty=txn.counterparty,
            category=txn.category,
            memo=txn.memo,
        )

    return AnomalyResponse(
        id=anomaly.id,
        tenant_id=anomaly.tenant_id,
        transaction_id=anomaly.transaction_id,
        anomaly_type=anomaly.anomaly_type,
        rule_name=getattr(anomaly, "rule_name", None),
        description=anomaly.description,
        risk_score=anomaly.risk_score,
        risk_level=get_risk_level(anomaly.risk_score),
        explanation=anomaly.explanation,
        is_resolved=anomaly.is_resolved,
        created_at=anomaly.created_at,
        updated_at=anomaly.updated_at,
        feedback=anomaly.feedback,
        feedback_notes=anomaly.feedback_notes,
        transaction=transaction_model,
        detection_metadata=anomaly.detection_metadata,
    )
