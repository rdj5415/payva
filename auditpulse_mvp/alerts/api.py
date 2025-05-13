"""API endpoints for the notification system.

This module provides API endpoints for testing and managing notifications.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status, Body
from pydantic import BaseModel, UUID4, Field
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.alerts.notification_service import (
    NotificationService,
    get_notification_service,
    NotificationStatus,
)
from auditpulse_mvp.database.session import get_db
from auditpulse_mvp.api.deps import get_current_user, require_admin
from auditpulse_mvp.database.models import User, Anomaly

logger = logging.getLogger(__name__)

router = APIRouter()


class NotificationResponse(BaseModel):
    """Response model for notification status."""

    success: bool
    status: Dict[str, str]
    message: str


class NotificationSettingsUpdate(BaseModel):
    """Model for updating notification settings for a user."""

    email_notifications: Optional[bool] = None
    slack_notifications: Optional[bool] = None
    sms_notifications: Optional[bool] = None
    slack_user_id: Optional[str] = None
    phone_number: Optional[str] = None


class NotificationSettings(BaseModel):
    """Model for user notification settings."""

    email_notifications: bool
    slack_notifications: bool
    sms_notifications: bool
    email: str
    slack_user_id: Optional[str] = None
    phone_number: Optional[str] = None


@router.post(
    "/test/{notification_type}",
    response_model=NotificationResponse,
    summary="Test notification delivery",
    status_code=status.HTTP_200_OK,
)
async def test_notification(
    notification_type: str,
    recipient: str = Body(..., description="Email, Slack ID, or phone number"),
    subject: str = Body(..., description="Notification subject"),
    message: str = Body(..., description="Notification message"),
    risk_level: str = Body(
        "medium", description="Risk level (negligible, low, medium, high, critical)"
    ),
    risk_score: float = Body(50.0, description="Risk score (0-100)"),
    current_user: User = Depends(require_admin),
):
    """Test sending a notification to the specified recipient.

    Only admins can use this endpoint.

    Args:
        notification_type: Type of notification (email, slack, sms)
        recipient: Recipient address (email, Slack ID, or phone number)
        subject: Notification subject
        message: Notification message
        risk_level: Risk level
        risk_score: Risk score

    Returns:
        NotificationResponse: Response with status of the notification
    """
    try:
        # Validate notification type
        if notification_type not in ["email", "slack", "sms"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid notification type: {notification_type}",
            )

        # Get notification service
        service = get_notification_service()

        # Create a test payload
        from auditpulse_mvp.alerts.base import NotificationPayload, NotificationPriority

        payload = NotificationPayload(
            tenant_id=current_user.tenant_id,
            anomaly_id=uuid.uuid4(),
            transaction_id=uuid.uuid4(),
            subject=subject,
            message=message,
            risk_level=risk_level,
            risk_score=risk_score,
            explanation="This is a test notification from the AuditPulse API.",
            dashboard_url="/dashboard",
            priority=NotificationPriority.MEDIUM,
        )

        # Send the notification based on type
        result = NotificationStatus.FAILED

        if notification_type == "email":
            result = await service.email_provider.send(recipient, payload)
        elif notification_type == "slack":
            result = await service.slack_provider.send(recipient, payload)
        elif notification_type == "sms":
            result = await service.sms_provider.send(recipient, payload)

        # Return the result
        success = result == NotificationStatus.SENT

        return NotificationResponse(
            success=success,
            status={notification_type: result.value},
            message=(
                f"Test notification sent successfully to {recipient}"
                if success
                else f"Failed to send test notification to {recipient}"
            ),
        )

    except Exception as e:
        logger.exception(f"Error sending test notification: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error sending test notification: {str(e)}",
        )


@router.post(
    "/send/{anomaly_id}",
    response_model=NotificationResponse,
    summary="Send notification for an anomaly",
    status_code=status.HTTP_200_OK,
)
async def send_anomaly_notification(
    anomaly_id: UUID4,
    force: bool = Query(
        False, description="Force sending even if notification was already sent"
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Send notifications for an existing anomaly.

    Args:
        anomaly_id: ID of the anomaly to send notifications for
        force: Whether to force sending even if notification was already sent

    Returns:
        NotificationResponse: Response with status of the notification
    """
    try:
        # Verify the anomaly exists and belongs to the user's tenant
        stmt = "SELECT id, tenant_id FROM anomalies WHERE id = :anomaly_id"
        result = await db.execute(stmt, {"anomaly_id": anomaly_id})
        anomaly = result.first()

        if not anomaly:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Anomaly not found: {anomaly_id}",
            )

        if anomaly.tenant_id != current_user.tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to send notifications for this anomaly",
            )

        # Get notification service and send
        service = get_notification_service()
        results = await service.send_anomaly_notification(anomaly_id, db, force)

        # Check if any notification was sent successfully
        success = any(status == NotificationStatus.SENT for status in results.values())

        return NotificationResponse(
            success=success,
            status={k: v.value for k, v in results.items()},
            message=(
                f"Notification sent successfully for anomaly {anomaly_id}"
                if success
                else f"Failed to send notification for anomaly {anomaly_id}"
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error sending anomaly notification: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error sending anomaly notification: {str(e)}",
        )


@router.get(
    "/settings",
    response_model=NotificationSettings,
    summary="Get notification settings for the current user",
    status_code=status.HTTP_200_OK,
)
async def get_notification_settings(
    current_user: User = Depends(get_current_user),
):
    """Get notification settings for the current user.

    Returns:
        NotificationSettings: The user's notification settings
    """
    return NotificationSettings(
        email_notifications=current_user.email_notifications,
        slack_notifications=current_user.slack_notifications,
        sms_notifications=current_user.sms_notifications,
        email=current_user.email,
        slack_user_id=current_user.slack_user_id,
        phone_number=current_user.phone_number,
    )


@router.patch(
    "/settings",
    response_model=NotificationSettings,
    summary="Update notification settings for the current user",
    status_code=status.HTTP_200_OK,
)
async def update_notification_settings(
    settings_update: NotificationSettingsUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update notification settings for the current user.

    Args:
        settings_update: The settings to update

    Returns:
        NotificationSettings: The updated notification settings
    """
    try:
        # Build update dictionary with only provided fields
        update_data = {}
        for field, value in settings_update.dict(exclude_unset=True).items():
            if value is not None:
                update_data[field] = value

        if not update_data:
            return get_notification_settings(current_user)

        # Update the user
        stmt = f"""
            UPDATE users 
            SET {', '.join(f'{k} = :{k}' for k in update_data.keys())}
            WHERE id = :user_id
        """

        params = {**update_data, "user_id": current_user.id}
        await db.execute(stmt, params)
        await db.commit()

        # Refresh the user object
        for field, value in update_data.items():
            setattr(current_user, field, value)

        return NotificationSettings(
            email_notifications=current_user.email_notifications,
            slack_notifications=current_user.slack_notifications,
            sms_notifications=current_user.sms_notifications,
            email=current_user.email,
            slack_user_id=current_user.slack_user_id,
            phone_number=current_user.phone_number,
        )

    except Exception as e:
        logger.exception(f"Error updating notification settings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating notification settings: {str(e)}",
        )
