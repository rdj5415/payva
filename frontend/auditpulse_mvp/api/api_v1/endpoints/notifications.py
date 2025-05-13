"""API endpoints for notification management."""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field, validator

from auditpulse_mvp.api.deps import get_current_user, get_db
from auditpulse_mvp.database.models.user import User
from auditpulse_mvp.database.models.notification import (
    Notification,
    NotificationTemplate,
    NotificationDeliveryAttempt,
)
from auditpulse_mvp.notifications.notification_manager import (
    NotificationManager,
    NotificationRecipient,
    NotificationRequest,
    NotificationPriority,
)
from auditpulse_mvp.notifications.templates import TemplateManager
from auditpulse_mvp.utils.settings import get_settings
from auditpulse_mvp.tasks.task_manager import TaskManager

router = APIRouter()
settings = get_settings()


# Pydantic models
class TemplateCreate(BaseModel):
    """Schema for creating a notification template."""

    template_id: str = Field(..., description="Unique template identifier")
    name: str = Field(..., description="Template name")
    subject: str = Field(..., description="Template subject")
    body: str = Field(..., description="Template body (plain text)")
    html_body: Optional[str] = Field(None, description="Template HTML body")
    description: Optional[str] = Field(None, description="Template description")
    placeholders: Optional[Dict[str, str]] = Field(
        None, description="Template placeholder descriptions"
    )


class TemplateResponse(BaseModel):
    """Schema for notification template response."""

    template_id: str
    name: str
    subject: str
    description: Optional[str] = None
    version: int
    created_at: datetime
    updated_at: datetime


class TemplateDetailResponse(TemplateResponse):
    """Schema for detailed notification template response."""

    body: str
    html_body: Optional[str] = None
    placeholders: Optional[Dict[str, str]] = None


class NotificationCreate(BaseModel):
    """Schema for creating a notification."""

    template_id: str = Field(..., description="Template identifier")
    recipient: NotificationRecipient = Field(..., description="Notification recipient")
    template_data: Optional[Dict[str, Any]] = Field(
        {}, description="Data for template rendering"
    )
    priority: NotificationPriority = Field(
        NotificationPriority.MEDIUM, description="Notification priority"
    )
    channels: Optional[List[str]] = Field(
        None, description="Channels to send notification through"
    )
    scheduled_at: Optional[datetime] = Field(
        None, description="When to send the notification"
    )


class NotificationResponse(BaseModel):
    """Schema for notification response."""

    id: UUID
    template_id: str
    status: str
    priority: str
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None


class NotificationDetailResponse(NotificationResponse):
    """Schema for detailed notification response."""

    recipient: Dict[str, Any]
    template_data: Optional[Dict[str, Any]] = None
    delivery_attempts: List[Dict[str, Any]] = []


class NotificationBatchCreate(BaseModel):
    """Schema for creating multiple notifications at once."""

    notifications: List[NotificationCreate] = Field(
        ..., description="List of notifications to create"
    )
    max_concurrent: Optional[int] = Field(
        10, description="Maximum number of concurrent notification processing tasks"
    )


class NotificationBatchResponse(BaseModel):
    """Schema for batch notification response."""

    status: str
    total: int
    success_count: int
    error_count: int
    notification_ids: List[str]


class NotificationPreferencesUpdate(BaseModel):
    """Schema for updating notification preferences."""

    channels: Dict[str, List[str]] = Field(
        ..., description="Channel preferences for different notification types"
    )

    @validator("channels")
    def validate_channels(cls, v):
        """Validate that channels are valid."""
        valid_types = [
            "anomaly_detection",
            "system_alert",
            "model_performance",
            "account_security",
            "scheduled_reports",
        ]
        valid_channels = ["email", "slack", "sms", "webhook"]

        for notification_type, channels in v.items():
            if notification_type not in valid_types:
                raise ValueError(f"Invalid notification type: {notification_type}")

            for channel in channels:
                if channel not in valid_channels:
                    raise ValueError(f"Invalid channel: {channel}")

        return v


# Template endpoints
@router.post("/templates", response_model=TemplateDetailResponse, status_code=201)
async def create_template(
    template: TemplateCreate,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """Create a new notification template."""
    if not current_user.is_superuser and not current_user.has_permission(
        "manage_templates"
    ):
        raise HTTPException(status_code=403, detail="Not enough permissions")

    template_manager = TemplateManager(settings)

    try:
        template_data = await template_manager.create_template(
            template_id=template.template_id,
            name=template.name,
            subject=template.subject,
            body=template.body,
            html_body=template.html_body,
            description=template.description,
            placeholders=template.placeholders,
        )
        return template_data
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Could not create template: {str(e)}"
        )


@router.get("/templates", response_model=List[TemplateResponse])
async def list_templates(
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """List all notification templates."""
    template_manager = TemplateManager(settings)
    templates = await template_manager.list_templates()
    return templates


@router.get("/templates/{template_id}", response_model=TemplateDetailResponse)
async def get_template(
    template_id: str = Path(..., description="Template identifier"),
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """Get a notification template by ID."""
    template_manager = TemplateManager(settings)
    template = await template_manager.get_template(template_id, refresh=True)

    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    return template


@router.delete("/templates/{template_id}", status_code=204)
async def delete_template(
    template_id: str = Path(..., description="Template identifier"),
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """Delete a notification template."""
    if not current_user.is_superuser and not current_user.has_permission(
        "manage_templates"
    ):
        raise HTTPException(status_code=403, detail="Not enough permissions")

    template_manager = TemplateManager(settings)
    deleted = await template_manager.delete_template(template_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Template not found")

    return None


# Notification endpoints
@router.post("/notifications", response_model=NotificationResponse, status_code=201)
async def create_notification(
    notification: NotificationCreate,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
    task_manager: TaskManager = Depends(lambda: TaskManager.get_instance()),
):
    """Create a new notification."""
    notification_manager = NotificationManager(settings, task_manager)

    # Verify template exists
    template_manager = TemplateManager(settings)
    template = await template_manager.get_template(notification.template_id)

    if not template:
        raise HTTPException(
            status_code=404, detail=f"Template '{notification.template_id}' not found"
        )

    try:
        # Create notification request
        request = NotificationRequest(
            template_id=notification.template_id,
            recipient=notification.recipient,
            template_data=notification.template_data or {},
            priority=notification.priority,
            channels=notification.channels,
        )

        # Send notification
        result = await notification_manager.send_notification(
            request=request,
            user_id=current_user.id,
            scheduled_at=notification.scheduled_at,
        )

        # Get the notification from the database
        db_notification = (
            await db.query(Notification)
            .filter(Notification.id == result["notification_id"])
            .first()
        )

        if not db_notification:
            raise HTTPException(status_code=500, detail="Failed to create notification")

        return db_notification

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Could not create notification: {str(e)}"
        )


@router.post("/notifications/batch", response_model=NotificationBatchResponse)
async def create_batch_notifications(
    batch: NotificationBatchCreate,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
    task_manager: TaskManager = Depends(lambda: TaskManager.get_instance()),
):
    """Create multiple notifications at once."""
    notification_manager = NotificationManager(settings, task_manager)

    # Validate templates exist
    template_manager = TemplateManager(settings)
    template_ids = {n.template_id for n in batch.notifications}

    for template_id in template_ids:
        template = await template_manager.get_template(template_id)
        if not template:
            raise HTTPException(
                status_code=404, detail=f"Template '{template_id}' not found"
            )

    try:
        # Convert to notification requests
        requests = []
        for notification in batch.notifications:
            request = NotificationRequest(
                template_id=notification.template_id,
                recipient=notification.recipient,
                template_data=notification.template_data or {},
                priority=notification.priority,
                channels=notification.channels,
            )
            requests.append(
                {
                    "request": request,
                    "user_id": current_user.id,
                    "scheduled_at": notification.scheduled_at,
                }
            )

        # Send batch notifications
        result = await notification_manager.send_batch_notification(
            notifications=requests,
            max_concurrent=batch.max_concurrent,
        )

        return {
            "status": result["status"],
            "total": result["total"],
            "success_count": result["success_count"],
            "error_count": result["error_count"],
            "notification_ids": [str(n.id) for n in result["notifications"]],
        }

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Could not create batch notifications: {str(e)}"
        )


@router.get("/notifications", response_model=List[NotificationResponse])
async def list_notifications(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, description="Maximum number of notifications to return"),
    offset: int = Query(0, description="Pagination offset"),
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """List notifications for the current user."""
    query = db.query(Notification).filter(Notification.user_id == current_user.id)

    if status:
        query = query.filter(Notification.status == status)

    total = await query.count()

    notifications = (
        await query.order_by(Notification.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return notifications


@router.get(
    "/notifications/{notification_id}", response_model=NotificationDetailResponse
)
async def get_notification(
    notification_id: UUID = Path(..., description="Notification identifier"),
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """Get a notification by ID."""
    notification = (
        await db.query(Notification)
        .filter(
            Notification.id == notification_id,
            Notification.user_id == current_user.id,
        )
        .first()
    )

    if not notification:
        # Check if the user is an admin, who can view any notification
        if current_user.is_superuser or current_user.has_permission(
            "view_all_notifications"
        ):
            notification = (
                await db.query(Notification)
                .filter(Notification.id == notification_id)
                .first()
            )

        if not notification:
            raise HTTPException(status_code=404, detail="Notification not found")

    # Get delivery attempts
    delivery_attempts = (
        await db.query(NotificationDeliveryAttempt)
        .filter(NotificationDeliveryAttempt.notification_id == notification.id)
        .all()
    )

    # Convert to response format
    result = notification.to_dict()
    result["delivery_attempts"] = [attempt.to_dict() for attempt in delivery_attempts]

    return result


@router.get("/notifications/{notification_id}/status", response_model=Dict[str, Any])
async def get_notification_status(
    notification_id: UUID = Path(..., description="Notification identifier"),
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
    task_manager: TaskManager = Depends(lambda: TaskManager.get_instance()),
):
    """Get the status of a notification."""
    notification_manager = NotificationManager(settings, task_manager)

    # Check if notification exists and belongs to the user
    notification = (
        await db.query(Notification)
        .filter(
            Notification.id == notification_id,
            Notification.user_id == current_user.id,
        )
        .first()
    )

    if not notification:
        # Check if the user is an admin, who can view any notification
        if current_user.is_superuser or current_user.has_permission(
            "view_all_notifications"
        ):
            notification = (
                await db.query(Notification)
                .filter(Notification.id == notification_id)
                .first()
            )

        if not notification:
            raise HTTPException(status_code=404, detail="Notification not found")

    # Get notification status
    status = await notification_manager.get_notification_status(str(notification_id))

    return status


@router.put("/notifications/preferences", response_model=Dict[str, Any])
async def update_notification_preferences(
    preferences: NotificationPreferencesUpdate,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """Update notification preferences for the current user."""
    # Update user preferences
    current_user.notification_preferences = {
        "channels": preferences.channels,
    }

    await db.commit()

    return {
        "status": "success",
        "message": "Notification preferences updated",
        "notification_preferences": current_user.notification_preferences,
    }


@router.get("/notifications/preferences", response_model=Dict[str, Any])
async def get_notification_preferences(
    current_user: User = Depends(get_current_user),
):
    """Get notification preferences for the current user."""
    return {
        "notification_preferences": current_user.notification_preferences or {},
        "default_channels": current_user.default_notification_channels,
    }
