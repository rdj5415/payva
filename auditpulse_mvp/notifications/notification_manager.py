"""Notification manager for sending alerts through multiple channels."""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid

from fastapi import Depends
from pydantic import BaseModel, EmailStr, Field, validator
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.database.models import (
    Notification,
    NotificationChannel,
    NotificationStatus,
    User,
)
from auditpulse_mvp.notifications.channels.email import EmailNotifier
from auditpulse_mvp.notifications.channels.slack import SlackNotifier
from auditpulse_mvp.notifications.channels.sms import SmsNotifier
from auditpulse_mvp.notifications.channels.webhook import WebhookNotifier
from auditpulse_mvp.tasks.task_manager import TaskManager, TaskPriority
from auditpulse_mvp.utils.settings import get_settings, Settings

logger = logging.getLogger(__name__)


class NotificationPriority(str, Enum):
    """Notification priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationTemplate(BaseModel):
    """Notification template model."""

    template_id: str
    subject: str
    body: str
    placeholders: Dict[str, str] = Field(default_factory=dict)


class NotificationRecipient(BaseModel):
    """Notification recipient model."""

    user_id: Optional[uuid.UUID] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    slack_channel: Optional[str] = None
    webhook_url: Optional[str] = None

    @validator("phone")
    def validate_phone(cls, v):
        """Validate phone number format."""
        if v and not v.startswith("+"):
            raise ValueError("Phone number must start with '+' and country code")
        return v

    @property
    def has_destination(self) -> bool:
        """Check if recipient has at least one destination."""
        return bool(self.email or self.phone or self.slack_channel or self.webhook_url)


class NotificationRequest(BaseModel):
    """Notification request model."""

    template_id: str
    recipient: NotificationRecipient
    template_data: Dict[str, Any] = Field(default_factory=dict)
    priority: NotificationPriority = NotificationPriority.MEDIUM
    channels: List[NotificationChannel] = Field(
        default_factory=lambda: [
            NotificationChannel.EMAIL,
            NotificationChannel.SLACK,
        ]
    )


class NotificationManager:
    """Manages sending notifications through multiple channels."""

    def __init__(
        self,
        db_session: AsyncSession = Depends(get_db_session),
        task_manager: TaskManager = Depends(),
        settings: Settings = Depends(get_settings),
    ):
        """Initialize the notification manager.

        Args:
            db_session: Database session
            task_manager: Task manager instance
            settings: Application settings
        """
        self.db = db_session
        self.task_manager = task_manager
        self.settings = settings

        # Initialize channels
        self.email_notifier = EmailNotifier(settings=settings)
        self.slack_notifier = SlackNotifier(settings=settings)
        self.sms_notifier = SmsNotifier(settings=settings)
        self.webhook_notifier = WebhookNotifier(settings=settings)

    async def send_notification(
        self,
        request: NotificationRequest,
    ) -> Dict[str, Any]:
        """Send a notification through multiple channels.

        Args:
            request: Notification request

        Returns:
            Dict[str, Any]: Status of notification delivery
        """
        # Get template
        template = await self._get_template(request.template_id)

        if not template:
            logger.error(f"Template not found: {request.template_id}")
            return {
                "status": "error",
                "message": f"Template not found: {request.template_id}",
            }

        # Get user if user_id is provided
        user = None
        if request.recipient.user_id:
            stmt = select(User).where(User.id == request.recipient.user_id)
            result = await self.db.execute(stmt)
            user = result.scalar_one_or_none()

            if not user:
                logger.error(f"User not found: {request.recipient.user_id}")
                return {
                    "status": "error",
                    "message": f"User not found: {request.recipient.user_id}",
                }

        # Create notification records for each channel
        notification_ids = {}
        task_ids = {}

        for channel in request.channels:
            # Create notification record
            notification = await self._create_notification(
                template=template,
                recipient=request.recipient,
                template_data=request.template_data,
                channel=channel,
                priority=request.priority,
                user=user,
            )

            notification_ids[channel] = str(notification.id)

            # Schedule delivery task
            task_priority = self._get_task_priority(request.priority)
            task_id = await self.task_manager.schedule_task(
                task_name="send_notification",
                priority=task_priority,
                kwargs={
                    "notification_id": str(notification.id),
                    "channel": channel,
                },
            )

            task_ids[channel] = task_id

        return {
            "status": "scheduled",
            "notification_ids": notification_ids,
            "task_ids": task_ids,
        }

    async def send_batch_notification(
        self,
        template_id: str,
        recipients: List[NotificationRecipient],
        template_data: Dict[str, Any],
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        channels: Optional[List[NotificationChannel]] = None,
    ) -> Dict[str, Any]:
        """Send a notification to multiple recipients.

        Args:
            template_id: Template ID
            recipients: List of recipients
            template_data: Template data
            priority: Notification priority
            channels: List of channels to use

        Returns:
            Dict[str, Any]: Status of batch notification delivery
        """
        if channels is None:
            channels = [NotificationChannel.EMAIL, NotificationChannel.SLACK]

        # Create batch notification task
        task_priority = self._get_task_priority(priority)
        task_id = await self.task_manager.schedule_task(
            task_name="send_batch_notification",
            priority=task_priority,
            kwargs={
                "template_id": template_id,
                "recipients": [recipient.dict() for recipient in recipients],
                "template_data": template_data,
                "priority": priority,
                "channels": channels,
            },
        )

        return {
            "status": "scheduled",
            "task_id": task_id,
            "recipients_count": len(recipients),
        }

    async def get_notification_status(
        self,
        notification_id: Union[str, uuid.UUID],
    ) -> Dict[str, Any]:
        """Get the status of a notification.

        Args:
            notification_id: Notification ID

        Returns:
            Dict[str, Any]: Notification status
        """
        # Convert string ID to UUID if needed
        if isinstance(notification_id, str):
            notification_id = uuid.UUID(notification_id)

        # Get notification
        stmt = select(Notification).where(Notification.id == notification_id)
        result = await self.db.execute(stmt)
        notification = result.scalar_one_or_none()

        if not notification:
            logger.error(f"Notification not found: {notification_id}")
            return {
                "status": "error",
                "message": f"Notification not found: {notification_id}",
            }

        return {
            "notification_id": str(notification.id),
            "status": notification.status,
            "channel": notification.channel,
            "created_at": notification.created_at,
            "sent_at": notification.sent_at,
            "delivered_at": notification.delivered_at,
            "error": notification.error,
        }

    async def _get_template(self, template_id: str) -> Optional[NotificationTemplate]:
        """Get a notification template by ID.

        Args:
            template_id: Template ID

        Returns:
            Optional[NotificationTemplate]: Notification template
        """
        # In a real implementation, this would retrieve templates from a database or file
        # For now, we'll use a hardcoded template
        templates = {
            "welcome": NotificationTemplate(
                template_id="welcome",
                subject="Welcome to AuditPulse",
                body="Hello {name}, welcome to AuditPulse!",
                placeholders={"name": "User's name"},
            ),
            "anomaly_alert": NotificationTemplate(
                template_id="anomaly_alert",
                subject="Anomaly Detected: {anomaly_type}",
                body="An anomaly of type {anomaly_type} has been detected in your account {account_name}. Confidence score: {confidence}.",
                placeholders={
                    "anomaly_type": "Type of anomaly",
                    "account_name": "Account name",
                    "confidence": "Confidence score",
                },
            ),
            "risk_alert": NotificationTemplate(
                template_id="risk_alert",
                subject="Risk Alert: {risk_level}",
                body="A {risk_level} risk has been identified: {description}",
                placeholders={
                    "risk_level": "Risk level (High, Medium, Low)",
                    "description": "Risk description",
                },
            ),
        }

        return templates.get(template_id)

    async def _create_notification(
        self,
        template: NotificationTemplate,
        recipient: NotificationRecipient,
        template_data: Dict[str, Any],
        channel: NotificationChannel,
        priority: NotificationPriority,
        user: Optional[User] = None,
    ) -> Notification:
        """Create a notification record in the database.

        Args:
            template: Notification template
            recipient: Notification recipient
            template_data: Template data
            channel: Notification channel
            priority: Notification priority
            user: User object

        Returns:
            Notification: Created notification
        """
        # Render template
        subject = template.subject
        body = template.body

        for key, value in template_data.items():
            placeholder = f"{{{key}}}"
            subject = subject.replace(placeholder, str(value))
            body = body.replace(placeholder, str(value))

        # Determine recipient info based on channel
        recipient_info = {}

        if channel == NotificationChannel.EMAIL and recipient.email:
            recipient_info["email"] = recipient.email
        elif channel == NotificationChannel.SMS and recipient.phone:
            recipient_info["phone"] = recipient.phone
        elif channel == NotificationChannel.SLACK and recipient.slack_channel:
            recipient_info["slack_channel"] = recipient.slack_channel
        elif channel == NotificationChannel.WEBHOOK and recipient.webhook_url:
            recipient_info["webhook_url"] = recipient.webhook_url

        # Create notification record
        notification = Notification(
            id=uuid.uuid4(),
            user_id=user.id if user else None,
            channel=channel,
            status=NotificationStatus.PENDING,
            priority=priority,
            subject=subject,
            body=body,
            recipient_info=recipient_info,
            template_id=template.template_id,
            template_data=template_data,
            created_at=datetime.now(),
        )

        self.db.add(notification)
        await self.db.commit()
        await self.db.refresh(notification)

        return notification

    def _get_task_priority(
        self, notification_priority: NotificationPriority
    ) -> TaskPriority:
        """Map notification priority to task priority.

        Args:
            notification_priority: Notification priority

        Returns:
            TaskPriority: Task priority
        """
        priority_map = {
            NotificationPriority.LOW: TaskPriority.LOW,
            NotificationPriority.MEDIUM: TaskPriority.MEDIUM,
            NotificationPriority.HIGH: TaskPriority.HIGH,
            NotificationPriority.CRITICAL: TaskPriority.CRITICAL,
        }

        return priority_map.get(notification_priority, TaskPriority.MEDIUM)
