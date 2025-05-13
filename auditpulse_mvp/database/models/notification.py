"""Database models for notifications and notification templates."""

import uuid
from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    DateTime,
    ForeignKey,
    Boolean,
    JSON,
    Enum,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from auditpulse_mvp.database.base import Base
from auditpulse_mvp.notifications.channels import NotificationChannel


class NotificationTemplate(Base):
    """Notification template model."""

    __tablename__ = "notification_templates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    template_id = Column(String(100), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    subject = Column(String(255), nullable=False)
    body = Column(Text, nullable=False)
    html_body = Column(Text, nullable=True)
    placeholders = Column(Text, nullable=True)  # JSON string of required placeholders
    version = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    notifications = relationship("Notification", back_populates="template")


class Notification(Base):
    """Notification model."""

    __tablename__ = "notifications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    template_id = Column(
        String(100), ForeignKey("notification_templates.template_id"), nullable=False
    )
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    recipient = Column(JSON, nullable=False)  # Email, phone, slack channel, etc.
    template_data = Column(JSON, nullable=True)  # Data used for template rendering
    status = Column(
        String(50), index=True, default="pending", nullable=False
    )  # pending, processing, delivered, failed
    priority = Column(
        String(20), default="medium", nullable=False
    )  # low, medium, high, critical
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    scheduled_at = Column(DateTime, nullable=True)  # For scheduled notifications
    processed_at = Column(DateTime, nullable=True)  # When notification was processed

    # Relationships
    template = relationship("NotificationTemplate", back_populates="notifications")
    user = relationship("User", back_populates="notifications")
    delivery_attempts = relationship(
        "NotificationDeliveryAttempt", back_populates="notification"
    )

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "template_id": self.template_id,
            "user_id": str(self.user_id) if self.user_id else None,
            "recipient": self.recipient,
            "status": self.status,
            "priority": self.priority,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "scheduled_at": (
                self.scheduled_at.isoformat() if self.scheduled_at else None
            ),
            "processed_at": (
                self.processed_at.isoformat() if self.processed_at else None
            ),
        }


class NotificationDeliveryAttempt(Base):
    """Notification delivery attempt model."""

    __tablename__ = "notification_delivery_attempts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    notification_id = Column(
        UUID(as_uuid=True), ForeignKey("notifications.id"), nullable=False
    )
    channel = Column(String(50), nullable=False)  # email, sms, slack, webhook
    status = Column(
        String(50), default="pending", nullable=False
    )  # pending, delivered, failed
    response = Column(JSON, nullable=True)  # Response from the notification service
    error = Column(Text, nullable=True)  # Error message if failed
    attempt_number = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    notification = relationship("Notification", back_populates="delivery_attempts")

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "notification_id": str(self.notification_id),
            "channel": self.channel,
            "status": self.status,
            "response": self.response,
            "error": self.error,
            "attempt_number": self.attempt_number,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
