"""User database models."""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import Column, String, Boolean, DateTime, JSON, ForeignKey, Table
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship

from auditpulse_mvp.database.base import Base

# Many-to-many association table for user roles
user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", UUID(as_uuid=True), ForeignKey("users.id"), primary_key=True),
    Column("role_id", UUID(as_uuid=True), ForeignKey("roles.id"), primary_key=True),
)


class User(Base):
    """User model."""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    phone_number = Column(String, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    notification_preferences = Column(
        JSON, default=dict, nullable=False
    )  # Channel preferences for different notification types
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    last_login = Column(DateTime, nullable=True)

    # Relationships
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    organization_id = Column(
        UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=True
    )
    organization = relationship("Organization", back_populates="users")
    models = relationship("Model", back_populates="created_by")
    audit_logs = relationship("AuditLog", back_populates="user")
    notifications = relationship("Notification", back_populates="user")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "email": self.email,
            "full_name": self.full_name,
            "phone_number": self.phone_number,
            "is_active": self.is_active,
            "is_superuser": self.is_superuser,
            "organization_id": (
                str(self.organization_id) if self.organization_id else None
            ),
            "roles": [role.name for role in self.roles] if self.roles else [],
            "notification_preferences": self.notification_preferences,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }

    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        for role in self.roles:
            if permission in role.permissions:
                return True
        return False

    @property
    def default_notification_channels(self) -> Dict[str, List[str]]:
        """Get the default notification channels for different notification types."""
        if not self.notification_preferences:
            # Default preferences
            return {
                "anomaly_detection": ["email"],
                "system_alert": ["email"],
                "model_performance": ["email"],
                "account_security": ["email"],
                "scheduled_reports": ["email"],
            }
        return self.notification_preferences.get("channels", {})

    def get_channels_for_notification_type(self, notification_type: str) -> List[str]:
        """Get the channels for a specific notification type."""
        channels = self.default_notification_channels
        return channels.get(notification_type, ["email"])


class Role(Base):
    """Role model for user permissions."""

    __tablename__ = "roles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=False)
    description = Column(String, nullable=True)
    permissions = Column(ARRAY(String), default=list, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    users = relationship("User", secondary=user_roles, back_populates="roles")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "permissions": self.permissions,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
