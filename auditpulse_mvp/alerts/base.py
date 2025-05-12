"""Base classes for notification providers.

This module defines the base classes for notification providers and notification payloads.
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union

import uuid
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NotificationPriority(str, Enum):
    """Priority levels for notifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationStatus(str, Enum):
    """Status of a notification."""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"


class NotificationPayload(BaseModel):
    """Notification payload model with details about the anomaly to notify."""
    
    # Identification
    tenant_id: uuid.UUID
    anomaly_id: uuid.UUID
    transaction_id: uuid.UUID
    
    # Content
    subject: str
    message: str
    risk_level: str
    risk_score: float
    explanation: Optional[str] = None
    
    # Links and actions
    dashboard_url: str
    action_links: Dict[str, str] = Field(default_factory=dict)
    
    # Metadata
    priority: NotificationPriority = NotificationPriority.MEDIUM
    time: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            uuid.UUID: lambda id: str(id),
        }


class NotificationProvider(ABC):
    """Base abstract class for notification providers."""
    
    def __init__(self):
        """Initialize the notification provider."""
        self.name = self.__class__.__name__
        logger.info(f"Initializing {self.name}")
    
    @abstractmethod
    async def send(self, recipient: str, payload: NotificationPayload) -> NotificationStatus:
        """Send a notification to a recipient.
        
        Args:
            recipient: The recipient address/ID (email, Slack ID, phone number)
            payload: The notification payload
            
        Returns:
            NotificationStatus: Status of the sent notification
        """
        pass
    
    @abstractmethod
    async def is_configured(self) -> bool:
        """Check if the provider is properly configured.
        
        Returns:
            bool: True if the provider is configured, False otherwise
        """
        pass
    
    def format_message(self, payload: NotificationPayload) -> str:
        """Format the notification message with standard templates.
        
        Args:
            payload: The notification payload
            
        Returns:
            str: Formatted message
        """
        # Base implementation with a simple format
        risk_emoji = self._get_risk_emoji(payload.risk_level)
        
        message = (
            f"{risk_emoji} *AUDIT ALERT: {payload.subject}*\n\n"
            f"Risk Level: {payload.risk_level.upper()} ({payload.risk_score:.0f}%)\n"
            f"Transaction ID: {payload.transaction_id}\n\n"
        )
        
        if payload.explanation:
            message += f"Explanation: {payload.explanation}\n\n"
            
        message += f"View Details: {payload.dashboard_url}\n"
        
        return message
    
    def _get_risk_emoji(self, risk_level: str) -> str:
        """Get an emoji representing the risk level.
        
        Args:
            risk_level: The risk level
            
        Returns:
            str: Emoji corresponding to the risk level
        """
        risk_emojis = {
            "negligible": "✅",
            "low": "🔵",
            "medium": "🟡",
            "high": "🔴",
            "critical": "⚠️",
        }
        return risk_emojis.get(risk_level.lower(), "🔍") 