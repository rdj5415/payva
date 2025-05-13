"""
Notification service for AuditPulse MVP.
This module handles sending notifications via email and Slack for anomalies and system events.
"""

from .service import NotificationService, NotificationType, NotificationPriority
from .providers import EmailProvider, SlackProvider

__all__ = [
    "NotificationService",
    "NotificationType",
    "NotificationPriority",
    "EmailProvider",
    "SlackProvider",
] 