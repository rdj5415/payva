"""Notification system for AuditPulse MVP.

This package provides a notification system for sending alerts about detected anomalies
through different channels (email, Slack, SMS).
"""

from auditpulse_mvp.alerts.base import NotificationProvider, NotificationPayload
from auditpulse_mvp.alerts.email import EmailNotificationProvider
from auditpulse_mvp.alerts.slack import SlackNotificationProvider
from auditpulse_mvp.alerts.sms import SMSNotificationProvider
from auditpulse_mvp.alerts.notification_service import NotificationService

__all__ = [
    "NotificationProvider",
    "NotificationPayload",
    "EmailNotificationProvider",
    "SlackNotificationProvider",
    "SMSNotificationProvider",
    "NotificationService",
]
