"""
Main notification service implementation.
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

from sqlalchemy.ext.asyncio import AsyncSession

from ..database.models import Anomaly, NotificationConfig, Tenant, User
from .providers import EmailProvider, SlackProvider

logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Types of notifications that can be sent."""

    ANOMALY_DETECTED = "anomaly_detected"
    ANOMALY_RESOLVED = "anomaly_resolved"
    RISK_THRESHOLD_EXCEEDED = "risk_threshold_exceeded"
    SYSTEM_ALERT = "system_alert"
    DAILY_SUMMARY = "daily_summary"


class NotificationPriority(str, Enum):
    """Priority levels for notifications."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class NotificationService:
    """Service for sending notifications via various providers."""

    def __init__(
        self,
        email_provider: Optional[EmailProvider] = None,
        slack_provider: Optional[SlackProvider] = None,
    ):
        """Initialize the notification service.

        Args:
            email_provider: Email notification provider
            slack_provider: Slack notification provider
        """
        self.email_provider = email_provider
        self.slack_provider = slack_provider
        self._notification_queue: asyncio.Queue = asyncio.Queue()
        self._is_running = False

    async def start(self):
        """Start the notification service."""
        if self._is_running:
            return

        self._is_running = True
        asyncio.create_task(self._process_notifications())

    async def stop(self):
        """Stop the notification service."""
        self._is_running = False
        await self._notification_queue.join()

    async def _process_notifications(self):
        """Process notifications from the queue."""
        while self._is_running:
            try:
                notification = await self._notification_queue.get()
                await self._send_notification(notification)
                self._notification_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing notification: {e}")

    async def _send_notification(self, notification: Dict):
        """Send a notification using the appropriate providers.

        Args:
            notification: Notification data
        """
        try:
            if self.email_provider and notification.get("email_enabled", True):
                await self.email_provider.send(
                    recipients=notification["recipients"],
                    subject=notification["subject"],
                    body=notification["body"],
                    priority=notification.get("priority", NotificationPriority.MEDIUM),
                )

            if self.slack_provider and notification.get("slack_enabled", False):
                await self.slack_provider.send(
                    webhook_url=notification["slack_webhook"],
                    message=notification["slack_message"],
                    priority=notification.get("priority", NotificationPriority.MEDIUM),
                )
        except Exception as e:
            logger.error(f"Error sending notification: {e}")

    async def notify_anomaly_detected(
        self,
        session: AsyncSession,
        anomaly: Anomaly,
        tenant: Tenant,
        config: NotificationConfig,
    ):
        """Send notification for a detected anomaly.

        Args:
            session: Database session
            anomaly: Detected anomaly
            tenant: Tenant information
            config: Notification configuration
        """
        if not self._should_notify_anomaly(anomaly, config):
            return

        notification = {
            "type": NotificationType.ANOMALY_DETECTED,
            "priority": self._get_priority_for_risk_level(anomaly.risk_level),
            "email_enabled": config.email_enabled,
            "slack_enabled": config.slack_enabled,
            "recipients": config.email_recipients,
            "slack_webhook": config.slack_webhook,
            "subject": f"New {anomaly.risk_level} Risk Anomaly Detected",
            "body": self._generate_anomaly_email_body(anomaly, tenant),
            "slack_message": self._generate_anomaly_slack_message(anomaly, tenant),
        }

        await self._notification_queue.put(notification)

    async def notify_anomaly_resolved(
        self,
        session: AsyncSession,
        anomaly: Anomaly,
        tenant: Tenant,
        config: NotificationConfig,
    ):
        """Send notification for a resolved anomaly.

        Args:
            session: Database session
            anomaly: Resolved anomaly
            tenant: Tenant information
            config: Notification configuration
        """
        notification = {
            "type": NotificationType.ANOMALY_RESOLVED,
            "priority": NotificationPriority.LOW,
            "email_enabled": config.email_enabled,
            "slack_enabled": config.slack_enabled,
            "recipients": config.email_recipients,
            "slack_webhook": config.slack_webhook,
            "subject": f"Anomaly Resolved: {anomaly.risk_level} Risk",
            "body": self._generate_resolution_email_body(anomaly, tenant),
            "slack_message": self._generate_resolution_slack_message(anomaly, tenant),
        }

        await self._notification_queue.put(notification)

    async def send_daily_summary(
        self,
        session: AsyncSession,
        tenant: Tenant,
        config: NotificationConfig,
        summary_data: Dict,
    ):
        """Send daily summary notification.

        Args:
            session: Database session
            tenant: Tenant information
            config: Notification configuration
            summary_data: Summary data for the day
        """
        if config.notification_frequency != "Daily":
            return

        notification = {
            "type": NotificationType.DAILY_SUMMARY,
            "priority": NotificationPriority.LOW,
            "email_enabled": config.email_enabled,
            "slack_enabled": config.slack_enabled,
            "recipients": config.email_recipients,
            "slack_webhook": config.slack_webhook,
            "subject": f"Daily Anomaly Summary - {tenant.name}",
            "body": self._generate_daily_summary_email_body(summary_data, tenant),
            "slack_message": self._generate_daily_summary_slack_message(
                summary_data, tenant
            ),
        }

        await self._notification_queue.put(notification)

    def _should_notify_anomaly(
        self, anomaly: Anomaly, config: NotificationConfig
    ) -> bool:
        """Check if an anomaly should trigger a notification.

        Args:
            anomaly: Anomaly to check
            config: Notification configuration

        Returns:
            True if notification should be sent, False otherwise
        """
        if anomaly.risk_level == "HIGH" and config.notify_high_risk:
            return True
        if anomaly.risk_level == "MEDIUM" and config.notify_medium_risk:
            return True
        if anomaly.risk_level == "LOW" and config.notify_low_risk:
            return True
        return False

    def _get_priority_for_risk_level(self, risk_level: str) -> NotificationPriority:
        """Get notification priority for a risk level.

        Args:
            risk_level: Risk level

        Returns:
            Notification priority
        """
        if risk_level == "HIGH":
            return NotificationPriority.HIGH
        if risk_level == "MEDIUM":
            return NotificationPriority.MEDIUM
        return NotificationPriority.LOW

    def _generate_anomaly_email_body(self, anomaly: Anomaly, tenant: Tenant) -> str:
        """Generate email body for anomaly notification.

        Args:
            anomaly: Anomaly information
            tenant: Tenant information

        Returns:
            Email body text
        """
        return f"""
        Anomaly Detected - {tenant.name}

        Risk Level: {anomaly.risk_level}
        Type: {anomaly.anomaly_type}
        Score: {anomaly.score:.2f}
        Explanation: {anomaly.explanation}

        Transaction Details:
        Amount: ${anomaly.transaction.amount:.2f}
        Description: {anomaly.transaction.description}
        Merchant: {anomaly.transaction.merchant_name}
        Date: {anomaly.transaction.transaction_date}

        Please review this anomaly in the AuditPulse dashboard.
        """

    def _generate_anomaly_slack_message(self, anomaly: Anomaly, tenant: Tenant) -> str:
        """Generate Slack message for anomaly notification.

        Args:
            anomaly: Anomaly information
            tenant: Tenant information

        Returns:
            Slack message text
        """
        return f"""
        *{anomaly.risk_level} Risk Anomaly Detected*
        Tenant: {tenant.name}
        Type: {anomaly.anomaly_type}
        Score: {anomaly.score:.2f}

        *Transaction Details:*
        Amount: ${anomaly.transaction.amount:.2f}
        Description: {anomaly.transaction.description}
        Merchant: {anomaly.transaction.merchant_name}
        Date: {anomaly.transaction.transaction_date}

        _Explanation: {anomaly.explanation}_
        """

    def _generate_resolution_email_body(self, anomaly: Anomaly, tenant: Tenant) -> str:
        """Generate email body for anomaly resolution notification.

        Args:
            anomaly: Resolved anomaly
            tenant: Tenant information

        Returns:
            Email body text
        """
        return f"""
        Anomaly Resolved - {tenant.name}

        Risk Level: {anomaly.risk_level}
        Type: {anomaly.anomaly_type}
        Feedback: {anomaly.feedback_type}
        Resolution Notes: {anomaly.resolution_notes}

        Transaction Details:
        Amount: ${anomaly.transaction.amount:.2f}
        Description: {anomaly.transaction.description}
        Merchant: {anomaly.transaction.merchant_name}
        Date: {anomaly.transaction.transaction_date}
        """

    def _generate_resolution_slack_message(
        self, anomaly: Anomaly, tenant: Tenant
    ) -> str:
        """Generate Slack message for anomaly resolution notification.

        Args:
            anomaly: Resolved anomaly
            tenant: Tenant information

        Returns:
            Slack message text
        """
        return f"""
        *Anomaly Resolved*
        Tenant: {tenant.name}
        Risk Level: {anomaly.risk_level}
        Type: {anomaly.anomaly_type}
        Feedback: {anomaly.feedback_type}

        *Transaction Details:*
        Amount: ${anomaly.transaction.amount:.2f}
        Description: {anomaly.transaction.description}
        Merchant: {anomaly.transaction.merchant_name}
        Date: {anomaly.transaction.transaction_date}

        _Resolution Notes: {anomaly.resolution_notes}_
        """

    def _generate_daily_summary_email_body(
        self, summary_data: Dict, tenant: Tenant
    ) -> str:
        """Generate email body for daily summary notification.

        Args:
            summary_data: Summary data
            tenant: Tenant information

        Returns:
            Email body text
        """
        return f"""
        Daily Anomaly Summary - {tenant.name}
        Date: {datetime.now().strftime('%Y-%m-%d')}

        Total Anomalies: {summary_data['total_anomalies']}
        High Risk: {summary_data['high_risk_count']}
        Medium Risk: {summary_data['medium_risk_count']}
        Low Risk: {summary_data['low_risk_count']}

        Resolved Today: {summary_data['resolved_count']}
        Average Response Time: {summary_data['avg_response_time']:.1f} hours

        Accuracy: {summary_data['accuracy']:.1%}
        False Positives: {summary_data['false_positives']}
        False Negatives: {summary_data['false_negatives']}
        """

    def _generate_daily_summary_slack_message(
        self, summary_data: Dict, tenant: Tenant
    ) -> str:
        """Generate Slack message for daily summary notification.

        Args:
            summary_data: Summary data
            tenant: Tenant information

        Returns:
            Slack message text
        """
        return f"""
        *Daily Anomaly Summary - {tenant.name}*
        Date: {datetime.now().strftime('%Y-%m-%d')}

        *Anomaly Counts:*
        Total: {summary_data['total_anomalies']}
        High Risk: {summary_data['high_risk_count']}
        Medium Risk: {summary_data['medium_risk_count']}
        Low Risk: {summary_data['low_risk_count']}

        *Resolution Stats:*
        Resolved Today: {summary_data['resolved_count']}
        Avg Response Time: {summary_data['avg_response_time']:.1f} hours

        *Performance:*
        Accuracy: {summary_data['accuracy']:.1%}
        False Positives: {summary_data['false_positives']}
        False Negatives: {summary_data['false_negatives']}
        """
