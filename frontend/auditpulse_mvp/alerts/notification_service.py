"""Notification service for AuditPulse MVP.

This module provides a service for sending notifications through different channels.
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.alerts.base import (
    NotificationProvider,
    NotificationPayload,
    NotificationStatus,
    NotificationPriority,
)
from auditpulse_mvp.alerts.email import EmailNotificationProvider
from auditpulse_mvp.alerts.slack import SlackNotificationProvider
from auditpulse_mvp.alerts.sms import SMSNotificationProvider
from auditpulse_mvp.alerts.webhook import WebhookNotificationProvider
from auditpulse_mvp.database.models import Anomaly, User, Transaction, Tenant
from auditpulse_mvp.utils.settings import settings
from auditpulse_mvp.database.session import get_db

logger = logging.getLogger(__name__)


class NotificationService:
    """Service for sending notifications about anomalies."""
    
    def __init__(
        self,
        email_provider: Optional[EmailNotificationProvider] = None,
        slack_provider: Optional[SlackNotificationProvider] = None,
        sms_provider: Optional[SMSNotificationProvider] = None,
        webhook_provider: Optional[WebhookNotificationProvider] = None,
    ):
        """Initialize the notification service.
        
        Args:
            email_provider: Custom email provider. If not provided, a default is created.
            slack_provider: Custom Slack provider. If not provided, a default is created.
            sms_provider: Custom SMS provider. If not provided, a default is created.
            webhook_provider: Custom Webhook provider. If not provided, a default is created.
        """
        # Initialize providers
        self.email_provider = email_provider or EmailNotificationProvider()
        self.slack_provider = slack_provider or SlackNotificationProvider()
        self.sms_provider = sms_provider or SMSNotificationProvider()
        self.webhook_provider = webhook_provider or WebhookNotificationProvider()
        
        logger.info("Notification service initialized")
    
    async def send_anomaly_notification(
        self, 
        anomaly_id: uuid.UUID, 
        db: AsyncSession,
        force: bool = False,
    ) -> Dict[str, NotificationStatus]:
        """Send notifications for an anomaly to relevant users.
        
        Args:
            anomaly_id: ID of the anomaly to notify about
            db: Database session
            force: Whether to force sending even if notification was already sent
            
        Returns:
            Dict[str, NotificationStatus]: Status of notifications by channel (email, slack, sms, webhook)
        """
        try:
            # Get the anomaly with transaction and tenant information
            anomaly = await self._get_anomaly_with_related(anomaly_id, db)
            
            if not anomaly:
                logger.error(f"Anomaly not found: {anomaly_id}")
                return {
                    "email": NotificationStatus.FAILED,
                    "slack": NotificationStatus.FAILED,
                    "sms": NotificationStatus.FAILED,
                    "webhook": NotificationStatus.FAILED,
                }
            
            # Skip if notification already sent and not forcing
            if anomaly.notification_sent and not force:
                logger.info(f"Notification already sent for anomaly {anomaly_id}")
                return {
                    "email": NotificationStatus.SENT,
                    "slack": NotificationStatus.SENT,
                    "sms": NotificationStatus.SENT,
                    "webhook": NotificationStatus.SENT,
                }
            
            # Get users who should be notified (admins and auditors)
            users = await self._get_notification_recipients(anomaly.tenant_id, db)
            
            if not users:
                logger.warning(f"No notification recipients found for tenant {anomaly.tenant_id}")
                return {
                    "email": NotificationStatus.FAILED,
                    "slack": NotificationStatus.FAILED,
                    "sms": NotificationStatus.FAILED,
                    "webhook": NotificationStatus.FAILED,
                }
                
            # Create notification payload
            payload = self._create_notification_payload(anomaly)
            
            # Send notifications through each channel
            results = {
                "email": NotificationStatus.FAILED,
                "slack": NotificationStatus.FAILED,
                "sms": NotificationStatus.FAILED,
                "webhook": NotificationStatus.FAILED,
            }
            
            # Send email notifications
            if await self.email_provider.is_configured():
                email_recipients = [
                    user.email for user in users 
                    if user.email_notifications and user.email
                ]
                
                if email_recipients:
                    for recipient in email_recipients:
                        status = await self.email_provider.send(recipient, payload)
                        if status == NotificationStatus.SENT:
                            results["email"] = NotificationStatus.SENT
            
            # Send Slack notifications
            if await self.slack_provider.is_configured():
                slack_recipients = [
                    user.slack_user_id for user in users 
                    if user.slack_notifications and user.slack_user_id
                ]
                
                if slack_recipients:
                    for recipient in slack_recipients:
                        status = await self.slack_provider.send(recipient, payload)
                        if status == NotificationStatus.SENT:
                            results["slack"] = NotificationStatus.SENT
            
            # Send SMS notifications
            if await self.sms_provider.is_configured():
                sms_recipients = [
                    user.phone_number for user in users 
                    if user.sms_notifications and user.phone_number
                ]
                
                if sms_recipients:
                    for recipient in sms_recipients:
                        status = await self.sms_provider.send(recipient, payload)
                        if status == NotificationStatus.SENT:
                            results["sms"] = NotificationStatus.SENT
            
            # Send webhook notifications
            if await self.webhook_provider.is_configured():
                webhook_recipients = [
                    user.webhook_url for user in users
                    if hasattr(user, 'webhook_notifications') and user.webhook_notifications and user.webhook_url
                ]
                if webhook_recipients:
                    for recipient in webhook_recipients:
                        status = await self.webhook_provider.send(recipient, payload)
                        if status == NotificationStatus.SENT:
                            results["webhook"] = NotificationStatus.SENT
            
            # Update anomaly to mark notification as sent
            if any(status == NotificationStatus.SENT for status in results.values()):
                await self._mark_notification_sent(anomaly_id, db)
            
            return results
            
        except Exception as e:
            logger.exception(f"Error sending notifications for anomaly {anomaly_id}: {str(e)}")
            return {
                "email": NotificationStatus.FAILED,
                "slack": NotificationStatus.FAILED,
                "sms": NotificationStatus.FAILED,
                "webhook": NotificationStatus.FAILED,
            }
    
    async def notify_high_risk_anomalies(self, db: AsyncSession) -> int:
        """Send notifications for all new high-risk anomalies.
        
        Args:
            db: Database session
            
        Returns:
            int: Number of anomalies for which notifications were sent
        """
        try:
            # Get all high-risk anomalies that haven't been notified yet
            stmt = select(Anomaly).where(
                (Anomaly.notification_sent == False) &
                (Anomaly.risk_score >= settings.RISK_SCORE_THRESHOLD) &
                (Anomaly.is_resolved == False)
            )
            
            result = await db.execute(stmt)
            anomalies = result.scalars().all()
            
            if not anomalies:
                logger.info("No new high-risk anomalies to notify about")
                return 0
            
            # Send notifications for each anomaly
            notification_count = 0
            for anomaly in anomalies:
                results = await self.send_anomaly_notification(anomaly.id, db)
                if any(status == NotificationStatus.SENT for status in results.values()):
                    notification_count += 1
            
            logger.info(f"Sent notifications for {notification_count} high-risk anomalies")
            return notification_count
            
        except Exception as e:
            logger.exception(f"Error sending notifications for high-risk anomalies: {str(e)}")
            return 0
    
    def _create_notification_payload(self, anomaly: Anomaly) -> NotificationPayload:
        """Create a notification payload from an anomaly.
        
        Args:
            anomaly: The anomaly to create a payload for
            
        Returns:
            NotificationPayload: The notification payload
        """
        # Determine risk level and priority
        risk_level = self._get_risk_level(anomaly.risk_score)
        priority = self._get_priority_from_risk(anomaly.risk_score)
        
        # Create dashboard URL
        dashboard_url = (
            f"{settings.FRONTEND_URL}/dashboard/anomalies/{anomaly.id}"
            if hasattr(settings, "FRONTEND_URL") else
            f"/dashboard/anomalies/{anomaly.id}"
        )
        
        # Create payload
        payload = NotificationPayload(
            tenant_id=anomaly.tenant_id,
            anomaly_id=anomaly.id,
            transaction_id=anomaly.transaction_id,
            subject=f"Unusual Transaction: {anomaly.description}",
            message=anomaly.description,
            risk_level=risk_level,
            risk_score=anomaly.risk_score,
            explanation=anomaly.explanation,
            dashboard_url=dashboard_url,
            priority=priority,
            action_links={
                "Review": f"{dashboard_url}/review",
                "Dismiss": f"{dashboard_url}/dismiss"
            },
            metadata={
                "transaction_type": getattr(anomaly.transaction, "transaction_type", "unknown"),
                "rule_name": getattr(anomaly, "rule_name", "unknown"),
                "anomaly_type": anomaly.anomaly_type,
            },
        )
        
        return payload
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Get a risk level string from a risk score.
        
        Args:
            risk_score: Risk score (0-100)
            
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
    
    def _get_priority_from_risk(self, risk_score: float) -> NotificationPriority:
        """Get notification priority from a risk score.
        
        Args:
            risk_score: Risk score (0-100)
            
        Returns:
            NotificationPriority: Priority level
        """
        if risk_score < 40:
            return NotificationPriority.LOW
        elif risk_score < 70:
            return NotificationPriority.MEDIUM
        elif risk_score < 90:
            return NotificationPriority.HIGH
        else:
            return NotificationPriority.CRITICAL
    
    async def _get_anomaly_with_related(
        self, anomaly_id: uuid.UUID, db: AsyncSession
    ) -> Optional[Anomaly]:
        """Get an anomaly with its related transaction and tenant information.
        
        Args:
            anomaly_id: ID of the anomaly
            db: Database session
            
        Returns:
            Optional[Anomaly]: The anomaly with related information, or None if not found
        """
        stmt = (
            select(Anomaly)
            .where(Anomaly.id == anomaly_id)
            .execution_options(populate_existing=True)
        )
        result = await db.execute(stmt)
        anomaly = result.scalar_one_or_none()
        
        if anomaly:
            # Load related transaction
            stmt = select(Transaction).where(Transaction.id == anomaly.transaction_id)
            result = await db.execute(stmt)
            anomaly.transaction = result.scalar_one_or_none()
        
        return anomaly
    
    async def _get_notification_recipients(
        self, tenant_id: uuid.UUID, db: AsyncSession
    ) -> List[User]:
        """Get users who should receive notifications for a tenant.
        
        Args:
            tenant_id: ID of the tenant
            db: Database session
            
        Returns:
            List[User]: List of users who should receive notifications
        """
        # Get admins and auditors for the tenant
        stmt = (
            select(User)
            .where(
                (User.tenant_id == tenant_id) &
                (User.is_active == True) &
                (User.role.in_(["admin", "auditor"]))
            )
        )
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def _mark_notification_sent(self, anomaly_id: uuid.UUID, db: AsyncSession) -> None:
        """Mark an anomaly as having been notified about.
        
        Args:
            anomaly_id: ID of the anomaly
            db: Database session
        """
        stmt = (
            update(Anomaly)
            .where(Anomaly.id == anomaly_id)
            .values(
                notification_sent=True,
                notification_sent_at=datetime.now(),
            )
        )
        await db.execute(stmt)
        await db.commit()
        
        logger.info(f"Marked notification as sent for anomaly {anomaly_id}")


# Singleton instance for convenience
_notification_service = None


def get_notification_service() -> NotificationService:
    """Get the global notification service instance.
    
    Returns:
        NotificationService: The notification service
    """
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service


async def send_anomaly_notification(anomaly_id: uuid.UUID) -> Dict[str, NotificationStatus]:
    """Send notifications for an anomaly.
    
    This is a convenience function that uses the global notification service.
    
    Args:
        anomaly_id: ID of the anomaly to notify about
        
    Returns:
        Dict[str, NotificationStatus]: Status of notifications by channel
    """
    service = get_notification_service()
    async with get_db() as db:
        return await service.send_anomaly_notification(anomaly_id, db) 