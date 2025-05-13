"""Notification service for AuditPulse MVP.

This module provides functionality for sending notifications about anomalies
and system events via email and Slack.
"""
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional

import aiohttp
import jinja2
from fastapi import HTTPException
from pydantic import BaseModel, EmailStr

from auditpulse_mvp.database.models import Anomaly, Tenant
from auditpulse_mvp.utils.settings import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Initialize settings
settings = get_settings()

# Initialize Jinja2 environment
template_loader = jinja2.FileSystemLoader(searchpath="./templates")
template_env = jinja2.Environment(loader=template_loader)


class NotificationConfig(BaseModel):
    """Configuration for notification settings."""
    email_enabled: bool = True
    slack_enabled: bool = False
    email_recipients: List[EmailStr] = []
    slack_webhook_url: Optional[str] = None
    notification_threshold: str = "HIGH"  # Only notify for HIGH risk anomalies


class NotificationService:
    """Service for sending notifications about anomalies and system events."""
    
    def __init__(self, db_session):
        """Initialize the notification service.
        
        Args:
            db_session: Database session.
        """
        self.db_session = db_session
    
    async def send_anomaly_notification(
        self,
        tenant_id: str,
        anomaly: Anomaly,
    ) -> bool:
        """Send notification about a new anomaly.
        
        Args:
            tenant_id: Tenant ID.
            anomaly: Anomaly to notify about.
            
        Returns:
            True if notification was sent successfully.
        """
        # Get tenant
        tenant = self.db_session.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            raise HTTPException(status_code=404, detail="Tenant not found")
        
        # Get notification config
        config = self._get_notification_config(tenant)
        
        # Check if we should notify based on risk level
        if anomaly.risk_level.value < config.notification_threshold:
            logger.info(f"Skipping notification for {anomaly.id} - risk level too low")
            return True
        
        # Prepare notification content
        content = self._prepare_anomaly_content(anomaly)
        
        # Send notifications
        success = True
        
        if config.email_enabled:
            try:
                await self._send_email(
                    recipients=config.email_recipients,
                    subject=f"New {anomaly.risk_level.value} Risk Anomaly Detected",
                    content=content,
                )
            except Exception as e:
                logger.error(f"Failed to send email notification: {e}")
                success = False
        
        if config.slack_enabled and config.slack_webhook_url:
            try:
                await self._send_slack(
                    webhook_url=config.slack_webhook_url,
                    content=content,
                )
            except Exception as e:
                logger.error(f"Failed to send Slack notification: {e}")
                success = False
        
        return success
    
    async def send_system_notification(
        self,
        tenant_id: str,
        event_type: str,
        message: str,
        details: Optional[Dict] = None,
    ) -> bool:
        """Send notification about a system event.
        
        Args:
            tenant_id: Tenant ID.
            event_type: Type of event (e.g., "error", "warning", "info").
            message: Event message.
            details: Optional event details.
            
        Returns:
            True if notification was sent successfully.
        """
        # Get tenant
        tenant = self.db_session.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            raise HTTPException(status_code=404, detail="Tenant not found")
        
        # Get notification config
        config = self._get_notification_config(tenant)
        
        # Prepare notification content
        content = self._prepare_system_content(event_type, message, details)
        
        # Send notifications
        success = True
        
        if config.email_enabled:
            try:
                await self._send_email(
                    recipients=config.email_recipients,
                    subject=f"System {event_type.title()} - {message}",
                    content=content,
                )
            except Exception as e:
                logger.error(f"Failed to send email notification: {e}")
                success = False
        
        if config.slack_enabled and config.slack_webhook_url:
            try:
                await self._send_slack(
                    webhook_url=config.slack_webhook_url,
                    content=content,
                )
            except Exception as e:
                logger.error(f"Failed to send Slack notification: {e}")
                success = False
        
        return success
    
    def _get_notification_config(self, tenant: Tenant) -> NotificationConfig:
        """Get notification configuration for a tenant.
        
        Args:
            tenant: Tenant to get config for.
            
        Returns:
            Notification configuration.
        """
        # TODO: Get from tenant settings
        return NotificationConfig(
            email_enabled=True,
            slack_enabled=False,
            email_recipients=[tenant.admin_email],
            slack_webhook_url=None,
            notification_threshold="HIGH",
        )
    
    def _prepare_anomaly_content(self, anomaly: Anomaly) -> str:
        """Prepare notification content for an anomaly.
        
        Args:
            anomaly: Anomaly to prepare content for.
            
        Returns:
            Formatted notification content.
        """
        # Load template
        template = template_env.get_template("anomaly_notification.html")
        
        # Render template
        return template.render(
            anomaly=anomaly,
            transaction=anomaly.transaction,
            risk_level=anomaly.risk_level.value,
            anomaly_type=anomaly.anomaly_type.value,
            explanation=anomaly.explanation,
        )
    
    def _prepare_system_content(
        self,
        event_type: str,
        message: str,
        details: Optional[Dict] = None,
    ) -> str:
        """Prepare notification content for a system event.
        
        Args:
            event_type: Type of event.
            message: Event message.
            details: Optional event details.
            
        Returns:
            Formatted notification content.
        """
        # Load template
        template = template_env.get_template("system_notification.html")
        
        # Render template
        return template.render(
            event_type=event_type,
            message=message,
            details=details or {},
        )
    
    async def _send_email(
        self,
        recipients: List[str],
        subject: str,
        content: str,
    ) -> None:
        """Send email notification.
        
        Args:
            recipients: List of email recipients.
            subject: Email subject.
            content: Email content (HTML).
            
        Raises:
            Exception: If email sending fails.
        """
        # Create message
        msg = MIMEMultipart()
        msg["From"] = settings.SMTP_FROM_EMAIL
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        
        # Add content
        msg.attach(MIMEText(content, "html"))
        
        # Send email
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            if settings.SMTP_USE_TLS:
                server.starttls()
            if settings.SMTP_USERNAME and settings.SMTP_PASSWORD:
                server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            server.send_message(msg)
    
    async def _send_slack(
        self,
        webhook_url: str,
        content: str,
    ) -> None:
        """Send Slack notification.
        
        Args:
            webhook_url: Slack webhook URL.
            content: Notification content.
            
        Raises:
            Exception: If Slack notification fails.
        """
        # Prepare payload
        payload = {
            "text": content,
            "mrkdwn": True,
        }
        
        # Send to Slack
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Slack API error: {response.status}") 