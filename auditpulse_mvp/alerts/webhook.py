"""Webhook notification provider for AuditPulse AI.

This module provides a notification provider for sending alerts via webhooks (e.g., Zapier, custom endpoints).
"""

import logging
from typing import Optional
import httpx

from auditpulse_mvp.alerts.base import (
    NotificationProvider,
    NotificationPayload,
    NotificationStatus,
)
from auditpulse_mvp.utils.settings import settings

logger = logging.getLogger(__name__)


class WebhookNotificationProvider(NotificationProvider):
    """Webhook notification provider."""

    def __init__(self, default_webhook_url: Optional[str] = None):
        """Initialize the webhook notification provider.
        Args:
            default_webhook_url: Default webhook URL to use if not provided per notification.
        """
        super().__init__()
        self.default_webhook_url = default_webhook_url or getattr(
            settings, "WEBHOOK_URL", None
        )

    async def is_configured(self) -> bool:
        """Check if the provider is properly configured."""
        return self.default_webhook_url is not None

    async def send(
        self, recipient: str, payload: NotificationPayload
    ) -> NotificationStatus:
        """Send a webhook notification.
        Args:
            recipient: Webhook URL (overrides default if provided)
            payload: The notification payload
        Returns:
            NotificationStatus: Status of the sent notification
        """
        webhook_url = recipient or self.default_webhook_url
        if not webhook_url:
            logger.warning(
                "Webhook URL not configured. Cannot send webhook notification."
            )
            return NotificationStatus.FAILED
        try:
            data = {
                "subject": payload.subject,
                "message": payload.message,
                "risk_level": payload.risk_level,
                "risk_score": payload.risk_score,
                "transaction_id": str(payload.transaction_id),
                "anomaly_id": str(payload.anomaly_id),
                "explanation": payload.explanation,
                "dashboard_url": payload.dashboard_url,
                "priority": (
                    payload.priority.value if hasattr(payload, "priority") else None
                ),
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(webhook_url, json=data, timeout=10.0)
            if response.status_code in (200, 201, 202):
                logger.info(f"Webhook notification sent to {webhook_url}")
                return NotificationStatus.SENT
            else:
                logger.error(f"Failed to send webhook notification: {response.text}")
                return NotificationStatus.FAILED
        except Exception as e:
            logger.exception(f"Error sending webhook notification: {str(e)}")
            return NotificationStatus.FAILED
