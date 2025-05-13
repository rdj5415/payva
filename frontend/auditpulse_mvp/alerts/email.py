"""Email notification provider using SendGrid.

This module provides a notification provider for sending email alerts via SendGrid.
"""

import logging
from typing import Dict, List, Optional, Any

import httpx
from pydantic import EmailStr

from auditpulse_mvp.alerts.base import (
    NotificationProvider,
    NotificationPayload,
    NotificationStatus,
)
from auditpulse_mvp.utils.settings import settings

logger = logging.getLogger(__name__)


class EmailNotificationProvider(NotificationProvider):
    """Email notification provider using SendGrid."""

    def __init__(self, api_key: Optional[str] = None, from_email: str = None):
        """Initialize the email notification provider.

        Args:
            api_key: SendGrid API key. If not provided, uses settings.
            from_email: Sender email address. If not provided, uses a default.
        """
        super().__init__()
        self.api_key = api_key or (
            settings.SENDGRID_API_KEY.get_secret_value()
            if settings.SENDGRID_API_KEY
            else None
        )
        self.from_email = from_email or "alerts@auditpulse.ai"
        self.base_url = "https://api.sendgrid.com/v3/mail/send"

    async def is_configured(self) -> bool:
        """Check if the provider is properly configured.

        Returns:
            bool: True if the provider is configured, False otherwise
        """
        return self.api_key is not None

    async def send(
        self, recipient: str, payload: NotificationPayload
    ) -> NotificationStatus:
        """Send an email notification.

        Args:
            recipient: Email address of the recipient
            payload: The notification payload

        Returns:
            NotificationStatus: Status of the sent notification
        """
        if not self.is_configured():
            logger.warning(
                "SendGrid is not configured. Cannot send email notification."
            )
            return NotificationStatus.FAILED

        try:
            # Verify recipient email format
            try:
                EmailStr.validate(recipient)
            except ValueError:
                logger.error(f"Invalid email address: {recipient}")
                return NotificationStatus.FAILED

            # Format the HTML content
            html_content = self._format_html_email(payload)

            # Prepare SendGrid API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            data = {
                "personalizations": [
                    {
                        "to": [{"email": recipient}],
                        "subject": payload.subject,
                    }
                ],
                "from": {"email": self.from_email, "name": "AuditPulse Alerts"},
                "content": [{"type": "text/html", "value": html_content}],
            }

            # Add action buttons if available
            if payload.action_links:
                data["personalizations"][0]["dynamic_template_data"] = {
                    "action_links": [
                        {"text": text, "url": url}
                        for text, url in payload.action_links.items()
                    ]
                }

            # Send the request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=data,
                    timeout=10.0,
                )

            if response.status_code >= 200 and response.status_code < 300:
                logger.info(f"Email notification sent to {recipient}")
                return NotificationStatus.SENT
            else:
                logger.error(
                    f"Failed to send email notification: {response.status_code} {response.text}"
                )
                return NotificationStatus.FAILED

        except Exception as e:
            logger.exception(f"Error sending email notification: {str(e)}")
            return NotificationStatus.FAILED

    def _format_html_email(self, payload: NotificationPayload) -> str:
        """Format the notification as HTML for email.

        Args:
            payload: The notification payload

        Returns:
            str: HTML content for the email
        """
        risk_color = self._get_risk_color(payload.risk_level)
        risk_emoji = self._get_risk_emoji(payload.risk_level)

        # Basic styling with inline CSS
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{payload.subject}</title>
        </head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 0; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="text-align: center; margin-bottom: 20px;">
                    <img src="https://auditpulse.ai/logo.png" alt="AuditPulse Logo" style="max-width: 150px;">
                </div>
                
                <div style="background-color: #f9f9f9; border-left: 4px solid {risk_color}; padding: 15px; margin-bottom: 20px;">
                    <h2 style="margin-top: 0; color: #333;">{risk_emoji} Anomaly Detected</h2>
                    <p style="font-size: 18px; font-weight: bold;">{payload.subject}</p>
                </div>
                
                <div style="margin-bottom: 20px;">
                    <p><strong>Risk Level:</strong> <span style="color: {risk_color}; font-weight: bold;">{payload.risk_level.upper()}</span> ({payload.risk_score:.0f}%)</p>
                    <p><strong>Transaction ID:</strong> {payload.transaction_id}</p>
                    <p><strong>Detected At:</strong> {payload.time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
        """

        if payload.explanation:
            html += f"""
                <div style="margin-bottom: 20px; background-color: #f5f5f5; padding: 15px; border-radius: 4px;">
                    <h3 style="margin-top: 0;">Explanation</h3>
                    <p>{payload.explanation}</p>
                </div>
            """

        # Add action buttons
        html += f"""
                <div style="margin-bottom: 20px; text-align: center;">
                    <a href="{payload.dashboard_url}" style="display: inline-block; background-color: #0066cc; color: white; text-decoration: none; padding: 10px 20px; border-radius: 4px; margin-right: 10px;">View Details</a>
        """

        # Add additional action links if available
        for text, url in payload.action_links.items():
            html += f"""
                    <a href="{url}" style="display: inline-block; background-color: #555555; color: white; text-decoration: none; padding: 10px 20px; border-radius: 4px; margin-right: 10px;">{text}</a>
            """

        # Footer
        html += f"""
                </div>
                
                <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 12px; color: #777; text-align: center;">
                    <p>This is an automated notification from AuditPulse. Please do not reply to this email.</p>
                    <p>© 2024 AuditPulse AI • <a href="https://auditpulse.ai/privacy">Privacy Policy</a> • <a href="https://auditpulse.ai/terms">Terms of Service</a></p>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def _get_risk_color(self, risk_level: str) -> str:
        """Get a color representing the risk level.

        Args:
            risk_level: The risk level

        Returns:
            str: Color hex code corresponding to the risk level
        """
        risk_colors = {
            "negligible": "#00cc00",  # Green
            "low": "#3399ff",  # Blue
            "medium": "#ffcc00",  # Yellow
            "high": "#ff3300",  # Red
            "critical": "#cc0000",  # Dark Red
        }
        return risk_colors.get(risk_level.lower(), "#777777")
