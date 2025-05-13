"""
Notification providers for email and Slack.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
from ..notifications.service import NotificationPriority

logger = logging.getLogger(__name__)


class NotificationProvider(ABC):
    """Base class for notification providers."""

    @abstractmethod
    async def send(
        self,
        recipients: List[str],
        subject: str,
        body: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
    ):
        """Send a notification.

        Args:
            recipients: List of recipient addresses
            subject: Notification subject
            body: Notification body
            priority: Notification priority
        """
        pass


class EmailProvider(NotificationProvider):
    """Email notification provider using SMTP."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        use_tls: bool = True,
    ):
        """Initialize the email provider.

        Args:
            smtp_host: SMTP server hostname
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_email: Sender email address
            use_tls: Whether to use TLS
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.use_tls = use_tls

    async def send(
        self,
        recipients: List[str],
        subject: str,
        body: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
    ):
        """Send an email notification.

        Args:
            recipients: List of recipient email addresses
            subject: Email subject
            body: Email body
            priority: Notification priority
        """
        try:
            message = MIMEMultipart()
            message["From"] = self.from_email
            message["To"] = ", ".join(recipients)
            message["Subject"] = subject

            # Add priority header
            priority_map = {
                NotificationPriority.HIGH: "High",
                NotificationPriority.MEDIUM: "Normal",
                NotificationPriority.LOW: "Low",
            }
            message["X-Priority"] = priority_map.get(priority, "Normal")

            # Add body
            message.attach(MIMEText(body, "plain"))

            # Send email
            async with aiosmtplib.SMTP(
                hostname=self.smtp_host,
                port=self.smtp_port,
                use_tls=self.use_tls,
            ) as smtp:
                await smtp.login(self.username, self.password)
                await smtp.send_message(message)

            logger.info(f"Email sent to {len(recipients)} recipients")
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            raise


class SlackProvider(NotificationProvider):
    """Slack notification provider using webhooks."""

    def __init__(self, default_webhook_url: Optional[str] = None):
        """Initialize the Slack provider.

        Args:
            default_webhook_url: Default Slack webhook URL
        """
        self.default_webhook_url = default_webhook_url

    async def send(
        self,
        webhook_url: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
    ):
        """Send a Slack notification.

        Args:
            webhook_url: Slack webhook URL
            message: Message to send
            priority: Notification priority
        """
        try:
            # Use default webhook URL if none provided
            webhook_url = webhook_url or self.default_webhook_url
            if not webhook_url:
                raise ValueError("No webhook URL provided")

            # Add priority indicator to message
            priority_indicators = {
                NotificationPriority.HIGH: "🔴",
                NotificationPriority.MEDIUM: "🟡",
                NotificationPriority.LOW: "🟢",
            }
            message = f"{priority_indicators.get(priority, '')} {message}"

            # Send message
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json={"text": message},
                ) as response:
                    response.raise_for_status()

            logger.info("Slack message sent successfully")
        except Exception as e:
            logger.error(f"Error sending Slack message: {e}")
            raise
