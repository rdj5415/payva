"""
Email notification channel.
"""

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Any, Optional, Union

from auditpulse_mvp.utils.settings import Settings

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Email notification channel using SMTP."""

    def __init__(self, settings: Settings):
        """Initialize the email notifier.

        Args:
            settings: Application settings
        """
        self.settings = settings
        # Use properly defined settings or fallback to empty strings to avoid None access
        self.smtp_host = getattr(settings, "SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = getattr(settings, "SMTP_PORT", 587)
        self.smtp_user = getattr(settings, "SMTP_USERNAME", "")
        self.smtp_password = getattr(settings, "SMTP_PASSWORD", "")
        self.sender_email = getattr(settings, "SMTP_FROM_EMAIL", "no-reply@example.com")
        self.use_ssl = getattr(settings, "SMTP_USE_TLS", True)

    async def send(
        self,
        recipient_email: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send an email notification.

        Args:
            recipient_email: Recipient email address
            subject: Email subject
            body: Email body (plain text)
            html_body: Optional HTML body

        Returns:
            Dict[str, Any]: Send result
        """
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.sender_email
            msg["To"] = recipient_email

            # Add plain text and HTML parts
            msg.attach(MIMEText(body, "plain"))
            if html_body:
                msg.attach(MIMEText(html_body, "html"))

            # Connect to SMTP server
            server: Union[smtplib.SMTP, smtplib.SMTP_SSL]
            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port)
            else:
                # Using regular SMTP with STARTTLS
                server = smtplib.SMTP(self.smtp_host, self.smtp_port)
                server.starttls()

            # Login and send
            server.login(self.smtp_user, self.smtp_password)
            server.sendmail(self.sender_email, recipient_email, msg.as_string())
            server.quit()

            logger.info(f"Email sent successfully to {recipient_email}")
            return {
                "status": "delivered",
                "recipient": recipient_email,
                "message_id": msg["Message-ID"],
            }

        except Exception as e:
            logger.error(f"Error sending email to {recipient_email}: {e}")
            return {
                "status": "error",
                "recipient": recipient_email,
                "error": str(e),
            }
