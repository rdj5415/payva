"""
Configuration for the notification service.
"""

import os
from typing import Optional

from pydantic import EmailStr, HttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class NotificationSettings(BaseSettings):
    """Settings for the notification service."""

    # Email settings
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_FROM_EMAIL: Optional[EmailStr] = None
    SMTP_USE_TLS: bool = True

    # Slack settings
    SLACK_DEFAULT_WEBHOOK: Optional[HttpUrl] = None

    # Notification settings
    NOTIFICATION_QUEUE_SIZE: int = 1000
    NOTIFICATION_RETRY_ATTEMPTS: int = 3
    NOTIFICATION_RETRY_DELAY: int = 5  # seconds

    model_config = SettingsConfigDict(
        env_prefix="AUDITPULSE_",
        case_sensitive=True,
    )


# Create settings instance
settings = NotificationSettings()

# Validate required settings
if not settings.SMTP_USERNAME:
    raise ValueError("SMTP_USERNAME is required")
if not settings.SMTP_PASSWORD:
    raise ValueError("SMTP_PASSWORD is required")
if not settings.SMTP_FROM_EMAIL:
    raise ValueError("SMTP_FROM_EMAIL is required")
