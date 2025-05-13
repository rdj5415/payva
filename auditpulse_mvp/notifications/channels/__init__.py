"""Notification channel factory and initialization."""

from enum import Enum, auto
from typing import Dict, Any, Optional, Type

from auditpulse_mvp.utils.settings import Settings
from auditpulse_mvp.notifications.channels.email import EmailNotifier
from auditpulse_mvp.notifications.channels.slack import SlackNotifier
from auditpulse_mvp.notifications.channels.sms import SMSNotifier
from auditpulse_mvp.notifications.channels.webhook import WebhookNotifier
from auditpulse_mvp.notifications.channels.base import NotificationProvider


class NotificationChannel(str, Enum):
    """Supported notification channels."""

    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    WEBHOOK = "webhook"


class NotificationChannelFactory:
    """Factory for creating notification channel instances."""

    def __init__(self, settings: Settings):
        """Initialize the notification channel factory.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._channels: Dict[str, Type[NotificationProvider]] = {}

    def get_channel(self, channel_type: NotificationChannel):
        """Get or create a notification channel instance.

        Args:
            channel_type: Type of notification channel

        Returns:
            Channel instance
        """
        if channel_type not in self._channels:
            self._channels[channel_type] = self._create_channel(channel_type)

        return self._channels[channel_type]

    def _create_channel(self, channel_type: NotificationChannel):
        """Create a new notification channel instance.

        Args:
            channel_type: Type of notification channel

        Returns:
            Channel instance
        """
        if channel_type == NotificationChannel.EMAIL:
            return EmailNotifier(self.settings)
        elif channel_type == NotificationChannel.SLACK:
            return SlackNotifier(self.settings)
        elif channel_type == NotificationChannel.SMS:
            return SMSNotifier(self.settings)
        elif channel_type == NotificationChannel.WEBHOOK:
            return WebhookNotifier(self.settings)
        else:
            raise ValueError(f"Unsupported notification channel: {channel_type}")


_channels: Dict[str, Type[NotificationProvider]] = {
    "email": EmailNotifier,
    "slack": SlackNotifier,
    "webhook": WebhookNotifier,
}


def get_provider(provider_type: str) -> Type[NotificationProvider]:
    """Get a notification provider by type.

    Args:
        provider_type: The type of provider to get

    Returns:
        The provider class
    """
    if provider_type not in _channels:
        raise ValueError(f"Unsupported notification provider: {provider_type}")
    return _channels[provider_type]
