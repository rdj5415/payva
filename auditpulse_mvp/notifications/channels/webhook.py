"""Webhook notification channel for sending webhook alerts."""

import logging
import json
import aiohttp
from typing import Dict, Any, Optional, Union
from aiohttp import ClientTimeout

from auditpulse_mvp.utils.settings import Settings

logger = logging.getLogger(__name__)


class WebhookNotifier:
    """Webhook notifier for sending webhook notifications."""

    def __init__(self, settings: Settings):
        """Initialize the webhook notifier.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.default_timeout = settings.WEBHOOK_TIMEOUT_SECONDS or 10

    async def send(
        self,
        webhook_url: str,
        payload: Dict[str, Any],
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        auth: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Send a webhook notification.

        Args:
            webhook_url: Webhook URL
            payload: Webhook payload
            method: HTTP method (default: POST)
            headers: Optional HTTP headers
            timeout: Optional request timeout in seconds
            auth: Optional authentication ({"username": "user", "password": "pass"})

        Returns:
            Dict[str, Any]: Send result
        """
        if not webhook_url:
            logger.error("No webhook URL provided")
            return {
                "status": "error",
                "error": "No webhook URL provided",
            }

        if not headers:
            headers = {}

        # Set default content type if not specified
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

        # Add custom headers from settings
        if hasattr(self.settings, "webhook_headers") and self.settings.webhook_headers:
            for key, value in self.settings.webhook_headers.items():
                headers[key] = value

        auth_tuple = None
        try:
            # Setup authentication if provided
            if auth and "username" in auth and "password" in auth:
                auth_tuple = aiohttp.BasicAuth(auth["username"], auth["password"])

            # Convert timeout to ClientTimeout object if provided
            client_timeout = (
                ClientTimeout(total=timeout) if timeout is not None else None
            )

            # Send webhook request
            async with aiohttp.ClientSession() as session:
                method_func = getattr(session, method.lower(), session.post)

                async with method_func(
                    webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=client_timeout,
                    auth=auth_tuple,
                ) as response:
                    response_text = await response.text()

                    try:
                        response_data = json.loads(response_text)
                    except json.JSONDecodeError:
                        response_data = response_text

                    if 200 <= response.status < 300:
                        logger.info(f"Webhook sent successfully to {webhook_url}")
                        return {
                            "status": "delivered",
                            "webhook_url": webhook_url,
                            "status_code": response.status,
                            "response": response_data,
                        }
                    else:
                        logger.error(
                            f"Webhook error: {response.status} response from {webhook_url}"
                        )
                        return {
                            "status": "error",
                            "webhook_url": webhook_url,
                            "error": f"HTTP {response.status}",
                            "status_code": response.status,
                            "response": response_data,
                        }

        except aiohttp.ClientError as e:
            logger.error(f"Network error sending webhook to {webhook_url}: {e}")
            return {
                "status": "error",
                "webhook_url": webhook_url,
                "error": f"Network error: {str(e)}",
            }

        except Exception as e:
            logger.error(f"Error sending webhook to {webhook_url}: {e}")
            return {
                "status": "error",
                "webhook_url": webhook_url,
                "error": str(e),
            }
