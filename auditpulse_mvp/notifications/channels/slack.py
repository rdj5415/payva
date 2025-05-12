"""Slack notification channel for sending Slack alerts."""

import logging
import aiohttp
from typing import Dict, Any, Optional, List

from auditpulse_mvp.utils.settings import Settings

logger = logging.getLogger(__name__)

class SlackNotifier:
    """Slack notifier for sending Slack notifications."""
    
    def __init__(self, settings: Settings):
        """Initialize the Slack notifier.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.bot_token = settings.SLACK_BOT_TOKEN
        self.webhook_url = settings.SLACK_WEBHOOK_URL
        
    async def send(
        self,
        channel: str,
        text: str,
        blocks: Optional[List[Dict[str, Any]]] = None,
        thread_ts: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a Slack notification using the Slack API.
        
        Args:
            channel: Slack channel ID or name
            text: Message text
            blocks: Optional message blocks for rich formatting
            thread_ts: Optional thread timestamp to reply to
            
        Returns:
            Dict[str, Any]: Send result
        """
        try:
            if not self.bot_token:
                logger.error("Slack bot token is not configured")
                return {
                    "status": "error",
                    "channel": channel,
                    "error": "Slack bot token is not configured",
                }
                
            # Prepare message payload
            payload = {
                "channel": channel,
                "text": text,
            }
            
            if blocks:
                payload["blocks"] = blocks
                
            if thread_ts:
                payload["thread_ts"] = thread_ts
                
            # Send message via Slack API
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.bot_token}",
                    "Content-Type": "application/json; charset=utf-8",
                }
                
                async with session.post(
                    "https://slack.com/api/chat.postMessage",
                    headers=headers,
                    json=payload,
                ) as response:
                    result = await response.json()
                    
                    if not result.get("ok", False):
                        logger.error(f"Error sending Slack message: {result.get('error')}")
                        return {
                            "status": "error",
                            "channel": channel,
                            "error": result.get("error"),
                        }
                        
                    logger.info(f"Slack message sent successfully to {channel}")
                    return {
                        "status": "delivered",
                        "channel": channel,
                        "timestamp": result.get("ts"),
                        "message_id": result.get("ts"),
                    }
                    
        except Exception as e:
            logger.error(f"Error sending Slack message to {channel}: {e}")
            return {
                "status": "error",
                "channel": channel,
                "error": str(e),
            }
            
    async def send_webhook(
        self,
        text: str,
        blocks: Optional[List[Dict[str, Any]]] = None,
        channel: Optional[str] = None,
        username: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a Slack notification using a webhook.
        
        Args:
            text: Message text
            blocks: Optional message blocks for rich formatting
            channel: Optional channel override
            username: Optional username override
            
        Returns:
            Dict[str, Any]: Send result
        """
        try:
            if not self.webhook_url:
                logger.error("Slack webhook URL is not configured")
                return {
                    "status": "error",
                    "error": "Slack webhook URL is not configured",
                }
                
            # Prepare message payload
            payload = {"text": text}
            
            if blocks:
                payload["blocks"] = blocks
                
            if channel:
                payload["channel"] = channel
                
            if username:
                payload["username"] = username
                
            # Send message via webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                ) as response:
                    # Webhook returns 200 OK with body "ok" if successful
                    if response.status == 200 and await response.text() == "ok":
                        logger.info(f"Slack webhook message sent successfully")
                        return {
                            "status": "delivered",
                            "channel": channel,
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"Error sending Slack webhook message: {error_text}")
                        return {
                            "status": "error",
                            "channel": channel,
                            "error": error_text,
                        }
                        
        except Exception as e:
            logger.error(f"Error sending Slack webhook message: {e}")
            return {
                "status": "error",
                "error": str(e),
            } 