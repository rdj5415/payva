"""Slack notification provider.

This module provides a notification provider for sending alerts via Slack.
"""
import logging
from typing import Dict, List, Optional, Any

import httpx
import json

from auditpulse_mvp.alerts.base import NotificationProvider, NotificationPayload, NotificationStatus
from auditpulse_mvp.utils.settings import settings

logger = logging.getLogger(__name__)


class SlackNotificationProvider(NotificationProvider):
    """Slack notification provider."""
    
    def __init__(self, bot_token: Optional[str] = None):
        """Initialize the Slack notification provider.
        
        Args:
            bot_token: Slack bot token. If not provided, uses settings.
        """
        super().__init__()
        self.bot_token = bot_token or (
            settings.SLACK_BOT_TOKEN.get_secret_value() 
            if settings.SLACK_BOT_TOKEN else None
        )
        self.base_url = "https://slack.com/api"
    
    async def is_configured(self) -> bool:
        """Check if the provider is properly configured.
        
        Returns:
            bool: True if the provider is configured, False otherwise
        """
        return self.bot_token is not None
    
    async def send(self, recipient: str, payload: NotificationPayload) -> NotificationStatus:
        """Send a Slack notification.
        
        Args:
            recipient: Slack channel ID or user ID
            payload: The notification payload
            
        Returns:
            NotificationStatus: Status of the sent notification
        """
        if not self.is_configured():
            logger.warning("Slack is not configured. Cannot send Slack notification.")
            return NotificationStatus.FAILED
        
        try:
            # Format the Slack message blocks
            blocks = self._format_slack_blocks(payload)
            
            # Prepare Slack API request
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json; charset=utf-8",
            }
            
            data = {
                "channel": recipient,
                "text": payload.subject,  # Fallback text
                "blocks": blocks,
            }
            
            # Send the request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat.postMessage",
                    headers=headers,
                    json=data,
                    timeout=10.0,
                )
                
            response_data = response.json()
            
            if response.status_code == 200 and response_data.get("ok", False):
                logger.info(f"Slack notification sent to {recipient}")
                return NotificationStatus.SENT
            else:
                error = response_data.get("error", "Unknown error")
                logger.error(f"Failed to send Slack notification: {error}")
                return NotificationStatus.FAILED
                
        except Exception as e:
            logger.exception(f"Error sending Slack notification: {str(e)}")
            return NotificationStatus.FAILED
    
    def _format_slack_blocks(self, payload: NotificationPayload) -> List[Dict[str, Any]]:
        """Format the notification as Slack blocks.
        
        Args:
            payload: The notification payload
            
        Returns:
            List[Dict[str, Any]]: Slack blocks for the message
        """
        risk_emoji = self._get_risk_emoji(payload.risk_level)
        
        # Header section
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{risk_emoji} Anomaly Alert: {payload.subject}",
                    "emoji": True
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Risk Level:*\n{payload.risk_level.upper()} ({payload.risk_score:.0f}%)"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Transaction ID:*\n{payload.transaction_id}"
                    }
                ]
            }
        ]
        
        # Add explanation if available
        if payload.explanation:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Explanation:*\n{payload.explanation}"
                }
            })
        
        # Add context information
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Detected at: {payload.time.strftime('%Y-%m-%d %H:%M:%S')}"
                }
            ]
        })
        
        # Add action buttons
        actions = [
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "View Details",
                    "emoji": True
                },
                "url": payload.dashboard_url,
                "style": "primary"
            }
        ]
        
        # Add additional action links if available
        for text, url in payload.action_links.items():
            actions.append({
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": text,
                    "emoji": True
                },
                "url": url
            })
        
        blocks.append({
            "type": "actions",
            "elements": actions
        })
        
        return blocks 