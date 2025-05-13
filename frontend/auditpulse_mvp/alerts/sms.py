"""SMS notification provider using Twilio.

This module provides a notification provider for sending SMS alerts via Twilio.
"""
import logging
import re
from typing import Dict, List, Optional, Any

import httpx
import base64

from auditpulse_mvp.alerts.base import NotificationProvider, NotificationPayload, NotificationStatus
from auditpulse_mvp.utils.settings import settings

logger = logging.getLogger(__name__)


class SMSNotificationProvider(NotificationProvider):
    """SMS notification provider using Twilio."""
    
    def __init__(
        self, 
        account_sid: Optional[str] = None, 
        auth_token: Optional[str] = None,
        from_number: Optional[str] = None
    ):
        """Initialize the SMS notification provider.
        
        Args:
            account_sid: Twilio account SID. If not provided, uses settings.
            auth_token: Twilio auth token. If not provided, uses settings.
            from_number: Sender phone number. If not provided, uses a default.
        """
        super().__init__()
        self.account_sid = account_sid or settings.TWILIO_ACCOUNT_SID
        self.auth_token = auth_token or (
            settings.TWILIO_AUTH_TOKEN.get_secret_value() 
            if settings.TWILIO_AUTH_TOKEN else None
        )
        self.from_number = from_number or "+15555555555"  # Replace with actual default
        self.base_url = "https://api.twilio.com/2010-04-01"
    
    async def is_configured(self) -> bool:
        """Check if the provider is properly configured.
        
        Returns:
            bool: True if the provider is configured, False otherwise
        """
        return self.account_sid is not None and self.auth_token is not None
    
    async def send(self, recipient: str, payload: NotificationPayload) -> NotificationStatus:
        """Send an SMS notification.
        
        Args:
            recipient: Phone number of the recipient
            payload: The notification payload
            
        Returns:
            NotificationStatus: Status of the sent notification
        """
        if not self.is_configured():
            logger.warning("Twilio is not configured. Cannot send SMS notification.")
            return NotificationStatus.FAILED
        
        try:
            # Verify phone number format
            if not self._is_valid_phone_number(recipient):
                logger.error(f"Invalid phone number format: {recipient}")
                return NotificationStatus.FAILED
                
            # Format the SMS message (with character limits in mind)
            message = self._format_sms_message(payload)
            
            # Basic auth token for Twilio
            auth = base64.b64encode(
                f"{self.account_sid}:{self.auth_token}".encode()
            ).decode()
            
            # Prepare Twilio API request
            headers = {
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/x-www-form-urlencoded",
            }
            
            data = {
                "To": recipient,
                "From": self.from_number,
                "Body": message,
            }
            
            # Add a media URL for rich content if available
            if payload.action_links and "view" in payload.action_links:
                data["MediaUrl"] = payload.action_links["view"]
            
            # Send the request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/Accounts/{self.account_sid}/Messages.json",
                    headers=headers,
                    data=data,
                    timeout=10.0,
                )
                
            response_data = response.json()
            
            if response.status_code >= 200 and response.status_code < 300:
                logger.info(f"SMS notification sent to {recipient}")
                return NotificationStatus.SENT
            else:
                error = response_data.get("message", "Unknown error")
                logger.error(f"Failed to send SMS notification: {error}")
                return NotificationStatus.FAILED
                
        except Exception as e:
            logger.exception(f"Error sending SMS notification: {str(e)}")
            return NotificationStatus.FAILED
    
    def _format_sms_message(self, payload: NotificationPayload) -> str:
        """Format the notification as an SMS message.
        
        Args:
            payload: The notification payload
            
        Returns:
            str: Formatted SMS message
        """
        # Keep SMS messages concise due to character limits
        risk_emoji = self._get_risk_emoji(payload.risk_level)
        
        # Base message with critical information
        message = (
            f"🔔 AuditPulse Alert: {payload.subject}\n\n"
            f"Risk: {risk_emoji} {payload.risk_level.upper()} ({payload.risk_score:.0f}%)\n"
            f"Transaction: {str(payload.transaction_id)[:8]}...\n"
        )
        
        # Add a very brief explanation if available
        if payload.explanation and len(payload.explanation) > 0:
            # Truncate explanation to keep SMS brief
            max_explanation_length = 100
            explanation = payload.explanation
            if len(explanation) > max_explanation_length:
                explanation = explanation[:max_explanation_length - 3] + "..."
            message += f"\n{explanation}\n"
        
        # Add view link
        message += f"\nDetails: {payload.dashboard_url}"
        
        return message
    
    def _is_valid_phone_number(self, phone_number: str) -> bool:
        """Validate phone number format.
        
        Args:
            phone_number: Phone number to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Basic validation for E.164 format
        pattern = r"^\+[1-9]\d{1,14}$"
        return bool(re.match(pattern, phone_number)) 