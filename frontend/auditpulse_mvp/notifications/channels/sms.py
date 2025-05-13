"""SMS notification channel for sending text message alerts."""

import logging
from typing import Dict, Any, Optional

from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from auditpulse_mvp.utils.settings import Settings

logger = logging.getLogger(__name__)

class SMSNotifier:
    """SMS notifier for sending text message notifications via Twilio."""
    
    def __init__(self, settings: Settings):
        """Initialize the SMS notifier.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.account_sid = settings.TWILIO_ACCOUNT_SID
        self.auth_token = settings.TWILIO_AUTH_TOKEN
        self.from_number = settings.TWILIO_FROM_NUMBER
        self.client = None
        
        if self.account_sid and self.auth_token:
            self.client = Client(self.account_sid, self.auth_token)
        
    async def send(
        self,
        to_number: str,
        message: str,
        media_urls: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """Send an SMS message using Twilio.
        
        Args:
            to_number: Recipient phone number in E.164 format (+1XXXXXXXXXX)
            message: SMS message text
            media_urls: Optional list of media URLs to include (MMS)
            
        Returns:
            Dict[str, Any]: Send result
        """
        try:
            if not self.client:
                logger.error("Twilio client is not configured")
                return {
                    "status": "error",
                    "to_number": to_number,
                    "error": "Twilio client is not configured",
                }
                
            if not self.from_number:
                logger.error("Twilio from number is not configured")
                return {
                    "status": "error",
                    "to_number": to_number,
                    "error": "Twilio from number is not configured",
                }
                
            # Prepare message parameters
            params = {
                "body": message,
                "from_": self.from_number,
                "to": to_number,
            }
            
            if media_urls:
                params["media_url"] = media_urls
                
            # Send the SMS
            message = self.client.messages.create(**params)
            
            logger.info(f"SMS sent successfully to {to_number}, SID: {message.sid}")
            return {
                "status": "delivered",  # Twilio doesn't confirm delivery immediately
                "to_number": to_number,
                "message_id": message.sid,
            }
            
        except TwilioRestException as e:
            logger.error(f"Twilio API error sending SMS to {to_number}: {e}")
            return {
                "status": "error",
                "to_number": to_number,
                "error": f"Twilio API error: {e.msg}",
                "error_code": e.code,
            }
            
        except Exception as e:
            logger.error(f"Error sending SMS to {to_number}: {e}")
            return {
                "status": "error",
                "to_number": to_number,
                "error": str(e),
            } 