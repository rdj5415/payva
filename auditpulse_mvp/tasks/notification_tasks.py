"""Notification tasks for processing notifications in the background."""

import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import asyncio

from auditpulse_mvp.database.session import get_db
from auditpulse_mvp.database.models.notification import (
    Notification,
    NotificationDeliveryAttempt,
)
from auditpulse_mvp.database.models.user import User
from auditpulse_mvp.notifications.channels import (
    NotificationChannel,
    NotificationChannelFactory,
)
from auditpulse_mvp.notifications.templates import TemplateManager
from auditpulse_mvp.utils.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def process_notification(
    notification_id: str,
    max_retries: int = 3,
    retry_delay: int = 60,
) -> Dict[str, Any]:
    """Process a notification and send it through the appropriate channels.

    Args:
        notification_id: ID of the notification to process
        max_retries: Maximum number of retry attempts per channel
        retry_delay: Delay between retries in seconds

    Returns:
        Dict[str, Any]: Result of the notification processing
    """
    async with get_db() as db:
        # Get notification
        notification = (
            await db.query(Notification)
            .filter(Notification.id == notification_id)
            .first()
        )

        if not notification:
            logger.error(f"Notification not found: {notification_id}")
            return {
                "status": "error",
                "error": f"Notification not found: {notification_id}",
            }

        # Update notification status to processing
        notification.status = "processing"
        notification.processed_at = datetime.utcnow()
        await db.commit()

        try:
            # Get the template manager
            template_manager = TemplateManager(settings)

            # Render the template
            rendered_template = await template_manager.render_template(
                notification.template_id,
                notification.template_data or {},
            )

            # Get the channel factory
            channel_factory = NotificationChannelFactory(settings)

            # Get the user if associated with the notification
            user = None
            if notification.user_id:
                user = (
                    await db.query(User).filter(User.id == notification.user_id).first()
                )

            # Determine channels to send to
            channels_to_use = []
            if "channels" in notification.recipient:
                channels_to_use = notification.recipient["channels"]
            elif user:
                # Get channels from user preferences based on notification type
                notification_type = notification.template_data.get(
                    "notification_type", "general"
                )
                channels_to_use = user.get_channels_for_notification_type(
                    notification_type
                )
            else:
                # Default to email if no channels specified
                channels_to_use = ["email"]

            # Process each channel
            results = {}
            success_count = 0
            for channel in channels_to_use:
                try:
                    # Create delivery attempt record
                    delivery_attempt = NotificationDeliveryAttempt(
                        notification_id=notification.id,
                        channel=channel,
                        status="pending",
                        attempt_number=1,
                    )
                    db.add(delivery_attempt)
                    await db.commit()
                    await db.refresh(delivery_attempt)

                    # Send through the appropriate channel
                    result = await send_via_channel(
                        channel,
                        notification.recipient,
                        rendered_template,
                        notification.template_data,
                        channel_factory,
                        max_retries,
                        retry_delay,
                    )

                    # Update delivery attempt with result
                    delivery_attempt.status = result.get("status", "failed")
                    delivery_attempt.response = result
                    if result.get("error"):
                        delivery_attempt.error = result["error"]

                    await db.commit()

                    results[channel] = result
                    if result.get("status") == "delivered":
                        success_count += 1

                except Exception as e:
                    logger.error(
                        f"Error processing channel {channel} for notification {notification_id}: {e}"
                    )
                    results[channel] = {
                        "status": "error",
                        "error": str(e),
                    }

                    # Update delivery attempt with error
                    if "delivery_attempt" in locals():
                        delivery_attempt.status = "failed"
                        delivery_attempt.error = str(e)
                        await db.commit()

            # Update notification status based on results
            if success_count == len(channels_to_use):
                notification.status = "delivered"
            elif success_count > 0:
                notification.status = "partially_delivered"
            else:
                notification.status = "failed"

            await db.commit()

            return {
                "status": notification.status,
                "notification_id": str(notification.id),
                "results": results,
            }

        except Exception as e:
            logger.error(f"Error processing notification {notification_id}: {e}")

            # Update notification status to failed
            notification.status = "failed"
            await db.commit()

            return {
                "status": "error",
                "notification_id": str(notification.id),
                "error": str(e),
            }


async def send_via_channel(
    channel: str,
    recipient: Dict[str, Any],
    rendered_template: Dict[str, str],
    template_data: Dict[str, Any],
    channel_factory: NotificationChannelFactory,
    max_retries: int = 3,
    retry_delay: int = 60,
) -> Dict[str, Any]:
    """Send a notification through a specific channel.

    Args:
        channel: Channel to use
        recipient: Recipient information
        rendered_template: Rendered template
        template_data: Template data
        channel_factory: Channel factory
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds

    Returns:
        Dict[str, Any]: Result of the send operation
    """
    retries = 0
    last_error = None

    while retries < max_retries:
        try:
            if channel == NotificationChannel.EMAIL:
                return await send_email(recipient, rendered_template, channel_factory)
            elif channel == NotificationChannel.SLACK:
                return await send_slack(
                    recipient, rendered_template, template_data, channel_factory
                )
            elif channel == NotificationChannel.SMS:
                return await send_sms(recipient, rendered_template, channel_factory)
            elif channel == NotificationChannel.WEBHOOK:
                return await send_webhook(
                    recipient, rendered_template, template_data, channel_factory
                )
            else:
                return {
                    "status": "error",
                    "error": f"Unsupported channel: {channel}",
                }
        except Exception as e:
            last_error = str(e)
            retries += 1
            if retries < max_retries:
                await asyncio.sleep(retry_delay)

    return {
        "status": "failed",
        "error": last_error or "Max retries exceeded",
        "retries": retries,
    }


async def send_email(
    recipient: Dict[str, Any],
    rendered_template: Dict[str, str],
    channel_factory: NotificationChannelFactory,
) -> Dict[str, Any]:
    """Send an email notification.

    Args:
        recipient: Recipient information
        rendered_template: Rendered template
        channel_factory: Channel factory

    Returns:
        Dict[str, Any]: Result of the send operation
    """
    email_notifier = channel_factory.get_channel(NotificationChannel.EMAIL)

    # Get recipient email address
    to_email = recipient.get("email")
    if not to_email:
        return {
            "status": "error",
            "error": "Recipient email address not provided",
        }

    # Send email
    return await email_notifier.send(
        to_email=to_email,
        subject=rendered_template["subject"],
        body=rendered_template["body"],
        html_body=rendered_template.get("html_body"),
    )


async def send_slack(
    recipient: Dict[str, Any],
    rendered_template: Dict[str, str],
    template_data: Dict[str, Any],
    channel_factory: NotificationChannelFactory,
) -> Dict[str, Any]:
    """Send a Slack notification.

    Args:
        recipient: Recipient information
        rendered_template: Rendered template
        template_data: Template data
        channel_factory: Channel factory

    Returns:
        Dict[str, Any]: Result of the send operation
    """
    slack_notifier = channel_factory.get_channel(NotificationChannel.SLACK)

    # Get Slack channel or webhook
    slack_channel = recipient.get("slack_channel")
    slack_webhook_url = recipient.get("slack_webhook_url")

    if not slack_channel and not slack_webhook_url:
        return {
            "status": "error",
            "error": "Slack channel or webhook URL not provided",
        }

    # Prepare message blocks
    blocks = template_data.get("slack_blocks", [])
    if not blocks:
        # Create basic blocks if none provided
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": rendered_template["body"],
                },
            },
        ]

    # Send via webhook if URL provided
    if slack_webhook_url:
        return await slack_notifier.send_webhook(
            text=rendered_template["subject"],
            blocks=blocks,
            channel=slack_channel,  # Optional override
        )
    # Otherwise send via API
    else:
        return await slack_notifier.send(
            channel=slack_channel,
            text=rendered_template["subject"],
            blocks=blocks,
        )


async def send_sms(
    recipient: Dict[str, Any],
    rendered_template: Dict[str, str],
    channel_factory: NotificationChannelFactory,
) -> Dict[str, Any]:
    """Send an SMS notification.

    Args:
        recipient: Recipient information
        rendered_template: Rendered template
        channel_factory: Channel factory

    Returns:
        Dict[str, Any]: Result of the send operation
    """
    sms_notifier = channel_factory.get_channel(NotificationChannel.SMS)

    # Get recipient phone number
    phone_number = recipient.get("phone_number")
    if not phone_number:
        return {
            "status": "error",
            "error": "Recipient phone number not provided",
        }

    # Send SMS
    return await sms_notifier.send(
        to_number=phone_number,
        message=rendered_template["body"],
    )


async def send_webhook(
    recipient: Dict[str, Any],
    rendered_template: Dict[str, str],
    template_data: Dict[str, Any],
    channel_factory: NotificationChannelFactory,
) -> Dict[str, Any]:
    """Send a webhook notification.

    Args:
        recipient: Recipient information
        rendered_template: Rendered template
        template_data: Template data
        channel_factory: Channel factory

    Returns:
        Dict[str, Any]: Result of the send operation
    """
    webhook_notifier = channel_factory.get_channel(NotificationChannel.WEBHOOK)

    # Get webhook URL
    webhook_url = recipient.get("webhook_url")
    if not webhook_url:
        return {
            "status": "error",
            "error": "Webhook URL not provided",
        }

    # Prepare payload
    payload = template_data.get("webhook_payload", {})
    if not payload:
        # Create basic payload if none provided
        payload = {
            "subject": rendered_template["subject"],
            "message": rendered_template["body"],
            "timestamp": datetime.utcnow().isoformat(),
        }

    # Get additional webhook options
    method = template_data.get("webhook_method", "POST")
    headers = template_data.get("webhook_headers")
    timeout = template_data.get("webhook_timeout")
    auth = template_data.get("webhook_auth")

    # Send webhook
    return await webhook_notifier.send(
        webhook_url=webhook_url,
        payload=payload,
        method=method,
        headers=headers,
        timeout=timeout,
        auth=auth,
    )


async def process_batch_notifications(
    notifications: List[Dict[str, Any]],
    max_concurrent: int = 10,
) -> Dict[str, Any]:
    """Process a batch of notifications concurrently.

    Args:
        notifications: List of notification data
        max_concurrent: Maximum number of concurrent tasks

    Returns:
        Dict[str, Any]: Result of the batch processing
    """
    async with get_db() as db:
        created_notifications = []

        # Create notifications in the database
        for notification_data in notifications:
            notification = Notification(
                template_id=notification_data["template_id"],
                user_id=notification_data.get("user_id"),
                recipient=notification_data["recipient"],
                template_data=notification_data.get("template_data", {}),
                priority=notification_data.get("priority", "medium"),
                scheduled_at=notification_data.get("scheduled_at"),
            )
            db.add(notification)
            await db.commit()
            await db.refresh(notification)
            created_notifications.append(notification)

        # Process notifications concurrently
        tasks = []
        for notification in created_notifications:
            task = process_notification(
                notification_id=str(notification.id),
                max_retries=notification_data.get("max_retries", 3),
                retry_delay=notification_data.get("retry_delay", 60),
            )
            tasks.append(task)

        # Process in batches to avoid overloading
        results = []
        for i in range(0, len(tasks), max_concurrent):
            batch = tasks[i : i + max_concurrent]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            results.extend(batch_results)

        # Process results
        success_count = 0
        error_count = 0
        for result in results:
            if isinstance(result, Exception):
                error_count += 1
            elif result.get("status") in ["delivered", "partially_delivered"]:
                success_count += 1
            else:
                error_count += 1

        return {
            "status": "completed",
            "total": len(notifications),
            "success_count": success_count,
            "error_count": error_count,
            "results": results,
        }
