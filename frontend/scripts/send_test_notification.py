#!/usr/bin/env python
"""Script to send a test notification."""

import asyncio
import argparse
import logging
import sys
import os
import uuid
from datetime import datetime

# Add parent directory to path to be able to import from auditpulse_mvp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from auditpulse_mvp.notifications.notification_manager import (
    NotificationManager,
    NotificationRequest,
    NotificationRecipient,
    NotificationPriority,
)
from auditpulse_mvp.tasks.task_manager import TaskManager
from auditpulse_mvp.database.session import init_db
from auditpulse_mvp.utils.settings import get_settings
from auditpulse_mvp.notifications.templates import TemplateManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


async def send_test_notification(
    template_id: str,
    email: str,
    priority: str = "medium",
    template_data: dict = None,
    channels: list = None,
):
    """Send a test notification.

    Args:
        template_id: Template ID to use
        email: Recipient email
        priority: Notification priority
        template_data: Data for template rendering
        channels: Channels to send notification through
    """
    logger.info(f"Sending test notification using template: {template_id}")

    # Initialize database
    await init_db()

    # Initialize task manager
    task_manager = TaskManager.get_instance()
    await task_manager.start()

    # Initialize template manager
    template_manager = TemplateManager(settings)

    # Check if template exists
    template = await template_manager.get_template(template_id)
    if not template:
        logger.error(f"Template not found: {template_id}")
        logger.info("Available templates:")
        templates = await template_manager.list_templates()
        for t in templates:
            logger.info(f"  - {t['template_id']}: {t['name']}")
        return

    # Create notification manager
    notification_manager = NotificationManager(settings, task_manager)

    # Create notification recipient
    recipient = NotificationRecipient(
        email=email,
        channels=channels or ["email"],
    )

    # Create notification request
    request = NotificationRequest(
        template_id=template_id,
        recipient=recipient,
        template_data=template_data or {},
        priority=NotificationPriority(priority),
        channels=channels,
    )

    # Generate a random user ID
    user_id = uuid.uuid4()

    try:
        # Send notification
        result = await notification_manager.send_notification(
            request=request,
            user_id=user_id,
        )

        logger.info(f"Notification sent: {result}")
        logger.info(f"Notification ID: {result.get('notification_id')}")
        logger.info(f"Status: {result.get('status')}")

        # Wait a bit for the notification to be processed
        logger.info("Waiting for notification to be processed...")
        await asyncio.sleep(5)

        # Get notification status
        status = await notification_manager.get_notification_status(
            result.get("notification_id")
        )
        logger.info(f"Notification status: {status.get('status')}")

        if "delivery_attempts" in status:
            for attempt in status["delivery_attempts"]:
                logger.info(
                    f"Delivery attempt via {attempt.get('channel')}: {attempt.get('status')}"
                )
                if attempt.get("error"):
                    logger.error(f"Error: {attempt.get('error')}")
    finally:
        # Shutdown task manager
        await task_manager.stop()


def main():
    """Main function for sending test notification."""
    parser = argparse.ArgumentParser(description="Send a test notification")
    parser.add_argument(
        "--template",
        "-t",
        default="anomaly_detected",
        help="Template ID to use",
    )
    parser.add_argument(
        "--email",
        "-e",
        required=True,
        help="Recipient email",
    )
    parser.add_argument(
        "--priority",
        "-p",
        default="medium",
        choices=["low", "medium", "high", "critical"],
        help="Notification priority",
    )
    parser.add_argument(
        "--channel",
        "-c",
        action="append",
        choices=["email", "slack", "sms", "webhook"],
        help="Channel to send notification through (can be used multiple times)",
    )
    args = parser.parse_args()

    # Prepare template data based on template ID
    template_data = {}
    if args.template == "anomaly_detected":
        template_data = {
            "user_name": "Test User",
            "transaction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "transaction_amount": "$1,234.56",
            "transaction_description": "Test Transaction",
            "account_name": "Test Account",
            "anomaly_score": "0.92",
            "anomaly_reason": "Amount significantly higher than usual",
            "dashboard_url": "https://example.com/dashboard",
        }
    elif args.template == "model_performance_report":
        template_data = {
            "user_name": "Test User",
            "model_name": "AnomalyDetector v1.0",
            "accuracy": "0.95",
            "precision": "0.92",
            "recall": "0.89",
            "f1_score": "0.91",
            "roc_auc": "0.94",
            "start_date": (datetime.now() - datetime.timedelta(days=7)).strftime(
                "%Y-%m-%d"
            ),
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "total_transactions": "1,234",
            "anomalies_detected": "32",
            "false_positive_rate": "0.08",
            "performance_change": "2.5",
            "report_url": "https://example.com/reports/123",
        }
    elif args.template == "system_alert":
        template_data = {
            "alert_type": "Database Performance",
            "alert_title": "Database Connection Pool Exhausted",
            "severity": "high",
            "alert_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "component": "Database",
            "alert_description": "The database connection pool has been exhausted. This may indicate a connection leak or unusually high load.",
            "actions": [
                "Check for long-running queries",
                "Increase connection pool size",
                "Restart database service if necessary",
            ],
            "alert_id": str(uuid.uuid4()),
        }
    elif args.template == "account_security":
        template_data = {
            "user_name": "Test User",
            "alert_type": "login_attempt",
            "alert_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "location": "San Francisco, CA, USA",
            "ip_address": "192.168.1.1",
            "device": "Windows 10, Chrome 90.0.4430.212",
            "attempt_count": "3",
        }
    elif args.template == "weekly_report":
        template_data = {
            "user_name": "Test User",
            "week_start": (datetime.now() - datetime.timedelta(days=7)).strftime(
                "%Y-%m-%d"
            ),
            "week_end": datetime.now().strftime("%Y-%m-%d"),
            "accounts": [
                {"name": "Checking Account", "balance": "$2,345.67"},
                {"name": "Savings Account", "balance": "$12,345.67"},
                {"name": "Credit Card", "balance": "-$345.67"},
            ],
            "total_deposits": "$3,456.78",
            "total_withdrawals": "$2,345.67",
            "net_change": "$1,111.11",
            "anomalies": [
                {
                    "date": (datetime.now() - datetime.timedelta(days=3)).strftime(
                        "%Y-%m-%d"
                    ),
                    "description": "Unusual payment to XYZ Corp",
                    "amount": "$876.54",
                },
                {
                    "date": (datetime.now() - datetime.timedelta(days=1)).strftime(
                        "%Y-%m-%d"
                    ),
                    "description": "Large transfer to Savings",
                    "amount": "$1,500.00",
                },
            ],
            "top_categories": [
                {"name": "Groceries", "amount": "$345.67", "percentage": "25"},
                {"name": "Entertainment", "amount": "$234.56", "percentage": "18"},
                {"name": "Dining Out", "amount": "$198.76", "percentage": "15"},
                {"name": "Utilities", "amount": "$156.78", "percentage": "12"},
            ],
            "report_url": "https://example.com/reports/weekly/123",
        }

    # Run the notification
    asyncio.run(
        send_test_notification(
            template_id=args.template,
            email=args.email,
            priority=args.priority,
            template_data=template_data,
            channels=args.channel,
        )
    )

    logger.info("Done")


if __name__ == "__main__":
    main()
