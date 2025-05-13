"""Scheduler for notification tasks.

This module provides a scheduler for running notification tasks periodically.
"""
import asyncio
import logging
from datetime import datetime, timedelta
import time

from auditpulse_mvp.alerts.notification_service import get_notification_service
from auditpulse_mvp.database.session import get_db
from auditpulse_mvp.utils.settings import settings

logger = logging.getLogger(__name__)


class NotificationScheduler:
    """Scheduler for running notification tasks."""
    
    def __init__(self):
        """Initialize the notification scheduler."""
        self._is_running = False
        self._tasks = []
        self._notification_interval = getattr(
            settings, "NOTIFICATION_CHECK_INTERVAL_MINUTES", 15
        )
        logger.info(f"Notification scheduler initialized with interval: {self._notification_interval} minutes")
    
    async def start(self):
        """Start the notification scheduler."""
        if self._is_running:
            logger.warning("Notification scheduler is already running")
            return
            
        self._is_running = True
        logger.info("Starting notification scheduler")
        
        # Start the periodic notification task
        notification_task = asyncio.create_task(self._schedule_notifications())
        self._tasks.append(notification_task)
    
    async def stop(self):
        """Stop the notification scheduler."""
        if not self._is_running:
            logger.warning("Notification scheduler is not running")
            return
            
        self._is_running = False
        logger.info("Stopping notification scheduler")
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            
        # Wait for tasks to complete
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            
        self._tasks = []
        
    async def _schedule_notifications(self):
        """Schedule periodic notification checks."""
        try:
            while self._is_running:
                # Log the next scheduled run time
                next_run = datetime.now() + timedelta(minutes=self._notification_interval)
                logger.info(
                    f"Next notification check scheduled at: {next_run.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                
                # Wait for the specified interval
                await asyncio.sleep(self._notification_interval * 60)
                
                # Check if we're still running
                if not self._is_running:
                    break
                    
                # Run the notification check
                await self._run_notification_check()
                
        except asyncio.CancelledError:
            logger.info("Notification scheduler task cancelled")
            raise
        except Exception as e:
            logger.exception(f"Error in notification scheduler: {e}")
            if self._is_running:
                # Restart the scheduler after a delay
                await asyncio.sleep(60)
                asyncio.create_task(self._schedule_notifications())
    
    async def _run_notification_check(self):
        """Check for and send notifications for high-risk anomalies."""
        try:
            logger.info("Running notification check for high-risk anomalies")
            start_time = time.time()
            
            # Get the notification service
            notification_service = get_notification_service()
            
            # Process notifications
            async with get_db() as db:
                notifications_sent = await notification_service.notify_high_risk_anomalies(db)
            
            elapsed_time = time.time() - start_time
            logger.info(
                f"Notification check completed in {elapsed_time:.2f}s. "
                f"Sent {notifications_sent} notifications."
            )
            
        except Exception as e:
            logger.exception(f"Error during notification check: {e}")


# Singleton instance
_notification_scheduler = None


def get_notification_scheduler() -> NotificationScheduler:
    """Get the global notification scheduler instance.
    
    Returns:
        NotificationScheduler: The notification scheduler
    """
    global _notification_scheduler
    if _notification_scheduler is None:
        _notification_scheduler = NotificationScheduler()
    return _notification_scheduler


async def start_notification_scheduler():
    """Start the notification scheduler."""
    scheduler = get_notification_scheduler()
    await scheduler.start()


async def stop_notification_scheduler():
    """Stop the notification scheduler."""
    scheduler = get_notification_scheduler()
    await scheduler.stop() 