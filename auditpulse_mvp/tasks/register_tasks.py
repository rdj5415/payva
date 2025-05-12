"""Task registration module for registering background tasks."""

import logging
from auditpulse_mvp.tasks.task_manager import TaskManager
from auditpulse_mvp.tasks.plaid_tasks import sync_plaid_transactions
from auditpulse_mvp.tasks.notification_tasks import process_notification, process_batch_notifications

logger = logging.getLogger(__name__)

async def register_tasks(task_manager: TaskManager):
    """Register all background tasks with the task manager.
    
    Args:
        task_manager: Task manager instance
    """
    logger.info("Registering background tasks")
    
    # Register Plaid tasks
    task_manager.register_task(
        "sync_plaid_transactions",
        sync_plaid_transactions,
        max_retries=3,
        retry_delay=300,  # 5 minutes
    )
    
    # Register notification tasks
    task_manager.register_task(
        "process_notification",
        process_notification,
        max_retries=3,
        retry_delay=60,  # 1 minute
    )
    
    task_manager.register_task(
        "process_batch_notifications",
        process_batch_notifications,
        max_retries=2,
        retry_delay=120,  # 2 minutes
    )
    
    logger.info("Task registration completed") 