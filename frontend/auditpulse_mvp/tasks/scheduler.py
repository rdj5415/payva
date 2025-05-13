"""Task scheduler implementation using APScheduler.

This module provides a unified interface for scheduling and managing
background tasks across the application.
"""

import asyncio
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from apscheduler.executors.pool import ProcessPoolExecutor, ThreadPoolExecutor
from apscheduler.jobstores.redis import RedisJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Depends

from auditpulse_mvp.utils.settings import get_settings, Settings

# Configure logging
logger = logging.getLogger(__name__)

# Global scheduler instance
_scheduler = None


def get_scheduler(
    settings: Settings = Depends(get_settings),
) -> Union[AsyncIOScheduler, BackgroundScheduler]:
    """Get the task scheduler instance.

    Args:
        settings: Application settings.

    Returns:
        The scheduler instance.
    """
    global _scheduler
    if _scheduler is None:
        setup_scheduler(settings)
    return _scheduler


def setup_scheduler(settings: Settings) -> Union[AsyncIOScheduler, BackgroundScheduler]:
    """Set up and configure the task scheduler.

    Args:
        settings: Application settings.

    Returns:
        The configured scheduler instance.
    """
    global _scheduler

    # Check if we already have a scheduler
    if _scheduler is not None:
        return _scheduler

    # Configure job stores
    jobstores = {
        "default": RedisJobStore(
            jobs_key="auditpulse:scheduler:jobs",
            run_times_key="auditpulse:scheduler:runtimes",
            host=settings.REDIS_URL.split("//")[1].split(":")[0],
            port=int(settings.REDIS_URL.split(":")[-1].split("/")[0]),
            db=int(settings.REDIS_URL.split("/")[-1]),
        )
    }

    # Configure executors
    executors = {
        "default": ThreadPoolExecutor(20),
        "processpool": ProcessPoolExecutor(5),
    }

    job_defaults = {"coalesce": True, "max_instances": 3, "misfire_grace_time": 30}

    # Use AsyncIOScheduler in dev/test, BackgroundScheduler in production
    if settings.ENVIRONMENT in ["development", "test"]:
        _scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
        )
    else:
        _scheduler = BackgroundScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
        )

    # Start the scheduler
    try:
        _scheduler.start()
        logger.info(f"Task scheduler started ({type(_scheduler).__name__})")
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
        raise

    return _scheduler


def register_task(
    task_func: Callable,
    task_id: Optional[str] = None,
    trigger: str = "interval",
    **trigger_args: Any,
) -> str:
    """Register a task with the scheduler.

    Args:
        task_func: The function to schedule.
        task_id: Optional ID for the task. If None, uses function name.
        trigger: The trigger type (interval, cron, date).
        **trigger_args: Arguments for the trigger.

    Returns:
        The ID of the scheduled job.
    """
    global _scheduler
    if _scheduler is None:
        settings = get_settings()
        setup_scheduler(settings)

    # Generate job ID if not provided
    if task_id is None:
        task_id = f"{task_func.__module__}.{task_func.__name__}"

    # Check if job already exists
    if _scheduler.get_job(task_id):
        logger.info(f"Task already registered: {task_id}")
        return task_id

    logger.info(
        f"Registering task: {task_id}, trigger: {trigger}, args: {trigger_args}"
    )

    # Handle async functions
    if inspect.iscoroutinefunction(task_func):
        # Wrap async function to run in event loop
        def async_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(task_func(*args, **kwargs))

        _scheduler.add_job(
            async_wrapper,
            trigger=trigger,
            id=task_id,
            replace_existing=True,
            **trigger_args,
        )
    else:
        _scheduler.add_job(
            task_func,
            trigger=trigger,
            id=task_id,
            replace_existing=True,
            **trigger_args,
        )

    return task_id
