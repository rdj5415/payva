"""Task Manager for AuditPulse MVP.

This module provides a robust task management system with monitoring, retries, and prioritization.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
import uuid

from apscheduler.job import Job
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, extract

from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.database.models import TaskLog, TaskStatus
from auditpulse_mvp.utils.settings import get_settings, Settings

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class TaskManager:
    """Manages task scheduling, monitoring, and retries."""
    
    def __init__(
        self,
        db_session: AsyncSession = Depends(get_db_session),
        settings: Settings = Depends(get_settings),
    ):
        """Initialize the task manager.
        
        Args:
            db_session: Database session
            settings: Application settings
        """
        self.db = db_session
        self.settings = settings
        self.scheduler = AsyncIOScheduler()
        self._task_handlers: Dict[str, Callable] = {}
        self._task_queues: Dict[TaskPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in TaskPriority
        }
        self._is_running = False
        self._worker_tasks: List[asyncio.Task] = []
        
    async def start(self):
        """Start the task manager."""
        if self._is_running:
            logger.warning("Task manager is already running")
            return
            
        self._is_running = True
        self.scheduler.start()
        
        # Start worker tasks for each priority level
        for priority in TaskPriority:
            worker = asyncio.create_task(
                self._process_queue(priority)
            )
            self._worker_tasks.append(worker)
            
        logger.info("Task manager started")
        
    async def stop(self):
        """Stop the task manager."""
        if not self._is_running:
            logger.warning("Task manager is not running")
            return
            
        self._is_running = False
        self.scheduler.shutdown()
        
        # Cancel all worker tasks
        for task in self._worker_tasks:
            task.cancel()
            
        # Wait for tasks to complete
        for task in self._worker_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
                
        self._worker_tasks = []
        logger.info("Task manager stopped")
        
    def register_task(
        self,
        task_name: str,
        handler: Callable,
        max_retries: int = 3,
        retry_delay: int = 300,  # 5 minutes
    ):
        """Register a task handler.
        
        Args:
            task_name: Name of the task
            handler: Async function to handle the task
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self._task_handlers[task_name] = {
            "handler": handler,
            "max_retries": max_retries,
            "retry_delay": retry_delay,
        }
        logger.info(f"Registered task handler: {task_name}")
        
    async def schedule_task(
        self,
        task_name: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        run_at: Optional[datetime] = None,
        interval: Optional[timedelta] = None,
        cron: Optional[str] = None,
    ) -> str:
        """Schedule a task for execution.
        
        Args:
            task_name: Name of the task to schedule
            priority: Task priority
            args: Positional arguments for the task
            kwargs: Keyword arguments for the task
            run_at: Specific time to run the task
            interval: Interval between task runs
            cron: Cron expression for scheduling
            
        Returns:
            str: Task ID
            
        Raises:
            ValueError: If task_name is not registered or scheduling parameters are invalid
        """
        if task_name not in self._task_handlers:
            raise ValueError(f"Unknown task: {task_name}")
            
        task_id = str(uuid.uuid4())
        task_info = {
            "id": task_id,
            "name": task_name,
            "priority": priority,
            "args": args or [],
            "kwargs": kwargs or {},
            "status": TaskStatus.PENDING,
            "created_at": datetime.now(),
            "retry_count": 0,
        }
        
        # Log task creation
        await self._log_task(task_info)
        
        if run_at:
            # Schedule for specific time
            self.scheduler.add_job(
                self._execute_task,
                "date",
                run_date=run_at,
                args=[task_info],
                id=task_id,
            )
        elif interval:
            # Schedule with interval
            self.scheduler.add_job(
                self._execute_task,
                IntervalTrigger(seconds=interval.total_seconds()),
                args=[task_info],
                id=task_id,
            )
        elif cron:
            # Schedule with cron expression
            self.scheduler.add_job(
                self._execute_task,
                CronTrigger.from_crontab(cron),
                args=[task_info],
                id=task_id,
            )
        else:
            # Add to priority queue for immediate execution
            await self._task_queues[priority].put(task_info)
            
        return task_id
        
    async def _process_queue(self, priority: TaskPriority):
        """Process tasks in a priority queue.
        
        Args:
            priority: Priority level to process
        """
        queue = self._task_queues[priority]
        
        while self._is_running:
            try:
                # Get task from queue
                task_info = await queue.get()
                
                # Execute task
                await self._execute_task(task_info)
                
                # Mark task as done
                queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing task in queue {priority}: {e}")
                
    async def _execute_task(self, task_info: Dict[str, Any]):
        """Execute a task with retry logic.
        
        Args:
            task_info: Task information dictionary
        """
        task_id = task_info["id"]
        task_name = task_info["name"]
        handler_info = self._task_handlers[task_name]
        
        try:
            # Update task status
            task_info["status"] = TaskStatus.RUNNING
            task_info["started_at"] = datetime.now()
            await self._log_task(task_info)
            
            # Execute task
            handler = handler_info["handler"]
            result = await handler(*task_info["args"], **task_info["kwargs"])
            
            # Update task status
            task_info["status"] = TaskStatus.COMPLETED
            task_info["completed_at"] = datetime.now()
            task_info["result"] = result
            await self._log_task(task_info)
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            
            # Handle retries
            if task_info["retry_count"] < handler_info["max_retries"]:
                task_info["retry_count"] += 1
                task_info["status"] = TaskStatus.RETRYING
                task_info["last_error"] = str(e)
                await self._log_task(task_info)
                
                # Schedule retry
                retry_delay = handler_info["retry_delay"]
                await asyncio.sleep(retry_delay)
                await self._task_queues[task_info["priority"]].put(task_info)
            else:
                # Max retries exceeded
                task_info["status"] = TaskStatus.FAILED
                task_info["completed_at"] = datetime.now()
                task_info["last_error"] = str(e)
                await self._log_task(task_info)
                
    async def _log_task(self, task_info: Dict[str, Any]):
        """Log task information to database.
        
        Args:
            task_info: Task information dictionary
        """
        task_log = TaskLog(
            task_id=task_info["id"],
            task_name=task_info["name"],
            status=task_info["status"],
            priority=task_info["priority"].value,
            args=task_info.get("args"),
            kwargs=task_info.get("kwargs"),
            result=task_info.get("result"),
            error=task_info.get("last_error"),
            created_at=task_info["created_at"],
            started_at=task_info.get("started_at"),
            completed_at=task_info.get("completed_at"),
            retry_count=task_info.get("retry_count", 0),
        )
        
        self.db.add(task_log)
        await self.db.commit()
        
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Optional[Dict[str, Any]]: Task status information
        """
        # Get latest task log
        stmt = (
            select(TaskLog)
            .where(TaskLog.task_id == task_id)
            .order_by(TaskLog.created_at.desc())
        )
        result = await self.db.execute(stmt)
        task_log = result.scalar_one_or_none()
        
        if not task_log:
            return None
            
        return {
            "id": task_log.task_id,
            "name": task_log.task_name,
            "status": task_log.status,
            "priority": TaskPriority(task_log.priority).name,
            "created_at": task_log.created_at,
            "started_at": task_log.started_at,
            "completed_at": task_log.completed_at,
            "retry_count": task_log.retry_count,
            "error": task_log.error,
            "result": task_log.result,
        }
        
    async def get_task_stats(self) -> Dict[str, Any]:
        """Get task execution statistics.
        
        Returns:
            Dict[str, Any]: Task statistics
        """
        # Get task counts by status
        stmt = (
            select(TaskLog.status, func.count(TaskLog.id))
            .group_by(TaskLog.status)
        )
        result = await self.db.execute(stmt)
        status_counts = dict(result.all())
        
        # Get task counts by priority
        stmt = (
            select(TaskLog.priority, func.count(TaskLog.id))
            .group_by(TaskLog.priority)
        )
        result = await self.db.execute(stmt)
        priority_counts = {
            TaskPriority(priority).name: count
            for priority, count in result.all()
        }
        
        # Get average execution time
        stmt = (
            select(
                func.avg(
                    extract('epoch', TaskLog.completed_at) -
                    extract('epoch', TaskLog.started_at)
                )
            )
            .where(
                TaskLog.status == TaskStatus.COMPLETED,
                TaskLog.started_at.isnot(None),
                TaskLog.completed_at.isnot(None)
            )
        )
        result = await self.db.execute(stmt)
        avg_execution_time = result.scalar_one_or_none() or 0
        
        return {
            "status_counts": status_counts,
            "priority_counts": priority_counts,
            "avg_execution_time": avg_execution_time,
            "total_tasks": sum(status_counts.values()),
        } 