"""System administration classes for monitoring and management.

This module provides classes for system monitoring and task management.
"""

import logging
import datetime
import psutil
import os
import time
import inspect
from typing import Any, Dict, List, Optional, Union

from apscheduler.job import Job
from apscheduler.schedulers.base import BaseScheduler
from fastapi import Depends
from pydantic import UUID4
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.database.models import (
    Tenant,
    User,
    Transaction,
    Anomaly,
    ErrorLog,
    SystemMetric,
)
from auditpulse_mvp.tasks.scheduler import get_scheduler
from auditpulse_mvp.tasks.tasks import (
    sync_data_task,
    retrain_models_task,
    detect_anomalies_task,
    send_notifications_task,
    cleanup_old_data_task,
    backup_database_task,
    update_metrics_task,
)
from auditpulse_mvp.utils.settings import get_settings, Settings
from auditpulse_mvp.config import settings
from auditpulse_mvp.tasks.task_manager import TaskManager
import platform
from redis import Redis


# Configure logging
logger = logging.getLogger(__name__)


class SystemStatusAdmin:
    """Admin class for monitoring system status."""

    def __init__(
        self,
        db_session: AsyncSession = Depends(get_db_session),
        settings: Settings = Depends(get_settings),
        scheduler: Optional[BaseScheduler] = None,
    ):
        """Initialize the system status admin.

        Args:
            db_session: Database session.
            settings: Application settings.
            scheduler: Optional scheduler instance. If None, uses get_scheduler.
        """
        self.db = db_session
        self.settings = settings
        self.scheduler = scheduler or get_scheduler(settings)

    async def get_system_overview(self) -> Dict[str, Any]:
        """Get an overview of the system status.

        Returns:
            Dictionary with system status information.
        """
        # Get system resources
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(os.getcwd())

        # Get database stats
        db_stats = await self._get_database_stats()

        # Get scheduler status
        scheduler_stats = self._get_scheduler_stats()

        # Combine all stats
        return {
            "status": "running",
            "environment": self.settings.ENVIRONMENT,
            "timestamp": datetime.datetime.now(),
            "system_resources": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024**3),
                "disk_total_gb": disk.total / (1024**3),
            },
            "database": db_stats,
            "scheduler": scheduler_stats,
            "api": {
                "status": "healthy",
            },
        }

    async def _get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with database statistics.
        """
        # Get counts
        tenant_count = await self._count_records(Tenant)
        user_count = await self._count_records(User)
        transaction_count = await self._count_records(Transaction)
        anomaly_count = await self._count_records(Anomaly)

        # Get active tenant and user counts
        active_tenant_count = await self._count_records(
            Tenant, Tenant.is_active == True
        )
        active_user_count = await self._count_records(User, User.is_active == True)

        # Get unresolved anomalies count
        unresolved_anomaly_count = await self._count_records(
            Anomaly, Anomaly.is_resolved == False
        )

        return {
            "status": "connected",
            "counts": {
                "tenants": tenant_count,
                "active_tenants": active_tenant_count,
                "users": user_count,
                "active_users": active_user_count,
                "transactions": transaction_count,
                "anomalies": anomaly_count,
                "unresolved_anomalies": unresolved_anomaly_count,
            },
        }

    async def _count_records(self, model, condition=None) -> int:
        """Count records in a table, optionally with a condition.

        Args:
            model: SQLAlchemy model.
            condition: Optional condition to filter by.

        Returns:
            Count of records.
        """
        stmt = select(func.count()).select_from(model)
        if condition is not None:
            stmt = stmt.where(condition)

        result = await self.db.execute(stmt)
        return result.scalar() or 0

    def _get_scheduler_stats(self) -> Dict[str, Any]:
        """Get task scheduler statistics.

        Returns:
            Dictionary with scheduler statistics.
        """
        jobs = self.scheduler.get_jobs()

        # Group jobs by state
        pending_jobs = [job for job in jobs if job.next_run_time is not None]
        paused_jobs = [job for job in jobs if job.next_run_time is None]

        # Get information about each job
        job_info = [
            {
                "id": job.id,
                "name": job.id.split(".")[-1] if "." in job.id else job.id,
                "next_run_time": job.next_run_time,
                "trigger": str(job.trigger),
            }
            for job in jobs
        ]

        # Sort by next run time
        job_info.sort(key=lambda j: j["next_run_time"] or datetime.datetime.max)

        return {
            "status": "running",
            "job_count": len(jobs),
            "pending_jobs": len(pending_jobs),
            "paused_jobs": len(paused_jobs),
            "jobs": job_info,
        }

    async def get_tenant_metrics(
        self, tenant_id: Optional[UUID4] = None
    ) -> Dict[str, Any]:
        """Get metrics for a specific tenant or all tenants.

        Args:
            tenant_id: Optional tenant ID. If None, gets metrics for all tenants.

        Returns:
            Dictionary with tenant metrics.
        """
        if tenant_id:
            # Get metrics for a specific tenant
            tenant_stmt = select(Tenant).where(Tenant.id == tenant_id)
            tenant_result = await self.db.execute(tenant_stmt)
            tenant = tenant_result.scalar_one_or_none()

            if not tenant:
                return {"error": f"Tenant with ID {tenant_id} not found"}

            # Get counts for this tenant
            user_count = await self._count_records(User, User.tenant_id == tenant_id)
            transaction_count = await self._count_records(
                Transaction, Transaction.tenant_id == tenant_id
            )
            anomaly_count = await self._count_records(
                Anomaly, Anomaly.tenant_id == tenant_id
            )
            unresolved_anomaly_count = await self._count_records(
                Anomaly,
                (Anomaly.tenant_id == tenant_id) & (Anomaly.is_resolved == False),
            )

            return {
                "tenant_id": str(tenant_id),
                "name": tenant.name,
                "slug": tenant.slug,
                "is_active": tenant.is_active,
                "counts": {
                    "users": user_count,
                    "transactions": transaction_count,
                    "anomalies": anomaly_count,
                    "unresolved_anomalies": unresolved_anomaly_count,
                },
            }
        else:
            # Get metrics for all tenants
            tenant_stmt = select(Tenant)
            tenant_result = await self.db.execute(tenant_stmt)
            tenants = list(tenant_result.scalars().all())

            tenant_metrics = []
            for tenant in tenants:
                # Get counts for this tenant
                user_count = await self._count_records(
                    User, User.tenant_id == tenant.id
                )
                transaction_count = await self._count_records(
                    Transaction, Transaction.tenant_id == tenant.id
                )
                anomaly_count = await self._count_records(
                    Anomaly, Anomaly.tenant_id == tenant.id
                )

                tenant_metrics.append(
                    {
                        "tenant_id": str(tenant.id),
                        "name": tenant.name,
                        "slug": tenant.slug,
                        "is_active": tenant.is_active,
                        "counts": {
                            "users": user_count,
                            "transactions": transaction_count,
                            "anomalies": anomaly_count,
                        },
                    }
                )

            return {
                "tenant_count": len(tenants),
                "active_tenant_count": sum(1 for t in tenants if t.is_active),
                "tenants": tenant_metrics,
            }


class TaskAdmin:
    """Admin class for managing scheduled tasks."""

    def __init__(
        self,
        db_session: AsyncSession = Depends(get_db_session),
        settings: Settings = Depends(get_settings),
        scheduler: Optional[BaseScheduler] = None,
    ):
        """Initialize the task admin.

        Args:
            db_session: Database session.
            settings: Application settings.
            scheduler: Optional scheduler instance. If None, uses get_scheduler.
        """
        self.db = db_session
        self.settings = settings
        self.scheduler = scheduler or get_scheduler(settings)

    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Get all scheduled jobs.

        Returns:
            List of job information dictionaries.
        """
        jobs = self.scheduler.get_jobs()

        # Convert to dictionaries
        job_info = [
            {
                "id": job.id,
                "name": job.id.split(".")[-1] if "." in job.id else job.id,
                "next_run_time": job.next_run_time,
                "trigger": str(job.trigger),
                "pending": job.next_run_time is not None,
            }
            for job in jobs
        ]

        # Sort by next run time
        job_info.sort(key=lambda j: j["next_run_time"] or datetime.datetime.max)

        return job_info

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific job.

        Args:
            job_id: The job ID.

        Returns:
            Dictionary with job information, or None if not found.
        """
        job = self.scheduler.get_job(job_id)
        if not job:
            return None

        return {
            "id": job.id,
            "name": job.id.split(".")[-1] if "." in job.id else job.id,
            "next_run_time": job.next_run_time,
            "trigger": str(job.trigger),
            "pending": job.next_run_time is not None,
            "args": job.args,
            "kwargs": job.kwargs,
        }

    def run_task(
        self,
        task_name: str,
        tenant_ids: Optional[List[UUID4]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run a task immediately.

        Args:
            task_name: Name of the task to run (sync_data, retrain_models, etc.).
            tenant_ids: Optional list of tenant IDs to process.
            **kwargs: Additional arguments for the task.

        Returns:
            Dictionary with job information.

        Raises:
            ValueError: If the task name is invalid.
        """
        # Map task names to functions
        task_map = {
            "sync_data": sync_data_task,
            "retrain_models": retrain_models_task,
            "detect_anomalies": detect_anomalies_task,
            "send_notifications": send_notifications_task,
            "cleanup_data": cleanup_old_data_task,
        }

        if task_name not in task_map:
            raise ValueError(
                f"Invalid task name: {task_name}. Must be one of {', '.join(task_map.keys())}"
            )

        # Generate a unique job ID
        job_id = (
            f"manual_{task_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Add the job
        task_func = task_map[task_name]
        job = self.scheduler.add_job(
            task_func,
            "date",  # Run once, immediately
            args=[tenant_ids] if tenant_ids else [],
            kwargs=kwargs,
            id=job_id,
            replace_existing=True,
        )

        logger.info(f"Scheduled immediate task: {task_name} (ID: {job_id})")

        return {
            "id": job.id,
            "name": task_name,
            "next_run_time": job.next_run_time,
            "trigger": str(job.trigger),
            "tenant_ids": [str(tid) for tid in tenant_ids] if tenant_ids else None,
            "status": "scheduled",
        }

    def update_job(
        self,
        job_id: str,
        paused: Optional[bool] = None,
        **trigger_args: Any,
    ) -> Optional[Dict[str, Any]]:
        """Update a scheduled job.

        Args:
            job_id: The job ID.
            paused: Whether to pause or resume the job.
            **trigger_args: Arguments to modify the job's trigger.

        Returns:
            Dictionary with updated job information, or None if not found.
        """
        job = self.scheduler.get_job(job_id)
        if not job:
            return None

        # Pause or resume the job
        if paused is not None:
            if paused:
                self.scheduler.pause_job(job_id)
                logger.info(f"Paused job: {job_id}")
            else:
                self.scheduler.resume_job(job_id)
                logger.info(f"Resumed job: {job_id}")

        # Reschedule if trigger args are provided
        if trigger_args:
            self.scheduler.reschedule_job(
                job_id, trigger=job.trigger.name, **trigger_args
            )
            logger.info(f"Rescheduled job: {job_id} with args: {trigger_args}")

        # Get updated job
        updated_job = self.scheduler.get_job(job_id)
        return {
            "id": updated_job.id,
            "name": (
                updated_job.id.split(".")[-1]
                if "." in updated_job.id
                else updated_job.id
            ),
            "next_run_time": updated_job.next_run_time,
            "trigger": str(updated_job.trigger),
            "pending": updated_job.next_run_time is not None,
            "args": updated_job.args,
            "kwargs": updated_job.kwargs,
        }

    def remove_job(self, job_id: str) -> bool:
        """Remove a scheduled job.

        Args:
            job_id: The job ID.

        Returns:
            True if removed, False if not found.
        """
        job = self.scheduler.get_job(job_id)
        if not job:
            return False

        self.scheduler.remove_job(job_id)
        logger.info(f"Removed job: {job_id}")

        return True

    def schedule_recurring_task(
        self,
        task_name: str,
        interval_type: str = "interval",
        tenant_ids: Optional[List[UUID4]] = None,
        **trigger_args: Any,
    ) -> Dict[str, Any]:
        """Schedule a recurring task.

        Args:
            task_name: Name of the task to run (sync_data, retrain_models, etc.).
            interval_type: Type of interval (interval, cron, date).
            tenant_ids: Optional list of tenant IDs to process.
            **trigger_args: Arguments for the trigger.

        Returns:
            Dictionary with job information.

        Raises:
            ValueError: If the task name or interval type is invalid.
        """
        # Map task names to functions
        task_map = {
            "sync_data": sync_data_task,
            "retrain_models": retrain_models_task,
            "detect_anomalies": detect_anomalies_task,
            "send_notifications": send_notifications_task,
            "cleanup_data": cleanup_old_data_task,
        }

        if task_name not in task_map:
            raise ValueError(
                f"Invalid task name: {task_name}. Must be one of {', '.join(task_map.keys())}"
            )

        if interval_type not in ["interval", "cron", "date"]:
            raise ValueError(
                f"Invalid interval type: {interval_type}. Must be one of interval, cron, date"
            )

        # Generate a job ID
        job_id = (
            f"scheduled_{task_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Add the job
        task_func = task_map[task_name]
        job = self.scheduler.add_job(
            task_func,
            interval_type,
            args=[tenant_ids] if tenant_ids else [],
            id=job_id,
            replace_existing=True,
            **trigger_args,
        )

        logger.info(
            f"Scheduled recurring task: {task_name} (ID: {job_id}) with trigger: {interval_type}"
        )

        return {
            "id": job.id,
            "name": task_name,
            "next_run_time": job.next_run_time,
            "trigger": str(job.trigger),
            "tenant_ids": [str(tid) for tid in tenant_ids] if tenant_ids else None,
            "interval_type": interval_type,
            "trigger_args": trigger_args,
            "status": "scheduled",
        }


class SystemAdmin:
    """Admin class for system administration tasks."""

    def __init__(self, db_session: AsyncSession):
        """Initialize the system admin.

        Args:
            db_session: Database session.
        """
        self.db = db_session
        self.start_time = time.time()
        self.task_manager = TaskManager()

    async def get_status(self) -> Dict[str, Any]:
        """Get system status information.

        Returns:
            Dictionary with system status information.
        """
        # Get uptime
        uptime_seconds = int(time.time() - self.start_time)

        # Check services
        service_status = await self._check_services()

        # Get scheduled jobs
        jobs = await self.task_manager.get_scheduled_jobs()

        # Get resource usage
        resource_usage = self._get_resource_usage()

        return {
            "status": "operational",
            "uptime_seconds": uptime_seconds,
            "service_status": service_status,
            "jobs": jobs,
            "resource_usage": resource_usage,
        }

    async def run_task(self, task_name: str) -> Dict[str, Any]:
        """Run a system maintenance task.

        Args:
            task_name: Name of the task to run.

        Returns:
            Dictionary with task execution results.
        """
        # Map task names to functions
        task_map = {
            "sync_data": sync_data_task,
            "cleanup_old_data": cleanup_old_data_task,
            "backup_database": backup_database_task,
            "update_metrics": update_metrics_task,
            "retrain_models": retrain_models_task,
        }

        if task_name not in task_map:
            available_tasks = ", ".join(task_map.keys())
            raise ValueError(
                f"Unknown task: {task_name}. Available tasks: {available_tasks}"
            )

        # Get task function
        task_func = task_map[task_name]

        # Schedule task for immediate execution
        task_id = f"manual_{task_name}_{datetime.datetime.now().timestamp()}"
        started_at = datetime.datetime.now()

        # Run the task
        try:
            # Check if the task requires the db session
            if "db_session" in inspect.signature(task_func).parameters:
                result = await task_func(db_session=self.db)
            else:
                result = await task_func()

            return {
                "task_id": task_id,
                "status": "completed",
                "message": f"Task {task_name} completed successfully",
                "started_at": started_at,
                "details": result,
            }
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error executing task {task_name}: {error_message}")

            # Log the error
            await self.log_error(
                component="system_admin",
                error_type="task_execution_error",
                message=f"Error executing task {task_name}: {error_message}",
                details={"task_name": task_name, "error": error_message},
            )

            return {
                "task_id": task_id,
                "status": "failed",
                "message": f"Task {task_name} failed: {error_message}",
                "started_at": started_at,
                "details": {"error": error_message},
            }

    async def get_error_logs(
        self,
        limit: int = 100,
        skip: int = 0,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        component: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get system error logs with filtering and pagination.

        Args:
            limit: Maximum number of records to return.
            skip: Number of records to skip.
            start_date: Optional start date for filtering.
            end_date: Optional end date for filtering.
            component: Optional component for filtering.
            error_type: Optional error type for filtering.

        Returns:
            List of error logs.
        """
        # Build query
        query = text(
            """
            SELECT id, timestamp, component, error_type, message, details
            FROM error_logs
            WHERE 1=1
        """
        )

        # Add filters
        params = {}

        if start_date:
            query = text(f"{query} AND timestamp >= :start_date")
            params["start_date"] = start_date

        if end_date:
            query = text(f"{query} AND timestamp <= :end_date")
            params["end_date"] = end_date

        if component:
            query = text(f"{query} AND component = :component")
            params["component"] = component

        if error_type:
            query = text(f"{query} AND error_type = :error_type")
            params["error_type"] = error_type

        # Add ordering and pagination
        query = text(
            f"""
            {query}
            ORDER BY timestamp DESC
            LIMIT :limit OFFSET :skip
        """
        )

        params["limit"] = limit
        params["skip"] = skip

        # Execute query
        result = await self.db.execute(query, params)
        logs = result.fetchall()

        # Format results
        return [
            {
                "id": str(log[0]),
                "timestamp": log[1],
                "component": log[2],
                "error_type": log[3],
                "message": log[4],
                "details": log[5],
            }
            for log in logs
        ]

    async def log_error(
        self,
        component: str,
        error_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> ErrorLog:
        """Log an error in the system.

        Args:
            component: Component where the error occurred.
            error_type: Type of error.
            message: Error message.
            details: Optional error details.

        Returns:
            Created error log.
        """
        # Create error log
        error_log = ErrorLog(
            component=component,
            error_type=error_type,
            message=message,
            details=details or {},
        )

        # Save to database
        self.db.add(error_log)
        await self.db.commit()
        await self.db.refresh(error_log)

        logger.error(f"Error in {component} ({error_type}): {message}")
        return error_log

    async def get_system_metrics(
        self,
        metric_name: Optional[str] = None,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get system metrics with filtering.

        Args:
            metric_name: Optional metric name for filtering.
            start_date: Optional start date for filtering.
            end_date: Optional end date for filtering.
            limit: Maximum number of records to return.

        Returns:
            List of system metrics.
        """
        # Build query
        query = text(
            """
            SELECT id, timestamp, name, value
            FROM system_metrics
            WHERE 1=1
        """
        )

        # Add filters
        params = {}

        if metric_name:
            query = text(f"{query} AND name = :metric_name")
            params["metric_name"] = metric_name

        if start_date:
            query = text(f"{query} AND timestamp >= :start_date")
            params["start_date"] = start_date

        if end_date:
            query = text(f"{query} AND timestamp <= :end_date")
            params["end_date"] = end_date

        # Add ordering and limit
        query = text(
            f"""
            {query}
            ORDER BY timestamp DESC
            LIMIT :limit
        """
        )

        params["limit"] = limit

        # Execute query
        result = await self.db.execute(query, params)
        metrics = result.fetchall()

        # Format results
        return [
            {
                "id": str(metric[0]),
                "timestamp": metric[1],
                "name": metric[2],
                "value": metric[3],
            }
            for metric in metrics
        ]

    async def record_metric(
        self, name: str, value: Union[int, float, str, bool]
    ) -> SystemMetric:
        """Record a system metric.

        Args:
            name: Metric name.
            value: Metric value.

        Returns:
            Created system metric.
        """
        # Create metric
        metric = SystemMetric(
            name=name,
            value=value,
        )

        # Save to database
        self.db.add(metric)
        await self.db.commit()
        await self.db.refresh(metric)

        return metric

    async def _check_services(self) -> Dict[str, str]:
        """Check the status of system services.

        Returns:
            Dictionary with service status information.
        """
        service_status = {
            "api": "healthy",
            "database": "unknown",
            "task_queue": "unknown",
            "scheduler": "unknown",
            "cache": "unknown",
        }

        # Check database
        try:
            await self.db.execute(text("SELECT 1"))
            service_status["database"] = "healthy"
        except Exception as e:
            service_status["database"] = "unhealthy"
            logger.error(f"Database health check failed: {str(e)}")

        # Check task queue and scheduler
        try:
            scheduler_status = await self.task_manager.get_scheduler_status()
            service_status["scheduler"] = (
                "healthy" if scheduler_status["running"] else "unhealthy"
            )
            service_status["task_queue"] = (
                "healthy" if scheduler_status["connected"] else "unhealthy"
            )
        except Exception as e:
            service_status["scheduler"] = "unhealthy"
            service_status["task_queue"] = "unhealthy"
            logger.error(f"Task scheduler health check failed: {str(e)}")

        # Check Redis cache
        try:
            redis = Redis.from_url(settings.REDIS_URL)
            await redis.ping()
            await redis.close()
            service_status["cache"] = "healthy"
        except Exception as e:
            service_status["cache"] = "unhealthy"
            logger.error(f"Redis health check failed: {str(e)}")

        return service_status

    def _get_resource_usage(self) -> Dict[str, Any]:
        """Get system resource usage information.

        Returns:
            Dictionary with resource usage information.
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_mb": memory.used / (1024 * 1024),
                "memory_total_mb": memory.total / (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024 * 1024 * 1024),
                "disk_total_gb": disk.total / (1024 * 1024 * 1024),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            }
        except Exception as e:
            logger.error(f"Error getting resource usage: {str(e)}")
            return {
                "error": str(e),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            }

    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive system health check.

        Returns:
            Dictionary with health check results.
        """
        # Get service status
        service_status = await self._check_services()

        # Check database connections
        db_connections = await self._check_database_connections()

        # Check API endpoints
        api_status = await self._check_api_endpoints()

        # Check disk space
        disk_space = self._check_disk_space()

        # Check memory usage
        memory_usage = self._check_memory_usage()

        # Get recent errors
        recent_errors = await self.get_error_logs(limit=5)

        # Determine overall health
        services_healthy = all(
            status == "healthy" for status in service_status.values()
        )
        db_healthy = db_connections["status"] == "healthy"
        api_healthy = api_status["status"] == "healthy"
        disk_healthy = disk_space["status"] == "healthy"
        memory_healthy = memory_usage["status"] == "healthy"

        overall_status = "healthy"
        if not (
            services_healthy
            and db_healthy
            and api_healthy
            and disk_healthy
            and memory_healthy
        ):
            overall_status = "degraded"

        # Result
        return {
            "timestamp": datetime.datetime.now(),
            "overall_status": overall_status,
            "services": service_status,
            "database": db_connections,
            "api": api_status,
            "disk_space": disk_space,
            "memory": memory_usage,
            "recent_errors": recent_errors,
        }

    async def _check_database_connections(self) -> Dict[str, Any]:
        """Check database connections.

        Returns:
            Dictionary with database connection information.
        """
        try:
            # Check active connections
            query = text(
                """
                SELECT COUNT(*) FROM pg_stat_activity
                WHERE state = 'active'
            """
            )
            result = await self.db.execute(query)
            active_connections = result.scalar()

            # Check max connections
            query = text("SHOW max_connections")
            result = await self.db.execute(query)
            max_connections = int(result.scalar())

            # Calculate usage percentage
            usage_percent = (active_connections / max_connections) * 100

            status = "healthy"
            if usage_percent > 80:
                status = "warning"
            if usage_percent > 95:
                status = "critical"

            return {
                "status": status,
                "active_connections": active_connections,
                "max_connections": max_connections,
                "usage_percent": usage_percent,
            }
        except Exception as e:
            logger.error(f"Error checking database connections: {str(e)}")
            return {
                "status": "error",
                "error_message": str(e),
            }

    async def _check_api_endpoints(self) -> Dict[str, Any]:
        """Check API endpoints.

        Returns:
            Dictionary with API endpoint information.
        """
        # In a real implementation, this would send HTTP requests to key endpoints
        # Here we'll simulate the check
        return {
            "status": "healthy",
            "endpoints_checked": 0,  # Placeholder
            "endpoints_healthy": 0,  # Placeholder
        }

    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space.

        Returns:
            Dictionary with disk space information.
        """
        try:
            disk = psutil.disk_usage("/")

            status = "healthy"
            if disk.percent > 80:
                status = "warning"
            if disk.percent > 95:
                status = "critical"

            return {
                "status": status,
                "used_gb": disk.used / (1024 * 1024 * 1024),
                "total_gb": disk.total / (1024 * 1024 * 1024),
                "percent_used": disk.percent,
            }
        except Exception as e:
            logger.error(f"Error checking disk space: {str(e)}")
            return {
                "status": "error",
                "error_message": str(e),
            }

    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage.

        Returns:
            Dictionary with memory usage information.
        """
        try:
            memory = psutil.virtual_memory()

            status = "healthy"
            if memory.percent > 80:
                status = "warning"
            if memory.percent > 95:
                status = "critical"

            return {
                "status": status,
                "used_mb": memory.used / (1024 * 1024),
                "total_mb": memory.total / (1024 * 1024),
                "percent_used": memory.percent,
            }
        except Exception as e:
            logger.error(f"Error checking memory usage: {str(e)}")
            return {
                "status": "error",
                "error_message": str(e),
            }

    async def send_system_alerts(self, alert_type: str, message: str) -> bool:
        """Send system alerts to administrators.

        Args:
            alert_type: Type of alert.
            message: Alert message.

        Returns:
            True if alert was sent, False otherwise.
        """
        # Get admin users
        admin_users = await self._get_admin_users()

        # Log the alert
        logger.warning(f"System alert ({alert_type}): {message}")

        # In a real implementation, this would send alerts via email/SMS/etc.
        # Here we'll just log it
        logger.info(f"Would send alert to {len(admin_users)} admin users")

        return True

    async def _get_admin_users(self) -> List[User]:
        """Get all admin users.

        Returns:
            List of admin users.
        """
        query = text(
            """
            SELECT * FROM users
            WHERE is_superuser = true
        """
        )
        result = await self.db.execute(query)

        # Convert row proxies to dictionaries
        users = []
        for row in result:
            user_dict = {column: getattr(row, column) for column in row._mapping}
            users.append(User(**user_dict))

        return users
