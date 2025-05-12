"""Scheduler for ML Engine operations.

This module provides the scheduling functionality for periodic ML operations
such as nightly model retraining.
"""
import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Callable, Dict, Optional, Union

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.ml_engine.ml_engine import MLEngine
from auditpulse_mvp.utils.settings import settings

# Configure logging
logger = logging.getLogger(__name__)


class MLScheduler:
    """Scheduler for ML Engine operations.
    
    This class handles scheduling recurring ML tasks, such as nightly model
    retraining for all tenants.
    """

    def __init__(self, db_session: AsyncSession = Depends(get_db_session)):
        """Initialize the scheduler.
        
        Args:
            db_session: Database session for data access.
        """
        self.db_session = db_session
        self.ml_engine = MLEngine(db_session=db_session)
        self._retraining_task = None
        self._is_running = False
    
    async def start(self):
        """Start the scheduler."""
        if self._is_running:
            logger.warning("ML Scheduler is already running")
            return
        
        self._is_running = True
        self._retraining_task = asyncio.create_task(self._schedule_daily_retraining())
        logger.info("ML Scheduler started")
    
    async def stop(self):
        """Stop the scheduler."""
        if not self._is_running:
            logger.warning("ML Scheduler is not running")
            return
        
        self._is_running = False
        if self._retraining_task:
            self._retraining_task.cancel()
            try:
                await self._retraining_task
            except asyncio.CancelledError:
                pass
            self._retraining_task = None
        
        logger.info("ML Scheduler stopped")
    
    async def _schedule_daily_retraining(self):
        """Schedule daily model retraining at the specified time."""
        try:
            while self._is_running:
                # Compute time until next scheduled retraining
                now = datetime.now()
                target_time = time(
                    hour=settings.ml_retraining_hour,
                    minute=settings.ml_retraining_minute,
                )
                
                target_datetime = datetime.combine(now.date(), target_time)
                
                # If we've already passed today's target time, schedule for tomorrow
                if now.time() >= target_time:
                    target_datetime += timedelta(days=1)
                
                # Calculate the wait time
                wait_seconds = (target_datetime - now).total_seconds()
                logger.info(
                    f"Next ML model retraining scheduled at "
                    f"{target_datetime.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"(in {wait_seconds:.2f} seconds)"
                )
                
                # Wait until the scheduled time
                await asyncio.sleep(wait_seconds)
                
                # Perform retraining
                if self._is_running:  # Check again in case we were stopped during the wait
                    await self._run_retraining()
                
        except asyncio.CancelledError:
            logger.info("ML Scheduler task cancelled")
            raise
        except Exception as e:
            logger.exception(f"Error in ML Scheduler: {e}")
            if self._is_running:
                # Restart the scheduler after a delay
                await asyncio.sleep(60)
                asyncio.create_task(self._schedule_daily_retraining())
    
    async def _run_retraining(self):
        """Run the model retraining for all tenants."""
        try:
            start_time = datetime.now()
            logger.info(f"Starting scheduled ML model retraining at {start_time.isoformat()}")
            
            # Run the retraining
            results = await self.ml_engine.retrain_all_tenant_models()
            
            # Summarize results
            success_count = sum(1 for r in results.values() if r.get("status") == "success")
            error_count = sum(1 for r in results.values() if r.get("status") == "error")
            skipped_count = sum(1 for r in results.values() if r.get("status") == "skipped")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(
                f"Completed scheduled ML model retraining in {duration:.2f} seconds. "
                f"Results: {success_count} successful, {error_count} errors, {skipped_count} skipped"
            )
            
            # Log detailed results for errors
            for tenant_id, result in results.items():
                if result.get("status") == "error":
                    logger.error(
                        f"Error retraining model for tenant {tenant_id}: {result.get('message')}"
                    )
            
            return results
        except Exception as e:
            logger.exception(f"Error running scheduled ML model retraining: {e}")
            return {"status": "error", "error": str(e)}
    
    async def run_immediate_retraining(self):
        """Run model retraining immediately, outside the schedule."""
        return await self._run_retraining()


# Global scheduler instance for the application
_ml_scheduler: Optional[MLScheduler] = None


async def get_ml_scheduler(db_session: AsyncSession = Depends(get_db_session)) -> MLScheduler:
    """Get or create the global ML scheduler instance.
    
    Args:
        db_session: Database session for data access.
        
    Returns:
        MLScheduler: The global ML scheduler instance.
    """
    global _ml_scheduler
    if _ml_scheduler is None:
        _ml_scheduler = MLScheduler(db_session=db_session)
    return _ml_scheduler


async def start_ml_scheduler():
    """Start the ML scheduler on application startup."""
    if settings.enable_ml_scheduler:
        scheduler = await get_ml_scheduler()
        await scheduler.start()
        logger.info("ML Scheduler started on application startup")


async def stop_ml_scheduler():
    """Stop the ML scheduler on application shutdown."""
    global _ml_scheduler
    if _ml_scheduler is not None:
        await _ml_scheduler.stop()
        logger.info("ML Scheduler stopped on application shutdown") 