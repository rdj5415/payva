"""Scheduler for continuous learning operations.

This module provides scheduling functionality for periodic processing of
user feedback and model retraining.
"""

import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Any

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.database.models import Tenant
from auditpulse_mvp.learning.feedback_learning import (
    FeedbackLearner,
    update_thresholds_from_feedback,
)
from auditpulse_mvp.utils.settings import settings

# Configure logging
logger = logging.getLogger(__name__)


class FeedbackLearningScheduler:
    """Scheduler for continuous learning operations.

    This class handles scheduling of periodic feedback processing and
    model retraining based on user feedback.
    """

    def __init__(self, db_session: AsyncSession = Depends(get_db_session)):
        """Initialize the learning scheduler.

        Args:
            db_session: Database session for data access.
        """
        self.db_session = db_session
        self.feedback_learner = FeedbackLearner(db_session=db_session)
        self._learning_task = None
        self._is_running = False

    async def start(self):
        """Start the scheduler."""
        if self._is_running:
            logger.warning("Feedback Learning Scheduler is already running")
            return

        self._is_running = True
        self._learning_task = asyncio.create_task(self._schedule_nightly_learning())
        logger.info("Feedback Learning Scheduler started")

    async def stop(self):
        """Stop the scheduler."""
        if not self._is_running:
            logger.warning("Feedback Learning Scheduler is not running")
            return

        self._is_running = False
        if self._learning_task:
            self._learning_task.cancel()
            try:
                await self._learning_task
            except asyncio.CancelledError:
                pass
            self._learning_task = None

        logger.info("Feedback Learning Scheduler stopped")

    async def _schedule_nightly_learning(self):
        """Schedule nightly learning processes at the specified time."""
        try:
            while self._is_running:
                # Compute time until next scheduled learning
                now = datetime.now()
                target_time = time(
                    hour=settings.feedback_learning_hour or 3,  # Default to 3 AM
                    minute=settings.feedback_learning_minute or 0,
                )

                target_datetime = datetime.combine(now.date(), target_time)

                # If we've already passed today's target time, schedule for tomorrow
                if now.time() >= target_time:
                    target_datetime += timedelta(days=1)

                # Calculate the wait time
                wait_seconds = (target_datetime - now).total_seconds()
                logger.info(
                    f"Next feedback learning process scheduled at "
                    f"{target_datetime.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"(in {wait_seconds:.2f} seconds)"
                )

                # Wait until the scheduled time
                await asyncio.sleep(wait_seconds)

                # Perform learning
                if (
                    self._is_running
                ):  # Check again in case we were stopped during the wait
                    await self._run_feedback_learning()

        except asyncio.CancelledError:
            logger.info("Feedback Learning Scheduler task cancelled")
            raise
        except Exception as e:
            logger.exception(f"Error in Feedback Learning Scheduler: {e}")
            if self._is_running:
                # Restart the scheduler after a delay
                await asyncio.sleep(60)
                asyncio.create_task(self._schedule_nightly_learning())

    async def _run_feedback_learning(self):
        """Run the feedback learning process for all tenants."""
        try:
            start_time = datetime.now()
            logger.info(
                f"Starting scheduled feedback learning at {start_time.isoformat()}"
            )

            # Get all active tenants
            stmt = select(Tenant).where(Tenant.is_active == True)
            result = await self.db_session.execute(stmt)
            tenants = result.scalars().all()

            # Process each tenant
            results = {}
            for tenant in tenants:
                try:
                    # Process feedback for this tenant
                    tenant_result = await update_thresholds_from_feedback(
                        tenant.id, self.db_session
                    )
                    results[str(tenant.id)] = tenant_result

                    if tenant_result["status"] == "success":
                        logger.info(
                            f"Successfully processed feedback for tenant {tenant.id}: "
                            f"fp_rate={tenant_result.get('false_positive_rate', 'N/A')}, "
                            f"model_retrained={tenant_result.get('model_retrained', False)}"
                        )
                    elif tenant_result["status"] == "skipped":
                        logger.info(
                            f"Skipped feedback processing for tenant {tenant.id}: "
                            f"{tenant_result.get('reason', 'No reason provided')}"
                        )
                    else:
                        logger.warning(
                            f"Failed to process feedback for tenant {tenant.id}: "
                            f"{tenant_result.get('error', 'Unknown error')}"
                        )
                except Exception as e:
                    logger.exception(
                        f"Error processing feedback for tenant {tenant.id}: {e}"
                    )
                    results[str(tenant.id)] = {
                        "status": "error",
                        "error": str(e),
                    }

            # Calculate success rate
            total_tenants = len(tenants)
            success_count = sum(
                1 for r in results.values() if r.get("status") == "success"
            )
            skipped_count = sum(
                1 for r in results.values() if r.get("status") == "skipped"
            )
            error_count = sum(1 for r in results.values() if r.get("status") == "error")

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info(
                f"Completed scheduled feedback learning in {duration:.2f} seconds. "
                f"Results: {success_count} successful, {skipped_count} skipped, {error_count} errors "
                f"out of {total_tenants} tenants."
            )

            return {
                "status": "completed",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "total_tenants": total_tenants,
                "success_count": success_count,
                "skipped_count": skipped_count,
                "error_count": error_count,
                "results": results,
            }
        except Exception as e:
            logger.exception(f"Error running scheduled feedback learning: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    async def run_immediate_learning(self, tenant_id=None):
        """Run feedback learning immediately, outside the schedule.

        Args:
            tenant_id: Optional tenant ID to process only one tenant.
                      If None, process all active tenants.

        Returns:
            Dict containing the results of the learning process.
        """
        if tenant_id:
            # Process just one tenant
            return await update_thresholds_from_feedback(tenant_id, self.db_session)
        else:
            # Process all tenants
            return await self._run_feedback_learning()


async def get_feedback_learning_scheduler(
    db_session: AsyncSession = Depends(get_db_session),
) -> FeedbackLearningScheduler:
    """Get or create the global feedback learning scheduler instance.

    Args:
        db_session: Database session for data access.

    Returns:
        FeedbackLearningScheduler: The global feedback learning scheduler instance.
    """
    scheduler = FeedbackLearningScheduler(db_session=db_session)
    return scheduler


async def start_feedback_learning_scheduler():
    """Start the feedback learning scheduler on application startup."""
    if settings.enable_feedback_learning:
        scheduler = await get_feedback_learning_scheduler()
        await scheduler.start()
        logger.info("Feedback Learning Scheduler started on application startup")


async def stop_feedback_learning_scheduler():
    """Stop the feedback learning scheduler on application shutdown."""
    scheduler = await get_feedback_learning_scheduler()
    await scheduler.stop()
    logger.info("Feedback Learning Scheduler stopped on application shutdown")
