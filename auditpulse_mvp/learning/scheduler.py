"""Scheduler for automated feedback learning processes."""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, time
from typing import Dict, Any, List, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.database.models import Tenant
from auditpulse_mvp.learning.feedback_learning import (
    FeedbackLearner,
    get_feedback_learner,
)

# Configure logging
logger = logging.getLogger(__name__)

# Default time for feedback learning: 3 AM
DEFAULT_SCHEDULE_HOUR = 3
DEFAULT_SCHEDULE_MINUTE = 0


class FeedbackLearningScheduler:
    """Scheduler for running feedback learning on a regular basis.

    This class manages scheduling and execution of feedback learning processes
    to continuously improve anomaly detection based on user feedback.
    """

    def __init__(
        self,
        db_session: AsyncSession = None,
        hour: int = DEFAULT_SCHEDULE_HOUR,
        minute: int = DEFAULT_SCHEDULE_MINUTE,
    ):
        """Initialize the feedback learning scheduler.

        Args:
            db_session: Database session for querying tenants and anomalies
            hour: Hour of day to run feedback learning (default: 3 AM)
            minute: Minute of hour to run feedback learning (default: 0)
        """
        self.db_session = db_session
        self.scheduler = AsyncIOScheduler()
        self.feedback_learner = None
        self.is_running = False
        self.hour = hour
        self.minute = minute

    async def start(self):
        """Start the feedback learning scheduler."""
        if self.is_running:
            logger.info("Feedback learning scheduler is already running")
            return

        # Get the feedback learner
        self.feedback_learner = await get_feedback_learner(db_session=self.db_session)

        # Schedule the feedback learning process
        self.scheduler.add_job(
            self.run_feedback_learning,
            CronTrigger(hour=self.hour, minute=self.minute),
            id="feedback_learning",
            name="Feedback Learning",
            replace_existing=True,
        )

        # Start the scheduler
        self.scheduler.start()
        self.is_running = True

        logger.info(
            f"Feedback learning scheduler started, running daily at {self.hour:02d}:{self.minute:02d}"
        )

    def stop(self):
        """Stop the feedback learning scheduler."""
        if not self.is_running:
            logger.info("Feedback learning scheduler is not running")
            return

        # Shut down the scheduler
        self.scheduler.shutdown(wait=False)
        self.is_running = False

        logger.info("Feedback learning scheduler stopped")

    async def run_feedback_learning(self):
        """Run the feedback learning process for all tenants."""
        logger.info("Starting feedback learning process for all tenants")

        try:
            # Get all tenant IDs
            tenant_ids = await get_all_tenant_ids(self.db_session)

            # Process each tenant
            logger.info(f"Processing feedback for {len(tenant_ids)} tenants")
            results = {}

            for tenant_id in tenant_ids:
                try:
                    # Process feedback for this tenant
                    tenant_result = await self.feedback_learner.process_recent_feedback(
                        tenant_id=tenant_id, days=30
                    )

                    results[str(tenant_id)] = tenant_result

                    logger.info(
                        f"Feedback learning completed for tenant {tenant_id}: "
                        f"status={tenant_result['status']}, "
                        f"processed={tenant_result.get('processed_count', 0)}"
                    )
                except Exception as e:
                    # Log the error but continue with other tenants
                    logger.error(
                        f"Error processing feedback for tenant {tenant_id}: {e}"
                    )
                    results[str(tenant_id)] = {"status": "error", "error": str(e)}

            logger.info(f"Feedback learning process completed for all tenants")

            return {
                "status": "completed",
                "total_tenants": len(tenant_ids),
                "successful_tenants": sum(
                    1 for r in results.values() if r["status"] == "success"
                ),
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error running feedback learning process: {e}")
            return {"status": "error", "error": str(e)}


async def get_all_tenant_ids(db_session: AsyncSession) -> List[uuid.UUID]:
    """Get all tenant IDs from the database.

    Args:
        db_session: Database session

    Returns:
        List of tenant IDs
    """
    stmt = select(Tenant)
    result = await db_session.execute(stmt)
    tenants = result.scalars().all()
    return [tenant.id for tenant in tenants]


# Global scheduler instance
_feedback_learning_scheduler = None


async def get_feedback_learning_scheduler(
    db_session: AsyncSession = None,
) -> FeedbackLearningScheduler:
    """Get the global feedback learning scheduler instance.

    Args:
        db_session: Database session

    Returns:
        FeedbackLearningScheduler: The scheduler instance
    """
    global _feedback_learning_scheduler

    if _feedback_learning_scheduler is None:
        if db_session is None:
            db_session = await anext(get_db_session())

        _feedback_learning_scheduler = FeedbackLearningScheduler(db_session=db_session)

    return _feedback_learning_scheduler


# Fix function parameters with Optional type
async def update_rule_thresholds(
    db_session: Optional[AsyncSession] = None,
) -> None:
    """Update rule thresholds based on feedback data."""
