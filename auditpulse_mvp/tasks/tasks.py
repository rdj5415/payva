"""Task definitions for scheduled jobs.

This module defines all scheduled tasks and background jobs for the application.
"""

import asyncio
import datetime
import logging
from typing import List, Optional

from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.database.models import Tenant, Transaction
from auditpulse_mvp.ingestion.quickbooks import QuickBooksService
from auditpulse_mvp.ingestion.plaid import PlaidService
from auditpulse_mvp.ml_engine.ml_engine import MLEngine
from auditpulse_mvp.alerts.notification_service import NotificationService
from auditpulse_mvp.risk_engine.risk_engine import RiskEngine
from auditpulse_mvp.utils.settings import get_settings

# Configure logging
logger = logging.getLogger(__name__)


async def sync_data_task(tenant_ids: Optional[List[str]] = None):
    """Sync transaction data from external sources.

    Args:
        tenant_ids: Optional list of tenant IDs to sync. If None, syncs all active tenants.
    """
    logger.info(
        f"Starting transaction data sync task for tenants: {tenant_ids or 'all'}"
    )

    async with get_db_session() as db:
        # Get tenants to process
        if tenant_ids:
            # Query specific tenants
            tenants = (
                await db.query(Tenant)
                .filter(Tenant.id.in_(tenant_ids), Tenant.is_active == True)
                .all()
            )
        else:
            # Get all active tenants
            tenants = await db.query(Tenant).filter(Tenant.is_active == True).all()

        for tenant in tenants:
            try:
                # Check tenant data sources
                if tenant.quickbooks_enabled:
                    logger.info(f"Syncing QuickBooks data for tenant {tenant.id}")
                    qb_service = QuickBooksService(db, tenant.id)
                    await qb_service.sync_transactions()

                if tenant.plaid_enabled:
                    logger.info(f"Syncing Plaid data for tenant {tenant.id}")
                    plaid_service = PlaidService(db, tenant.id)
                    await plaid_service.sync_transactions()

            except Exception as e:
                logger.error(f"Error syncing data for tenant {tenant.id}: {e}")

    logger.info("Transaction data sync task completed")


async def retrain_models_task(tenant_ids: Optional[List[str]] = None):
    """Retrain machine learning models for anomaly detection.

    Args:
        tenant_ids: Optional list of tenant IDs to retrain. If None, retrains all active tenants.
    """
    logger.info(f"Starting ML model retraining task for tenants: {tenant_ids or 'all'}")

    async with get_db_session() as db:
        ml_engine = MLEngine(db)

        # Batch train models
        result = await ml_engine.batch_train_models(tenant_ids)

        logger.info(
            f"ML model retraining results: {result['success']} successful, {result['error']} failed"
        )

    logger.info("ML model retraining task completed")


async def detect_anomalies_task(
    tenant_ids: Optional[List[str]] = None, days_back: int = 7
):
    """Run anomaly detection on recent transactions.

    Args:
        tenant_ids: Optional list of tenant IDs to process. If None, processes all active tenants.
        days_back: Number of days back to check transactions.
    """
    logger.info(f"Starting anomaly detection task for tenants: {tenant_ids or 'all'}")

    # Calculate date threshold
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_back)

    async with get_db_session() as db:
        # Get risk engine
        risk_engine = RiskEngine(db)

        # Get tenants to process
        if tenant_ids:
            tenants = (
                await db.query(Tenant)
                .filter(Tenant.id.in_(tenant_ids), Tenant.is_active == True)
                .all()
            )
        else:
            tenants = await db.query(Tenant).filter(Tenant.is_active == True).all()

        for tenant in tenants:
            try:
                # Get recent transactions that haven't been evaluated
                transactions = (
                    await db.query(Transaction)
                    .filter(
                        Transaction.tenant_id == tenant.id,
                        Transaction.transaction_date >= cutoff_date,
                        Transaction.is_evaluated == False,
                        Transaction.is_deleted == False,
                    )
                    .all()
                )

                logger.info(
                    f"Processing {len(transactions)} transactions for tenant {tenant.id}"
                )

                # Process each transaction
                anomalies_found = 0
                for transaction in transactions:
                    anomaly = await risk_engine.evaluate_and_store(
                        tenant.id, transaction
                    )

                    # Mark transaction as evaluated
                    transaction.is_evaluated = True
                    await db.commit()

                    if anomaly:
                        anomalies_found += 1

                logger.info(f"Found {anomalies_found} anomalies for tenant {tenant.id}")

            except Exception as e:
                logger.error(f"Error detecting anomalies for tenant {tenant.id}: {e}")
                await db.rollback()

    logger.info("Anomaly detection task completed")


async def send_notifications_task():
    """Send pending notifications for detected anomalies."""
    logger.info("Starting notification delivery task")

    async with get_db_session() as db:
        notification_service = NotificationService(db)

        # Process pending notifications
        count = await notification_service.send_pending_notifications()

        logger.info(f"Sent {count} notifications")

    logger.info("Notification delivery task completed")


async def cleanup_old_data_task(retention_days: int = 730):  # Default: 2 years
    """Clean up old data to maintain database performance.

    Args:
        retention_days: Number of days to retain data.
    """
    logger.info(f"Starting data cleanup task (retention: {retention_days} days)")

    # Calculate cutoff date
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=retention_days)

    async with get_db_session() as db:
        try:
            # Soft delete old transactions
            deleted_count = (
                await db.query(Transaction)
                .filter(
                    Transaction.transaction_date < cutoff_date,
                    Transaction.is_deleted == False,
                )
                .update({"is_deleted": True})
            )

            await db.commit()
            logger.info(f"Soft-deleted {deleted_count} old transactions")

        except Exception as e:
            logger.error(f"Error during data cleanup: {e}")
            await db.rollback()

    logger.info("Data cleanup task completed")


async def backup_database_task():
    """Placeholder for database backup task."""
    # TODO: Implement actual backup logic
    return {"status": "success", "message": "Database backup not yet implemented."}


async def update_metrics_task():
    """Placeholder for update metrics task."""
    # TODO: Implement actual metrics update logic
    return {"status": "success", "message": "Metrics update not yet implemented."}
