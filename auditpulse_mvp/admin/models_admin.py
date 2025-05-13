"""Models admin classes for Transaction, Anomaly, and ML Models.

This module provides classes for managing transactions, anomalies, and ML models in the admin interface.
"""

import logging
import datetime
import json
from typing import Any, Dict, List, Optional, TypeVar, Type, Generic, Union

from sqlalchemy import select, update, delete, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import UUID4

from auditpulse_mvp.database.models import (
    Transaction,
    Anomaly,
    AnomalyType,
    FeedbackType,
)
from auditpulse_mvp.database.models import ModelVersion, ModelPerformance
from auditpulse_mvp.ml_engine.model_manager import ModelManager


# Configure logging
logger = logging.getLogger(__name__)


class TransactionAdmin:
    """Admin class for managing transactions."""

    def __init__(self, db_session: AsyncSession):
        """Initialize the transaction admin.

        Args:
            db_session: Database session.
        """
        self.db = db_session

    async def get_all(
        self,
        tenant_id: UUID4,
        skip: int = 0,
        limit: int = 100,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
    ) -> List[Transaction]:
        """Get all transactions with filtering and pagination.

        Args:
            tenant_id: The tenant ID.
            skip: Number of records to skip.
            limit: Maximum number of records to return.
            start_date: Optional start date for filtering.
            end_date: Optional end date for filtering.
            min_amount: Optional minimum amount for filtering.
            max_amount: Optional maximum amount for filtering.

        Returns:
            List of transactions.
        """
        # Build query conditions
        conditions = [
            Transaction.tenant_id == tenant_id,
            Transaction.is_deleted == False,
        ]

        if start_date:
            conditions.append(Transaction.transaction_date >= start_date)

        if end_date:
            conditions.append(Transaction.transaction_date <= end_date)

        if min_amount is not None:
            conditions.append(Transaction.amount >= min_amount)

        if max_amount is not None:
            conditions.append(Transaction.amount <= max_amount)

        # Execute query
        stmt = (
            select(Transaction)
            .where(and_(*conditions))
            .order_by(Transaction.transaction_date.desc())
            .offset(skip)
            .limit(limit)
        )

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_by_id(self, transaction_id: UUID4) -> Optional[Transaction]:
        """Get a transaction by ID.

        Args:
            transaction_id: The transaction ID.

        Returns:
            The transaction or None if not found.
        """
        stmt = select(Transaction).where(Transaction.id == transaction_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_stats(
        self,
        tenant_id: UUID4,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
    ) -> Dict[str, Any]:
        """Get transaction statistics.

        Args:
            tenant_id: The tenant ID.
            start_date: Optional start date for filtering.
            end_date: Optional end date for filtering.

        Returns:
            Dictionary of statistics.
        """
        # Build query conditions
        conditions = [
            Transaction.tenant_id == tenant_id,
            Transaction.is_deleted == False,
        ]

        if start_date:
            conditions.append(Transaction.transaction_date >= start_date)

        if end_date:
            conditions.append(Transaction.transaction_date <= end_date)

        # Count total transactions
        count_stmt = (
            select(func.count()).select_from(Transaction).where(and_(*conditions))
        )
        count_result = await self.db.execute(count_stmt)
        total_count = count_result.scalar()

        # Sum total amount
        sum_stmt = (
            select(func.sum(Transaction.amount))
            .select_from(Transaction)
            .where(and_(*conditions))
        )
        sum_result = await self.db.execute(sum_stmt)
        total_amount = sum_result.scalar() or 0

        # Count transactions by source
        source_stmt = (
            select(Transaction.source, func.count())
            .where(and_(*conditions))
            .group_by(Transaction.source)
        )
        source_result = await self.db.execute(source_stmt)
        sources = dict(source_result.all())

        # Get average amount
        avg_stmt = (
            select(func.avg(Transaction.amount))
            .select_from(Transaction)
            .where(and_(*conditions))
        )
        avg_result = await self.db.execute(avg_stmt)
        avg_amount = avg_result.scalar() or 0

        return {
            "total_count": total_count,
            "total_amount": total_amount,
            "average_amount": avg_amount,
            "sources": sources,
        }

    async def mark_deleted(self, transaction_id: UUID4) -> bool:
        """Mark a transaction as deleted.

        Args:
            transaction_id: The transaction ID.

        Returns:
            True if marked deleted, False if not found.
        """
        db_transaction = await self.get_by_id(transaction_id)
        if not db_transaction:
            return False

        db_transaction.is_deleted = True
        await self.db.commit()

        logger.info(f"Marked transaction as deleted: {transaction_id}")
        return True

    async def search_transactions(
        self,
        tenant_id: UUID4,
        query: str,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        limit: int = 50,
    ) -> List[Transaction]:
        """Search transactions by description or account.

        Args:
            tenant_id: The tenant ID.
            query: Search term.
            start_date: Optional start date for filtering.
            end_date: Optional end date for filtering.
            limit: Maximum number of records to return.

        Returns:
            List of matching transactions.
        """
        conditions = [
            Transaction.tenant_id == tenant_id,
            Transaction.is_deleted == False,
            or_(
                Transaction.description.ilike(f"%{query}%"),
                Transaction.account_id.ilike(f"%{query}%"),
                Transaction.category.ilike(f"%{query}%"),
                Transaction.merchant_name.ilike(f"%{query}%"),
            ),
        ]

        if start_date:
            conditions.append(Transaction.transaction_date >= start_date)

        if end_date:
            conditions.append(Transaction.transaction_date <= end_date)

        stmt = (
            select(Transaction)
            .where(and_(*conditions))
            .order_by(Transaction.transaction_date.desc())
            .limit(limit)
        )

        result = await self.db.execute(stmt)
        return list(result.scalars().all())


class AnomalyAdmin:
    """Admin class for managing anomalies."""

    def __init__(self, db_session: AsyncSession):
        """Initialize the anomaly admin.

        Args:
            db_session: Database session.
        """
        self.db = db_session

    async def get_all(
        self,
        tenant_id: UUID4,
        skip: int = 0,
        limit: int = 100,
        is_resolved: Optional[bool] = None,
        anomaly_type: Optional[AnomalyType] = None,
        min_score: Optional[float] = None,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
    ) -> List[Anomaly]:
        """Get all anomalies with filtering and pagination.

        Args:
            tenant_id: The tenant ID.
            skip: Number of records to skip.
            limit: Maximum number of records to return.
            is_resolved: Optional filter by resolution status.
            anomaly_type: Optional filter by anomaly type.
            min_score: Optional minimum score.
            start_date: Optional start date for filtering.
            end_date: Optional end date for filtering.

        Returns:
            List of anomalies.
        """
        # Join with Transaction to get transaction dates
        stmt = (
            select(Anomaly)
            .join(Transaction, Anomaly.transaction_id == Transaction.id)
            .where(Anomaly.tenant_id == tenant_id)
        )

        # Add filters
        if is_resolved is not None:
            stmt = stmt.where(Anomaly.is_resolved == is_resolved)

        if anomaly_type:
            stmt = stmt.where(Anomaly.anomaly_type == anomaly_type)

        if min_score is not None:
            stmt = stmt.where(Anomaly.score >= min_score)

        if start_date:
            stmt = stmt.where(Transaction.transaction_date >= start_date)

        if end_date:
            stmt = stmt.where(Transaction.transaction_date <= end_date)

        # Add pagination and ordering
        stmt = stmt.order_by(Anomaly.created_at.desc()).offset(skip).limit(limit)

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_by_id(self, anomaly_id: UUID4) -> Optional[Anomaly]:
        """Get an anomaly by ID.

        Args:
            anomaly_id: The anomaly ID.

        Returns:
            The anomaly or None if not found.
        """
        stmt = select(Anomaly).where(Anomaly.id == anomaly_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def resolve(
        self,
        anomaly_id: UUID4,
        resolution_notes: str,
        feedback_type: FeedbackType,
    ) -> Optional[Anomaly]:
        """Resolve an anomaly.

        Args:
            anomaly_id: The anomaly ID.
            resolution_notes: Notes about the resolution.
            feedback_type: Type of feedback (true_positive, false_positive).

        Returns:
            Updated anomaly or None if not found.
        """
        db_anomaly = await self.get_by_id(anomaly_id)
        if not db_anomaly:
            return None

        db_anomaly.is_resolved = True
        db_anomaly.resolution_notes = resolution_notes
        db_anomaly.feedback_type = feedback_type
        db_anomaly.resolved_at = datetime.datetime.now()

        await self.db.commit()
        await self.db.refresh(db_anomaly)

        logger.info(f"Resolved anomaly: {anomaly_id} with feedback: {feedback_type}")
        return db_anomaly

    async def get_stats(
        self,
        tenant_id: UUID4,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
    ) -> Dict[str, Any]:
        """Get anomaly statistics.

        Args:
            tenant_id: The tenant ID.
            start_date: Optional start date for filtering.
            end_date: Optional end date for filtering.

        Returns:
            Dictionary of statistics.
        """
        # Build base query for anomalies by this tenant
        base_query = select(Anomaly).where(Anomaly.tenant_id == tenant_id)

        # Add date filters
        if start_date or end_date:
            base_query = base_query.join(
                Transaction, Anomaly.transaction_id == Transaction.id
            )

            if start_date:
                base_query = base_query.where(
                    Transaction.transaction_date >= start_date
                )

            if end_date:
                base_query = base_query.where(Transaction.transaction_date <= end_date)

        # Count total anomalies
        count_stmt = select(func.count()).select_from(base_query.subquery())
        count_result = await self.db.execute(count_stmt)
        total_count = count_result.scalar()

        # Count by anomaly type
        type_stmt = (
            select(Anomaly.anomaly_type, func.count())
            .where(Anomaly.tenant_id == tenant_id)
            .group_by(Anomaly.anomaly_type)
        )
        type_result = await self.db.execute(type_stmt)
        anomaly_types = dict(type_result.all())

        # Count resolved vs. unresolved
        resolved_stmt = (
            select(Anomaly.is_resolved, func.count())
            .where(Anomaly.tenant_id == tenant_id)
            .group_by(Anomaly.is_resolved)
        )
        resolved_result = await self.db.execute(resolved_stmt)
        resolution_status = dict(resolved_result.all())

        # Count by feedback type
        feedback_stmt = (
            select(Anomaly.feedback_type, func.count())
            .where(Anomaly.tenant_id == tenant_id, Anomaly.feedback_type != None)
            .group_by(Anomaly.feedback_type)
        )
        feedback_result = await self.db.execute(feedback_stmt)
        feedback_types = dict(feedback_result.all())

        # Calculate average score
        avg_stmt = select(func.avg(Anomaly.score)).where(Anomaly.tenant_id == tenant_id)
        avg_result = await self.db.execute(avg_stmt)
        avg_score = avg_result.scalar() or 0

        return {
            "total_count": total_count,
            "by_type": anomaly_types,
            "resolution_status": resolution_status,
            "feedback_types": feedback_types,
            "average_score": avg_score,
        }


class ModelAdmin:
    """Admin class for managing ML models."""

    def __init__(self, db_session: AsyncSession):
        """Initialize the model admin.

        Args:
            db_session: Database session.
        """
        self.db = db_session
        self.model_manager = ModelManager(db_session)

    async def get_all_versions(
        self,
        model_type: str,
        skip: int = 0,
        limit: int = 100,
        include_inactive: bool = True,
    ) -> List[ModelVersion]:
        """Get all model versions with pagination.

        Args:
            model_type: The type of model.
            skip: Number of records to skip.
            limit: Maximum number of records to return.
            include_inactive: Whether to include inactive versions.

        Returns:
            List of model versions.
        """
        # Build query
        stmt = select(ModelVersion).where(ModelVersion.model_type == model_type)

        if not include_inactive:
            stmt = stmt.where(ModelVersion.is_active == True)

        # Add pagination and ordering
        stmt = stmt.order_by(ModelVersion.created_at.desc()).offset(skip).limit(limit)

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_active_versions(self) -> List[ModelVersion]:
        """Get all active model versions.

        Returns:
            List of active model versions.
        """
        stmt = select(ModelVersion).where(ModelVersion.is_active == True)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_version_by_id(self, version_id: UUID4) -> Optional[ModelVersion]:
        """Get a model version by ID.

        Args:
            version_id: The model version ID.

        Returns:
            The model version or None if not found.
        """
        stmt = select(ModelVersion).where(ModelVersion.id == version_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def deactivate_version(self, version_id: UUID4) -> Optional[ModelVersion]:
        """Deactivate a model version.

        Args:
            version_id: The model version ID.

        Returns:
            Updated model version or None if not found.
        """
        model_version = await self.get_version_by_id(version_id)
        if not model_version:
            return None

        if not model_version.is_active:
            return model_version  # Already inactive

        model_version.is_active = False
        model_version.deactivated_at = datetime.datetime.now()

        await self.db.commit()
        await self.db.refresh(model_version)

        logger.info(f"Deactivated model version: {version_id}")
        return model_version

    async def activate_version(self, version_id: UUID4) -> Optional[ModelVersion]:
        """Activate a model version (and deactivate other versions of the same type).

        Args:
            version_id: The model version ID.

        Returns:
            Activated model version or None if not found.
        """
        model_version = await self.get_version_by_id(version_id)
        if not model_version:
            return None

        if model_version.is_active:
            return model_version  # Already active

        # Deactivate other versions of the same type
        await self._deactivate_model_type(model_version.model_type)

        # Activate this version
        model_version.is_active = True
        model_version.activated_at = datetime.datetime.now()

        await self.db.commit()
        await self.db.refresh(model_version)

        logger.info(f"Activated model version: {version_id}")
        return model_version

    async def _deactivate_model_type(self, model_type: str) -> None:
        """Deactivate all versions of a model type.

        Args:
            model_type: The model type.
        """
        # Find all active versions of this model type
        stmt = select(ModelVersion).where(
            ModelVersion.model_type == model_type, ModelVersion.is_active == True
        )
        result = await self.db.execute(stmt)
        active_versions = list(result.scalars().all())

        # Deactivate all of them
        for version in active_versions:
            version.is_active = False
            version.deactivated_at = datetime.datetime.now()

        await self.db.commit()

    async def get_performance_history(
        self,
        model_type: str,
        version: Optional[str] = None,
        limit: int = 10,
    ) -> List[ModelPerformance]:
        """Get performance history for a model.

        Args:
            model_type: Type of model.
            version: Optional version identifier.
            limit: Maximum number of records to return.

        Returns:
            List of performance records.
        """
        # Build query
        stmt = select(ModelPerformance).where(ModelPerformance.model_type == model_type)

        if version:
            stmt = stmt.where(ModelPerformance.version == version)

        # Add pagination and ordering
        stmt = stmt.order_by(ModelPerformance.recorded_at.desc()).limit(limit)

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_all_model_types(self) -> List[str]:
        """Get all model types in the system.

        Returns:
            List of unique model types.
        """
        stmt = select(ModelVersion.model_type).distinct()
        result = await self.db.execute(stmt)
        return [row[0] for row in result.all()]

    async def export_model_config(self, version_id: UUID4) -> Dict[str, Any]:
        """Export a model's configuration and metadata.

        Args:
            version_id: The model version ID.

        Returns:
            Dictionary with model configuration.
        """
        model_version = await self.get_version_by_id(version_id)
        if not model_version:
            return {}

        # Get performance metrics
        stmt = (
            select(ModelPerformance)
            .where(
                ModelPerformance.model_type == model_version.model_type,
                ModelPerformance.version == model_version.version,
            )
            .order_by(ModelPerformance.recorded_at.desc())
            .limit(1)
        )
        result = await self.db.execute(stmt)
        latest_performance = result.scalar_one_or_none()

        # Create export configuration
        config = {
            "model_type": model_version.model_type,
            "version": model_version.version,
            "is_active": model_version.is_active,
            "created_at": model_version.created_at.isoformat(),
            "metadata": model_version.metadata,
            "performance_metrics": (
                latest_performance.metrics if latest_performance else None
            ),
            "last_evaluated": (
                latest_performance.recorded_at.isoformat()
                if latest_performance
                else None
            ),
        }

        return config

    async def rename_model_version(
        self, version_id: UUID4, new_version: str
    ) -> Optional[ModelVersion]:
        """Rename a model version.

        Args:
            version_id: The model version ID.
            new_version: New version name.

        Returns:
            Updated model version or None if not found.
        """
        model_version = await self.get_version_by_id(version_id)
        if not model_version:
            return None

        # Check if the new version name is already in use
        stmt = select(ModelVersion).where(
            ModelVersion.model_type == model_version.model_type,
            ModelVersion.version == new_version,
        )
        result = await self.db.execute(stmt)
        existing_version = result.scalar_one_or_none()

        if existing_version and existing_version.id != version_id:
            logger.error(
                f"Cannot rename model version: {new_version} already exists for type {model_version.model_type}"
            )
            return None

        # Rename the version
        old_version = model_version.version
        model_version.version = new_version

        await self.db.commit()
        await self.db.refresh(model_version)

        logger.info(
            f"Renamed model version from {old_version} to {new_version} for {model_version.model_type}"
        )
        return model_version
