"""Model Manager for ML Engine.

This module provides model versioning, rollback, and performance tracking capabilities.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import uuid

from fastapi import Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.database.models import ModelVersion, ModelPerformance
from auditpulse_mvp.utils.settings import get_settings, Settings

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages ML model versions and performance tracking."""
    
    def __init__(
        self,
        db_session: AsyncSession = Depends(get_db_session),
        settings: Settings = Depends(get_settings),
    ):
        """Initialize the model manager.
        
        Args:
            db_session: Database session
            settings: Application settings
        """
        self.db = db_session
        self.settings = settings
        
    async def create_version(
        self,
        model_type: str,
        model_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        is_active: bool = False,
    ) -> ModelVersion:
        """Create a new model version.
        
        Args:
            model_type: Type of model (e.g., 'anomaly_detection')
            model_data: Serialized model data
            metadata: Additional model metadata
            is_active: Whether this version should be active
            
        Returns:
            ModelVersion: Created model version
        """
        # Deactivate current active version if needed
        if is_active:
            await self._deactivate_current_version(model_type)
            
        # Create new version
        version = ModelVersion(
            model_type=model_type,
            version=await self._get_next_version(model_type),
            model_data=model_data,
            metadata=metadata or {},
            is_active=is_active,
            created_at=datetime.now(),
        )
        
        self.db.add(version)
        await self.db.commit()
        await self.db.refresh(version)
        
        logger.info(f"Created new model version: {version.version} for {model_type}")
        return version
        
    async def get_active_version(self, model_type: str) -> Optional[ModelVersion]:
        """Get the currently active model version.
        
        Args:
            model_type: Type of model
            
        Returns:
            Optional[ModelVersion]: Active model version if exists
        """
        stmt = (
            select(ModelVersion)
            .where(
                ModelVersion.model_type == model_type,
                ModelVersion.is_active == True
            )
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
        
    async def get_version(self, model_type: str, version: str) -> Optional[ModelVersion]:
        """Get a specific model version.
        
        Args:
            model_type: Type of model
            version: Version identifier
            
        Returns:
            Optional[ModelVersion]: Model version if exists
        """
        stmt = (
            select(ModelVersion)
            .where(
                ModelVersion.model_type == model_type,
                ModelVersion.version == version
            )
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
        
    async def list_versions(
        self,
        model_type: str,
        limit: int = 10,
        offset: int = 0,
    ) -> List[ModelVersion]:
        """List model versions.
        
        Args:
            model_type: Type of model
            limit: Maximum number of versions to return
            offset: Number of versions to skip
            
        Returns:
            List[ModelVersion]: List of model versions
        """
        stmt = (
            select(ModelVersion)
            .where(ModelVersion.model_type == model_type)
            .order_by(ModelVersion.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return result.scalars().all()
        
    async def activate_version(self, model_type: str, version: str) -> ModelVersion:
        """Activate a specific model version.
        
        Args:
            model_type: Type of model
            version: Version identifier
            
        Returns:
            ModelVersion: Activated model version
            
        Raises:
            ValueError: If version not found
        """
        # Get version to activate
        version_obj = await self.get_version(model_type, version)
        if not version_obj:
            raise ValueError(f"Model version not found: {model_type} {version}")
            
        # Deactivate current version
        await self._deactivate_current_version(model_type)
        
        # Activate new version
        version_obj.is_active = True
        version_obj.activated_at = datetime.now()
        await self.db.commit()
        await self.db.refresh(version_obj)
        
        logger.info(f"Activated model version: {version} for {model_type}")
        return version_obj
        
    async def rollback_version(self, model_type: str, version: str) -> ModelVersion:
        """Rollback to a previous model version.
        
        Args:
            model_type: Type of model
            version: Version identifier to rollback to
            
        Returns:
            ModelVersion: Rolled back model version
            
        Raises:
            ValueError: If version not found
        """
        return await self.activate_version(model_type, version)
        
    async def record_performance(
        self,
        model_type: str,
        version: str,
        metrics: Dict[str, float],
        dataset_size: int,
        evaluation_time: float,
    ) -> ModelPerformance:
        """Record model performance metrics.
        
        Args:
            model_type: Type of model
            version: Version identifier
            metrics: Performance metrics (e.g., accuracy, f1_score)
            dataset_size: Size of evaluation dataset
            evaluation_time: Time taken for evaluation
            
        Returns:
            ModelPerformance: Recorded performance metrics
        """
        performance = ModelPerformance(
            model_type=model_type,
            version=version,
            metrics=metrics,
            dataset_size=dataset_size,
            evaluation_time=evaluation_time,
            recorded_at=datetime.now(),
        )
        
        self.db.add(performance)
        await self.db.commit()
        await self.db.refresh(performance)
        
        logger.info(f"Recorded performance for model version: {version} of {model_type}")
        return performance
        
    async def get_performance_history(
        self,
        model_type: str,
        version: Optional[str] = None,
        limit: int = 10,
    ) -> List[ModelPerformance]:
        """Get performance history for a model.
        
        Args:
            model_type: Type of model
            version: Optional version identifier
            limit: Maximum number of records to return
            
        Returns:
            List[ModelPerformance]: Performance history
        """
        stmt = (
            select(ModelPerformance)
            .where(ModelPerformance.model_type == model_type)
        )
        
        if version:
            stmt = stmt.where(ModelPerformance.version == version)
            
        stmt = stmt.order_by(ModelPerformance.recorded_at.desc()).limit(limit)
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
        
    async def get_performance_summary(
        self,
        model_type: str,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get performance summary for a model.
        
        Args:
            model_type: Type of model
            version: Optional version identifier
            
        Returns:
            Dict[str, Any]: Performance summary
        """
        stmt = (
            select(
                func.avg(ModelPerformance.evaluation_time).label("avg_evaluation_time"),
                func.avg(ModelPerformance.dataset_size).label("avg_dataset_size"),
                func.count(ModelPerformance.id).label("evaluation_count"),
            )
            .where(ModelPerformance.model_type == model_type)
        )
        
        if version:
            stmt = stmt.where(ModelPerformance.version == version)
            
        result = await self.db.execute(stmt)
        summary = result.first()
        
        # Get latest metrics
        latest = await self.get_performance_history(model_type, version, limit=1)
        latest_metrics = latest[0].metrics if latest else {}
        
        return {
            "avg_evaluation_time": summary.avg_evaluation_time,
            "avg_dataset_size": summary.avg_dataset_size,
            "evaluation_count": summary.evaluation_count,
            "latest_metrics": latest_metrics,
        }
        
    async def _get_next_version(self, model_type: str) -> str:
        """Get the next version number for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            str: Next version number
        """
        stmt = (
            select(func.max(ModelVersion.version))
            .where(ModelVersion.model_type == model_type)
        )
        result = await self.db.execute(stmt)
        current_max = result.scalar_one_or_none()
        
        if not current_max:
            return "1.0.0"
            
        # Increment patch version
        major, minor, patch = map(int, current_max.split("."))
        return f"{major}.{minor}.{patch + 1}"
        
    async def _deactivate_current_version(self, model_type: str) -> None:
        """Deactivate the current active version of a model.
        
        Args:
            model_type: Type of model
        """
        stmt = (
            select(ModelVersion)
            .where(
                ModelVersion.model_type == model_type,
                ModelVersion.is_active == True
            )
        )
        result = await self.db.execute(stmt)
        current = result.scalar_one_or_none()
        
        if current:
            current.is_active = False
            current.deactivated_at = datetime.now()
            await self.db.commit() 