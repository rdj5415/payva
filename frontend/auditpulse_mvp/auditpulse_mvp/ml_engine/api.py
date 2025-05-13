"""API endpoints for the ML Engine.

This module provides API endpoints for interacting with the ML Engine,
including model training, inference, and management.
"""

import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.database.models import Tenant, Transaction
from auditpulse_mvp.ml_engine.ml_engine import MLEngine
from auditpulse_mvp.ml_engine.scheduler import get_ml_scheduler, MLScheduler

# Create router
router = APIRouter(
    prefix="/ml",
    tags=["ml"],
    responses={404: {"description": "Not found"}},
)


# Pydantic models for API requests and responses
class ModelParameters(BaseModel):
    """Model parameters for training."""
    
    n_estimators: Optional[int] = Field(
        None, description="Number of estimators in the ensemble"
    )
    max_samples: Optional[str] = Field(
        None, description="Number of samples to draw for each estimator"
    )
    contamination: Optional[float] = Field(
        None, description="Expected proportion of outliers in the dataset"
    )
    random_state: Optional[int] = Field(
        None, description="Random seed for reproducibility"
    )


class TrainingResponse(BaseModel):
    """Response from model training."""
    
    model_path: str = Field(..., description="Path to the trained model file")
    training_samples: int = Field(..., description="Number of samples used for training")
    anomaly_rate: float = Field(..., description="Proportion of anomalies detected in training data")
    anomaly_count: int = Field(..., description="Number of anomalies detected in training data")


class ModelInfo(BaseModel):
    """Information about a trained model."""
    
    tenant_id: str = Field(..., description="Tenant ID")
    created_at: str = Field(..., description="Creation timestamp")
    data_start_date: str = Field(..., description="Start date of training data")
    data_end_date: str = Field(..., description="End date of training data")
    training_samples: int = Field(..., description="Number of samples used for training")
    anomaly_rate: float = Field(..., description="Proportion of anomalies in training data")
    model_version: str = Field(..., description="Model version identifier")
    filename: Optional[str] = Field(None, description="Model filename")
    file_size: Optional[int] = Field(None, description="Model file size in bytes")
    last_modified: Optional[str] = Field(None, description="Last modified timestamp")


class TransactionScore(BaseModel):
    """Score for a transaction."""
    
    transaction_id: uuid.UUID = Field(..., description="Transaction ID")
    score: float = Field(..., description="Anomaly score (0-1, higher is more anomalous)")


class RetrainingStatus(BaseModel):
    """Status of a retraining operation."""
    
    tenant_count: int = Field(..., description="Number of tenants processed")
    success_count: int = Field(..., description="Number of successful retrainings")
    error_count: int = Field(..., description="Number of failed retrainings")
    skipped_count: int = Field(..., description="Number of skipped retrainings")


@router.post(
    "/train/{tenant_id}",
    response_model=TrainingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Train a new model for a tenant",
)
async def train_model(
    tenant_id: uuid.UUID = Path(..., description="The tenant ID"),
    parameters: Optional[ModelParameters] = None,
    db_session: AsyncSession = Depends(get_db_session),
):
    """Train a new Isolation Forest model for anomaly detection.
    
    This endpoint trains a new model for the specified tenant using
    their historical transaction data. The model is then saved and
    can be used for inference.
    
    Args:
        tenant_id: The tenant ID.
        parameters: Optional model parameters to customize the training.
        db_session: Database session for data access.
        
    Returns:
        Information about the trained model.
    """
    ml_engine = MLEngine(db_session=db_session)
    
    result = await ml_engine.train_model(
        tenant_id=tenant_id,
        model_params=parameters.dict(exclude_none=True) if parameters else None,
    )
    
    return TrainingResponse(
        model_path=result["model_path"],
        training_samples=result["training_samples"],
        anomaly_rate=result["anomaly_rate"],
        anomaly_count=result["anomaly_count"],
    )


@router.get(
    "/models/{tenant_id}",
    response_model=List[ModelInfo],
    summary="Get all models for a tenant",
)
async def get_models(
    tenant_id: uuid.UUID = Path(..., description="The tenant ID"),
    db_session: AsyncSession = Depends(get_db_session),
):
    """Get information about all trained models for a tenant.
    
    This endpoint returns a list of all models trained for the specified
    tenant, including metadata about each model.
    
    Args:
        tenant_id: The tenant ID.
        db_session: Database session for data access.
        
    Returns:
        List of model information objects.
    """
    ml_engine = MLEngine(db_session=db_session)
    models = await ml_engine.get_tenant_models(tenant_id)
    return models


@router.delete(
    "/models/{tenant_id}/{model_filename}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a model",
)
async def delete_model(
    tenant_id: uuid.UUID = Path(..., description="The tenant ID"),
    model_filename: str = Path(..., description="The model filename"),
    db_session: AsyncSession = Depends(get_db_session),
):
    """Delete a specific model.
    
    This endpoint deletes a model and its associated files. This
    operation cannot be undone.
    
    Args:
        tenant_id: The tenant ID.
        model_filename: The filename of the model to delete.
        db_session: Database session for data access.
    """
    ml_engine = MLEngine(db_session=db_session)
    success = await ml_engine.delete_model(tenant_id, model_filename)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_filename} not found for tenant {tenant_id}",
        )


@router.post(
    "/score/transactions/{tenant_id}",
    response_model=List[TransactionScore],
    summary="Score multiple transactions",
)
async def score_transactions(
    tenant_id: uuid.UUID = Path(..., description="The tenant ID"),
    transaction_ids: List[uuid.UUID] = Query(..., description="List of transaction IDs to score"),
    db_session: AsyncSession = Depends(get_db_session),
):
    """Score multiple transactions for anomalies.
    
    This endpoint scores the specified transactions using the latest
    trained model for the tenant. Higher scores indicate more anomalous
    transactions.
    
    Args:
        tenant_id: The tenant ID.
        transaction_ids: List of transaction IDs to score.
        db_session: Database session for data access.
        
    Returns:
        List of transaction scores.
    """
    if not transaction_ids:
        return []
    
    # Get the transactions from the database
    transactions = []
    for txn_id in transaction_ids:
        txn = await db_session.get(Transaction, txn_id)
        if txn and txn.tenant_id == tenant_id and not txn.is_deleted:
            transactions.append(txn)
    
    if not transactions:
        return []
    
    # Score the transactions
    ml_engine = MLEngine(db_session=db_session)
    scores = await ml_engine.batch_score_transactions(tenant_id, transactions)
    
    # Format the response
    return [
        TransactionScore(transaction_id=txn_id, score=score)
        for txn_id, score in scores.items()
    ]


@router.post(
    "/retrain/all",
    response_model=RetrainingStatus,
    summary="Retrain models for all tenants",
)
async def retrain_all_models(
    db_session: AsyncSession = Depends(get_db_session),
    scheduler: MLScheduler = Depends(get_ml_scheduler),
):
    """Retrain models for all tenants immediately.
    
    This endpoint triggers an immediate retraining of models for all
    active tenants. This is useful for manual retraining outside of
    the scheduled nightly retraining.
    
    Args:
        db_session: Database session for data access.
        scheduler: ML scheduler for retraining.
        
    Returns:
        Status of the retraining operation.
    """
    results = await scheduler.run_immediate_retraining()
    
    # Count results by status
    success_count = sum(1 for r in results.values() if r.get("status") == "success")
    error_count = sum(1 for r in results.values() if r.get("status") == "error")
    skipped_count = sum(1 for r in results.values() if r.get("status") == "skipped")
    
    return RetrainingStatus(
        tenant_count=len(results),
        success_count=success_count,
        error_count=error_count,
        skipped_count=skipped_count,
    ) 