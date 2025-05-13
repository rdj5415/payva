"""API for the Risk Scoring Engine.

This module provides FastAPI endpoints for managing risk scoring configurations,
retrieving risk scoring results, and rescoring anomalies.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.database.models import Anomaly, Tenant
from auditpulse_mvp.risk_engine.risk_engine import (
    RiskEngine,
    RiskWeights,
    RiskSensitivity,
    RiskScoringResult,
    get_risk_engine,
)
from auditpulse_mvp.utils.settings import settings

# Configure logging
logger = logging.getLogger(__name__)

# API router
risk_router = APIRouter(prefix="/risk", tags=["risk"])


# ==================== Request and Response Models ====================


class RiskWeightsRequest(BaseModel):
    """Request model for updating risk weights."""

    ml_weight: float
    rules_weight: float


class RiskSensitivityRequest(BaseModel):
    """Request model for updating risk sensitivity."""

    low_threshold: int
    medium_threshold: int
    high_threshold: int


class RiskConfigResponse(BaseModel):
    """Response model for risk configuration."""

    weights: RiskWeights
    sensitivity: RiskSensitivity
    tenant_id: Optional[str] = None
    updated_at: Optional[datetime] = None


class RiskScoringResponse(BaseModel):
    """Response model for risk scoring results."""

    results: List[RiskScoringResult]
    count: int
    tenant_id: Optional[str] = None
    scored_at: datetime = datetime.now()


class ErrorResponse(BaseModel):
    """Response model for errors."""

    detail: str
    status_code: int


# ==================== API Endpoints ====================


@risk_router.get("/config/{tenant_id}", response_model=RiskConfigResponse)
async def get_risk_config(
    tenant_id: str, db_session: AsyncSession = Depends(get_db_session)
) -> RiskConfigResponse:
    """Get risk configuration for a tenant.

    Args:
        tenant_id: The tenant ID
        db_session: Database session

    Returns:
        RiskConfigResponse: Current risk configuration

    Raises:
        HTTPException: If tenant not found
    """
    # Check if tenant exists
    stmt = select(Tenant).where(Tenant.id == tenant_id)
    result = await db_session.execute(stmt)
    tenant = result.scalar_one_or_none()

    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found",
        )

    # Get tenant's risk configuration from metadata
    tenant_config = tenant.metadata.get("risk_config", {}) if tenant.metadata else {}

    # Extract or create weights
    weights_data = tenant_config.get("weights", {})
    weights = RiskWeights(
        ml_weight=weights_data.get("ml_weight", 0.7),
        rules_weight=weights_data.get("rules_weight", 0.3),
    )

    # Extract or create sensitivity
    sensitivity_data = tenant_config.get("sensitivity", {})
    sensitivity = RiskSensitivity(
        low_threshold=sensitivity_data.get("low_threshold", 30),
        medium_threshold=sensitivity_data.get("medium_threshold", 60),
        high_threshold=sensitivity_data.get("high_threshold", 80),
    )

    # Create response
    return RiskConfigResponse(
        weights=weights,
        sensitivity=sensitivity,
        tenant_id=tenant_id,
        updated_at=(
            datetime.fromisoformat(
                tenant_config.get("updated_at", datetime.now().isoformat())
            )
            if "updated_at" in tenant_config
            else None
        ),
    )


@risk_router.put("/config/{tenant_id}/weights", response_model=RiskConfigResponse)
async def update_risk_weights(
    tenant_id: str,
    weights: RiskWeightsRequest,
    db_session: AsyncSession = Depends(get_db_session),
) -> RiskConfigResponse:
    """Update risk weights for a tenant.

    Args:
        tenant_id: The tenant ID
        weights: New weights configuration
        db_session: Database session

    Returns:
        RiskConfigResponse: Updated risk configuration

    Raises:
        HTTPException: If tenant not found or weights invalid
    """
    # Check if tenant exists
    stmt = select(Tenant).where(Tenant.id == tenant_id)
    result = await db_session.execute(stmt)
    tenant = result.scalar_one_or_none()

    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found",
        )

    try:
        # Validate weights
        risk_weights = RiskWeights(
            ml_weight=weights.ml_weight, rules_weight=weights.rules_weight
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid weights: {str(e)}"
        )

    # Initialize tenant metadata if needed
    if not tenant.metadata:
        tenant.metadata = {}

    # Initialize or get risk config
    if "risk_config" not in tenant.metadata:
        tenant.metadata["risk_config"] = {}

    # Update weights
    if "weights" not in tenant.metadata["risk_config"]:
        tenant.metadata["risk_config"]["weights"] = {}

    tenant.metadata["risk_config"]["weights"]["ml_weight"] = risk_weights.ml_weight
    tenant.metadata["risk_config"]["weights"][
        "rules_weight"
    ] = risk_weights.rules_weight

    # Update timestamp
    tenant.metadata["risk_config"]["updated_at"] = datetime.now().isoformat()

    # Save changes
    await db_session.commit()

    # Get current risk configuration
    return await get_risk_config(tenant_id, db_session)


@risk_router.put("/config/{tenant_id}/sensitivity", response_model=RiskConfigResponse)
async def update_risk_sensitivity(
    tenant_id: str,
    sensitivity: RiskSensitivityRequest,
    db_session: AsyncSession = Depends(get_db_session),
) -> RiskConfigResponse:
    """Update risk sensitivity for a tenant.

    Args:
        tenant_id: The tenant ID
        sensitivity: New sensitivity configuration
        db_session: Database session

    Returns:
        RiskConfigResponse: Updated risk configuration

    Raises:
        HTTPException: If tenant not found or sensitivity invalid
    """
    # Check if tenant exists
    stmt = select(Tenant).where(Tenant.id == tenant_id)
    result = await db_session.execute(stmt)
    tenant = result.scalar_one_or_none()

    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found",
        )

    try:
        # Validate sensitivity
        risk_sensitivity = RiskSensitivity(
            low_threshold=sensitivity.low_threshold,
            medium_threshold=sensitivity.medium_threshold,
            high_threshold=sensitivity.high_threshold,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid sensitivity: {str(e)}",
        )

    # Initialize tenant metadata if needed
    if not tenant.metadata:
        tenant.metadata = {}

    # Initialize or get risk config
    if "risk_config" not in tenant.metadata:
        tenant.metadata["risk_config"] = {}

    # Update sensitivity
    if "sensitivity" not in tenant.metadata["risk_config"]:
        tenant.metadata["risk_config"]["sensitivity"] = {}

    tenant.metadata["risk_config"]["sensitivity"][
        "low_threshold"
    ] = risk_sensitivity.low_threshold
    tenant.metadata["risk_config"]["sensitivity"][
        "medium_threshold"
    ] = risk_sensitivity.medium_threshold
    tenant.metadata["risk_config"]["sensitivity"][
        "high_threshold"
    ] = risk_sensitivity.high_threshold

    # Update timestamp
    tenant.metadata["risk_config"]["updated_at"] = datetime.now().isoformat()

    # Save changes
    await db_session.commit()

    # Get current risk configuration
    return await get_risk_config(tenant_id, db_session)


@risk_router.post("/anomaly/{anomaly_id}/rescore", response_model=RiskScoringResult)
async def rescore_anomaly(
    anomaly_id: int = Path(..., description="The anomaly ID to rescore"),
    generate_explanation: bool = Query(
        False, description="Whether to generate an explanation"
    ),
    db_session: AsyncSession = Depends(get_db_session),
    risk_engine: RiskEngine = Depends(get_risk_engine),
) -> RiskScoringResult:
    """Rescore a single anomaly.

    Args:
        anomaly_id: The anomaly ID
        generate_explanation: Whether to generate an explanation
        db_session: Database session
        risk_engine: Risk engine instance

    Returns:
        RiskScoringResult: Updated risk scoring result

    Raises:
        HTTPException: If anomaly not found
    """
    # Get the anomaly
    anomaly = await db_session.get(Anomaly, anomaly_id)

    if not anomaly:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Anomaly {anomaly_id} not found",
        )

    # Convert to dict for scoring
    anomaly_dict = {
        "id": anomaly.id,
        "ml_score": anomaly.ml_score or 0.0,
        "rule_score": anomaly.rule_score or 0,
        "transaction_amount": (
            float(anomaly.transaction.amount) if anomaly.transaction else 0.0
        ),
        "transaction_type": (
            anomaly.transaction.transaction_type if anomaly.transaction else "unknown"
        ),
        "merchant_name": (
            anomaly.transaction.merchant_name if anomaly.transaction else "unknown"
        ),
        "transaction_date": (
            anomaly.transaction.transaction_date.isoformat()
            if anomaly.transaction
            else datetime.now().isoformat()
        ),
        "generate_explanation": generate_explanation,
    }

    # Get tenant configuration for the anomaly's tenant
    tenant_id = str(anomaly.tenant_id)
    config_response = await get_risk_config(tenant_id, db_session)

    # Update risk engine with tenant's configuration
    risk_engine.update_weights(config_response.weights)
    risk_engine.update_sensitivity(config_response.sensitivity)

    # Score the anomaly
    result = risk_engine.score_anomaly(anomaly_dict)

    # Update the anomaly record with new risk score
    anomaly.risk_score = result.risk_score
    anomaly.risk_level = result.risk_level

    if not anomaly.detection_metadata:
        anomaly.detection_metadata = {}

    anomaly.detection_metadata["risk_factors"] = result.risk_factors
    anomaly.detection_metadata["rescored_at"] = datetime.now().isoformat()

    if result.explanation:
        anomaly.explanation = result.explanation
        anomaly.detection_metadata["explanation_generated"] = True

    # Save changes
    await db_session.commit()

    # Set updated timestamp in response
    result.updated_at = datetime.now().isoformat()

    return result


@risk_router.post("/tenant/{tenant_id}/rescore", response_model=RiskScoringResponse)
async def rescore_tenant_anomalies(
    tenant_id: str,
    days: int = Query(
        settings.risk_rescoring_days,
        description="Number of days of anomalies to rescore",
    ),
    limit: int = Query(100, description="Maximum number of anomalies to rescore"),
    generate_explanations: bool = Query(
        False, description="Whether to generate explanations"
    ),
    db_session: AsyncSession = Depends(get_db_session),
    risk_engine: RiskEngine = Depends(get_risk_engine),
) -> RiskScoringResponse:
    """Rescore anomalies for a tenant.

    Args:
        tenant_id: The tenant ID
        days: Number of days of anomalies to rescore
        limit: Maximum number of anomalies to rescore
        generate_explanations: Whether to generate explanations
        db_session: Database session
        risk_engine: Risk engine instance

    Returns:
        RiskScoringResponse: Summary of rescoring results

    Raises:
        HTTPException: If tenant not found
    """
    # Check if tenant exists
    stmt = select(Tenant).where(Tenant.id == tenant_id)
    result = await db_session.execute(stmt)
    tenant = result.scalar_one_or_none()

    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found",
        )

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Query for anomalies in the date range
    stmt = (
        select(Anomaly)
        .where(
            Anomaly.tenant_id == tenant_id,
            Anomaly.created_at >= start_date,
            Anomaly.created_at <= end_date,
        )
        .order_by(Anomaly.created_at.desc())
        .limit(limit)
    )

    result = await db_session.execute(stmt)
    anomalies = result.scalars().all()

    # Get tenant configuration
    config_response = await get_risk_config(tenant_id, db_session)

    # Update risk engine with tenant's configuration
    risk_engine.update_weights(config_response.weights)
    risk_engine.update_sensitivity(config_response.sensitivity)

    # Prepare anomalies for batch scoring
    anomaly_dicts = []
    for anomaly in anomalies:
        anomaly_dict = {
            "id": anomaly.id,
            "ml_score": anomaly.ml_score or 0.0,
            "rule_score": anomaly.rule_score or 0,
            "transaction_amount": (
                float(anomaly.transaction.amount) if anomaly.transaction else 0.0
            ),
            "transaction_type": (
                anomaly.transaction.transaction_type
                if anomaly.transaction
                else "unknown"
            ),
            "merchant_name": (
                anomaly.transaction.merchant_name if anomaly.transaction else "unknown"
            ),
            "transaction_date": (
                anomaly.transaction.transaction_date.isoformat()
                if anomaly.transaction
                else datetime.now().isoformat()
            ),
            "generate_explanation": generate_explanations,
        }
        anomaly_dicts.append(anomaly_dict)

    # Score anomalies in batch
    results = risk_engine.batch_score_anomalies(anomaly_dicts)

    # Update anomaly records with new risk scores
    for anomaly, result in zip(anomalies, results):
        anomaly.risk_score = result.risk_score
        anomaly.risk_level = result.risk_level

        if not anomaly.detection_metadata:
            anomaly.detection_metadata = {}

        anomaly.detection_metadata["risk_factors"] = result.risk_factors
        anomaly.detection_metadata["rescored_at"] = datetime.now().isoformat()

        if result.explanation:
            anomaly.explanation = result.explanation
            anomaly.detection_metadata["explanation_generated"] = True

    # Save changes
    await db_session.commit()

    # Create response
    return RiskScoringResponse(
        results=results,
        count=len(results),
        tenant_id=tenant_id,
        scored_at=datetime.now(),
    )
