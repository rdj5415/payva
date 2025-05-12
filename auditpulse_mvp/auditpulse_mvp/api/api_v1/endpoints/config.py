"""Configuration API endpoints.

This module provides API endpoints for managing system configuration settings.
"""
import logging
from enum import Enum
from typing import Dict, Optional, List, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Body
from pydantic import BaseModel, Field, validator
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.api.deps import (
    get_current_user, require_admin, get_current_tenant, log_audit_action, AuditAction
)
from auditpulse_mvp.database.models import User, Tenant, TenantConfiguration
from auditpulse_mvp.database.session import get_db
from auditpulse_mvp.utils.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


class SensitivityLevel(str, Enum):
    """Risk sensitivity level for anomaly detection."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CUSTOM = "custom"


class RiskEngineSettings(BaseModel):
    """Settings specific to the risk engine."""
    ml_threshold: float = Field(
        0.7, 
        ge=0.0, 
        le=1.0, 
        description="Threshold for ML model to flag a transaction as anomalous (0.0-1.0)"
    )
    rules_score_weight: float = Field(
        0.6, 
        ge=0.0, 
        le=1.0, 
        description="Weight for rules-based scores in combined risk score calculation (0.0-1.0)"
    )
    ml_score_weight: float = Field(
        0.4, 
        ge=0.0, 
        le=1.0, 
        description="Weight for ML-based scores in combined risk score calculation (0.0-1.0)"
    )
    min_transaction_amount: float = Field(
        100.0, 
        ge=0.0, 
        description="Minimum transaction amount to consider for risk scoring"
    )
    
    @validator('ml_score_weight')
    def weights_must_sum_to_one(cls, v, values):
        """Validate that rules_score_weight and ml_score_weight sum to 1.0."""
        if 'rules_score_weight' in values and abs(values['rules_score_weight'] + v - 1.0) > 0.001:
            raise ValueError('rules_score_weight and ml_score_weight must sum to 1.0')
        return v


class RuleSettings(BaseModel):
    """Settings for individual detection rules."""
    enabled: bool = True
    threshold: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None


class RulesConfiguration(BaseModel):
    """Configuration for all detection rules."""
    large_transaction: RuleSettings = Field(
        default_factory=lambda: RuleSettings(
            enabled=True, 
            threshold=10000.0,
            parameters={"scale_factor": 1.0}
        )
    )
    unusual_counterparty: RuleSettings = Field(
        default_factory=lambda: RuleSettings(
            enabled=True, 
            parameters={"min_frequency": 3}
        )
    )
    weekend_transaction: RuleSettings = Field(
        default_factory=lambda: RuleSettings(
            enabled=True,
            parameters={"score_multiplier": 1.5}
        )
    )
    irregular_amount: RuleSettings = Field(
        default_factory=lambda: RuleSettings(
            enabled=True,
            threshold=2.0,
            parameters={"std_dev_threshold": 2.0}
        )
    )
    round_number_transaction: RuleSettings = Field(
        default_factory=lambda: RuleSettings(
            enabled=True,
            parameters={"score_multiplier": 1.2}
        )
    )


class SensitivityConfiguration(BaseModel):
    """Configuration model for risk sensitivity settings."""
    sensitivity_level: SensitivityLevel
    risk_engine: RiskEngineSettings = Field(default_factory=RiskEngineSettings)
    rules: RulesConfiguration = Field(default_factory=RulesConfiguration)
    custom_settings: Optional[Dict[str, Any]] = None


def get_preset_configuration(level: SensitivityLevel) -> SensitivityConfiguration:
    """Get preset configuration for a sensitivity level.
    
    Args:
        level: Sensitivity level (LOW, MEDIUM, HIGH)
        
    Returns:
        SensitivityConfiguration: Preset configuration for the specified level
    """
    if level == SensitivityLevel.LOW:
        return SensitivityConfiguration(
            sensitivity_level=SensitivityLevel.LOW,
            risk_engine=RiskEngineSettings(
                ml_threshold=0.85,
                rules_score_weight=0.7,
                ml_score_weight=0.3,
                min_transaction_amount=500.0,
            ),
            rules=RulesConfiguration(
                large_transaction=RuleSettings(
                    enabled=True,
                    threshold=20000.0,
                    parameters={"scale_factor": 0.8}
                ),
                unusual_counterparty=RuleSettings(
                    enabled=True,
                    parameters={"min_frequency": 2}
                ),
                weekend_transaction=RuleSettings(
                    enabled=False,
                ),
                irregular_amount=RuleSettings(
                    enabled=True,
                    threshold=3.0,
                    parameters={"std_dev_threshold": 3.0}
                ),
                round_number_transaction=RuleSettings(
                    enabled=False,
                ),
            ),
        )
    elif level == SensitivityLevel.MEDIUM:
        return SensitivityConfiguration(
            sensitivity_level=SensitivityLevel.MEDIUM,
            risk_engine=RiskEngineSettings(
                ml_threshold=0.7,
                rules_score_weight=0.6,
                ml_score_weight=0.4,
                min_transaction_amount=100.0,
            ),
            rules=RulesConfiguration(
                large_transaction=RuleSettings(
                    enabled=True,
                    threshold=10000.0,
                    parameters={"scale_factor": 1.0}
                ),
                unusual_counterparty=RuleSettings(
                    enabled=True,
                    parameters={"min_frequency": 3}
                ),
                weekend_transaction=RuleSettings(
                    enabled=True,
                    parameters={"score_multiplier": 1.5}
                ),
                irregular_amount=RuleSettings(
                    enabled=True,
                    threshold=2.0,
                    parameters={"std_dev_threshold": 2.0}
                ),
                round_number_transaction=RuleSettings(
                    enabled=True,
                    parameters={"score_multiplier": 1.2}
                ),
            ),
        )
    elif level == SensitivityLevel.HIGH:
        return SensitivityConfiguration(
            sensitivity_level=SensitivityLevel.HIGH,
            risk_engine=RiskEngineSettings(
                ml_threshold=0.6,
                rules_score_weight=0.5,
                ml_score_weight=0.5,
                min_transaction_amount=50.0,
            ),
            rules=RulesConfiguration(
                large_transaction=RuleSettings(
                    enabled=True,
                    threshold=5000.0,
                    parameters={"scale_factor": 1.2}
                ),
                unusual_counterparty=RuleSettings(
                    enabled=True,
                    parameters={"min_frequency": 1}
                ),
                weekend_transaction=RuleSettings(
                    enabled=True,
                    parameters={"score_multiplier": 2.0}
                ),
                irregular_amount=RuleSettings(
                    enabled=True,
                    threshold=1.5,
                    parameters={"std_dev_threshold": 1.5}
                ),
                round_number_transaction=RuleSettings(
                    enabled=True,
                    parameters={"score_multiplier": 1.5}
                ),
            ),
        )
    else:
        # Default to medium for any other value
        return get_preset_configuration(SensitivityLevel.MEDIUM)


async def get_tenant_sensitivity_config(
    db: AsyncSession, tenant_id: UUID
) -> SensitivityConfiguration:
    """Get sensitivity configuration for a tenant.
    
    Args:
        db: Database session
        tenant_id: Tenant ID
        
    Returns:
        SensitivityConfiguration: Current sensitivity configuration for the tenant
    """
    # Query tenant configuration
    query = (
        select(TenantConfiguration)
        .where(
            TenantConfiguration.tenant_id == tenant_id,
            TenantConfiguration.key == "sensitivity_config"
        )
    )
    
    result = await db.execute(query)
    config = result.scalar_one_or_none()
    
    # If configuration exists, return it
    if config and config.value:
        try:
            # Use level from database but get preset if it's a standard level
            config_data = config.value
            sensitivity_level = config_data.get("sensitivity_level", SensitivityLevel.MEDIUM)
            
            if sensitivity_level != SensitivityLevel.CUSTOM:
                return get_preset_configuration(sensitivity_level)
            
            # For custom settings, construct the complete configuration
            return SensitivityConfiguration.parse_obj(config_data)
        except Exception as e:
            logger.error(f"Error parsing sensitivity configuration: {e}")
            # Fall back to medium sensitivity
            return get_preset_configuration(SensitivityLevel.MEDIUM)
    
    # Default to medium sensitivity if no configuration exists
    return get_preset_configuration(SensitivityLevel.MEDIUM)


@router.get(
    "/sensitivity",
    response_model=SensitivityConfiguration,
    summary="Get current sensitivity configuration",
)
async def get_sensitivity_config(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    current_tenant: Tenant = Depends(get_current_tenant),
) -> SensitivityConfiguration:
    """Get the current sensitivity configuration for the tenant.
    
    Args:
        db: Database session
        current_user: Current authenticated user
        current_tenant: Current tenant
        
    Returns:
        SensitivityConfiguration: Current sensitivity configuration
    """
    config = await get_tenant_sensitivity_config(db, current_tenant.id)
    
    # Log audit action
    await log_audit_action(
        db=db,
        action=AuditAction(
            action="get_sensitivity_config",
            user_id=current_user.id,
            tenant_id=current_tenant.id,
            resource_type="config",
        ),
    )
    
    return config


@router.put(
    "/sensitivity",
    response_model=SensitivityConfiguration,
    summary="Update sensitivity configuration",
)
async def update_sensitivity_config(
    config: SensitivityConfiguration = Body(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
    current_tenant: Tenant = Depends(get_current_tenant),
) -> SensitivityConfiguration:
    """Update the sensitivity configuration for the tenant.
    
    Args:
        config: New sensitivity configuration
        db: Database session
        current_user: Current authenticated user (must be admin)
        current_tenant: Current tenant
        
    Returns:
        SensitivityConfiguration: Updated sensitivity configuration
    """
    # Handle preset configurations
    if config.sensitivity_level != SensitivityLevel.CUSTOM:
        # Replace with preset configuration but keep the selected level
        config = get_preset_configuration(config.sensitivity_level)
    else:
        # For custom configurations, ensure all properties are present
        config.custom_settings = config.custom_settings or {}
    
    # Check if configuration exists
    query = (
        select(TenantConfiguration)
        .where(
            TenantConfiguration.tenant_id == current_tenant.id,
            TenantConfiguration.key == "sensitivity_config"
        )
    )
    
    result = await db.execute(query)
    existing_config = result.scalar_one_or_none()
    
    # Update or create configuration
    config_dict = config.dict()
    
    if existing_config:
        stmt = (
            update(TenantConfiguration)
            .where(
                TenantConfiguration.tenant_id == current_tenant.id,
                TenantConfiguration.key == "sensitivity_config"
            )
            .values(value=config_dict)
        )
        await db.execute(stmt)
    else:
        # Create new configuration
        new_config = TenantConfiguration(
            tenant_id=current_tenant.id,
            key="sensitivity_config",
            value=config_dict,
        )
        db.add(new_config)
    
    # Commit changes
    await db.commit()
    
    # Log audit action
    await log_audit_action(
        db=db,
        action=AuditAction(
            action="update_sensitivity_config",
            user_id=current_user.id,
            tenant_id=current_tenant.id,
            resource_type="config",
            details={"new_sensitivity_level": config.sensitivity_level},
        ),
    )
    
    return config 