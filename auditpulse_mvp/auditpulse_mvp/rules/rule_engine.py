"""Rule engine parameter updates based on feedback.

This module provides functions for updating rule thresholds and parameters
based on user feedback.
"""

import logging
import uuid
from typing import Dict, Any, Optional

from fastapi import HTTPException, status
from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text

from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.database.models import TenantConfiguration
from auditpulse_mvp.utils.settings import settings

# Configure logging
logger = logging.getLogger(__name__)


async def update_rule_thresholds(
    rule_name: str,
    adjustment: float,
    tenant_id: uuid.UUID,
    db_session: Optional[AsyncSession] = None,
) -> Dict[str, Any]:
    """Update rule thresholds based on feedback.

    This function adjusts thresholds for a specific rule based on
    user feedback patterns.

    Args:
        rule_name: Name of the rule to update
        adjustment: Adjustment value to apply to the threshold
        tenant_id: Tenant ID
        db_session: Optional database session

    Returns:
        Dict containing the updated threshold information

    Raises:
        HTTPException: If the rule cannot be found or updated
    """
    # If no db_session provided, create one
    if db_session is None:
        db_session = get_db_session()

    try:
        # Check if we have a sensitivity configuration
        config_query = select(TenantConfiguration).where(
            and_(
                TenantConfiguration.tenant_id == tenant_id,
                TenantConfiguration.key == "sensitivity_config",
            )
        )

        result = await db_session.execute(config_query)
        config_record = result.scalar_one_or_none()

        if not config_record or not config_record.value:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No sensitivity configuration found for tenant {tenant_id}",
            )

        # Get the current configuration
        config = config_record.value

        # Map rule_name to configuration key
        rule_config_map = {
            "Large Amount": "large_transaction",
            "Unusual Counterparty": "unusual_counterparty",
            "Weekend Transaction": "weekend_transaction",
            "Irregular Amount": "irregular_amount",
            "Round Number Transaction": "round_number_transaction",
            "Statistical Outlier": "irregular_amount",  # Maps to same as Irregular Amount
        }

        config_key = rule_config_map.get(rule_name)

        if not config_key or "rules" not in config or config_key not in config["rules"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Rule {rule_name} not found in configuration",
            )

        # Get the current rule configuration
        rule_config = config["rules"][config_key]

        # Update threshold based on adjustment
        if "threshold" in rule_config:
            current = rule_config["threshold"]
            # Adjust threshold by percentage
            new_value = current * (1.0 + adjustment)

            # Ensure threshold remains positive
            new_value = max(0.1, new_value)

            # Log change
            logger.info(
                f"Adjusting {rule_name} threshold: {current} -> {new_value} "
                f"(adjustment: {adjustment:+.2%})"
            )

            # Apply change
            rule_config["threshold"] = new_value
        elif "parameters" in rule_config and "threshold" in rule_config["parameters"]:
            current = rule_config["parameters"]["threshold"]
            # Adjust threshold by percentage
            new_value = current * (1.0 + adjustment)

            # Ensure threshold remains positive
            new_value = max(0.1, new_value)

            # Log change
            logger.info(
                f"Adjusting {rule_name} threshold: {current} -> {new_value} "
                f"(adjustment: {adjustment:+.2%})"
            )

            # Apply change
            rule_config["parameters"]["threshold"] = new_value
        elif (
            "parameters" in rule_config
            and "std_dev_threshold" in rule_config["parameters"]
        ):
            # Special case for statistical rules that use standard deviation thresholds
            current = rule_config["parameters"]["std_dev_threshold"]
            # Adjust threshold by percentage
            new_value = current * (1.0 + adjustment)

            # Ensure threshold remains reasonable
            new_value = max(0.5, new_value)

            # Log change
            logger.info(
                f"Adjusting {rule_name} std_dev_threshold: {current} -> {new_value} "
                f"(adjustment: {adjustment:+.2%})"
            )

            # Apply change
            rule_config["parameters"]["std_dev_threshold"] = new_value
        else:
            # If no threshold found, try adjusting other parameters
            adjusted = False

            if "parameters" in rule_config:
                for param_name, param_value in rule_config["parameters"].items():
                    if (
                        isinstance(param_value, (int, float))
                        and "threshold" in param_name.lower()
                    ):
                        # Found a threshold-like parameter
                        current = param_value
                        new_value = current * (1.0 + adjustment)

                        # Log change
                        logger.info(
                            f"Adjusting {rule_name} {param_name}: {current} -> {new_value} "
                            f"(adjustment: {adjustment:+.2%})"
                        )

                        # Apply change
                        rule_config["parameters"][param_name] = new_value
                        adjusted = True
                        break

            if not adjusted:
                # If we couldn't find a threshold to adjust, we might need to disable the rule
                if adjustment > 0.1:  # If a significant positive adjustment was needed
                    logger.info(
                        f"No threshold found to adjust for {rule_name}, disabling rule"
                    )
                    rule_config["enabled"] = False
                    adjusted = True

            if not adjusted:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"No adjustable threshold found for rule {rule_name}",
                )

        # Update the configuration in the database
        stmt = (
            update(TenantConfiguration)
            .where(
                and_(
                    TenantConfiguration.tenant_id == tenant_id,
                    TenantConfiguration.key == "sensitivity_config",
                )
            )
            .values(value=config)
        )

        await db_session.execute(stmt)
        await db_session.commit()

        # Record rule adjustment history
        # In a real implementation, this would be stored in a dedicated table
        # For now, we'll add it to the configuration

        if "rule_adjustments" not in config:
            config["rule_adjustments"] = []

        # Add adjustment record
        import datetime

        adjustment_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "rule_name": rule_name,
            "adjustment": adjustment,
            "config_key": config_key,
            "reason": "feedback_based_learning",
        }

        config["rule_adjustments"].append(adjustment_record)

        # Update again with adjustment history
        stmt = (
            update(TenantConfiguration)
            .where(
                and_(
                    TenantConfiguration.tenant_id == tenant_id,
                    TenantConfiguration.key == "sensitivity_config",
                )
            )
            .values(value=config)
        )

        await db_session.execute(stmt)
        await db_session.commit()

        return {
            "rule_name": rule_name,
            "config_key": config_key,
            "adjustment": adjustment,
            "timestamp": datetime.datetime.now().isoformat(),
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(f"Error updating rule threshold: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update rule threshold: {str(e)}",
        )
