"""Machine learning model parameter updates based on feedback.

This module provides functions for updating ML model parameters based on user feedback.
"""

import logging
import os
import uuid
from typing import Dict, Any, Optional

import joblib
from fastapi import HTTPException, status

from auditpulse_mvp.utils.settings import settings

# Configure logging
logger = logging.getLogger(__name__)


async def update_ml_model_parameters(
    model_name: str, params: Dict[str, Any], tenant_id: uuid.UUID
) -> Dict[str, Any]:
    """Update ML model parameters based on feedback.

    This function adjusts parameters like thresholds and regularization
    for a specific model based on user feedback patterns.

    Args:
        model_name: Name of the model to update
        params: Dictionary of parameter adjustments to apply
        tenant_id: Tenant ID

    Returns:
        Dict containing the updated parameters

    Raises:
        HTTPException: If the model cannot be found or updated
    """
    # For the MVP, we'll simulate the update by logging the changes
    # In a real implementation, this would load and update the actual model

    try:
        # First, attempt to find the model file
        model_base_path = settings.ml_models_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "ml_models"
        )

        tenant_models_dir = os.path.join(model_base_path, str(tenant_id))

        if not os.path.exists(tenant_models_dir):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No models directory found for tenant {tenant_id}",
            )

        # Find the most recent model file
        model_files = [
            f
            for f in os.listdir(tenant_models_dir)
            if f.endswith(".joblib") and model_name in f
        ]

        if not model_files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No model files found for model {model_name}",
            )

        # Sort by timestamp (assuming filenames contain timestamps)
        model_files.sort(reverse=True)
        model_path = os.path.join(tenant_models_dir, model_files[0])

        # Load the model
        model_data = joblib.load(model_path)

        # Apply parameter updates
        model = model_data.get("model")
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model data is invalid or corrupted",
            )

        # Update threshold
        if "threshold" in params:
            # In a real implementation, update the model's threshold parameter
            # For isolation forest, this might adjust contamination
            if hasattr(model, "contamination"):
                current = model.contamination
                adjustment = params["threshold"]

                # Ensure contamination stays within valid range (0-0.5)
                new_value = max(0.01, min(0.5, current + adjustment))

                # Log the change
                logger.info(
                    f"Adjusting {model_name} contamination: {current} -> {new_value} "
                    f"(adjustment: {adjustment})"
                )

                # Apply the change
                model.contamination = new_value
            else:
                logger.warning(
                    f"Model {model_name} does not have a contamination parameter"
                )

        # Apply regularization adjustments
        if "regularization" in params:
            # For isolation forest, this might adjust max_samples
            if hasattr(model, "max_samples"):
                reg_adjustment = params["regularization"]

                # Apply the adjustment to max_samples
                if isinstance(model.max_samples, str):
                    # If max_samples is "auto", convert to numeric
                    model.max_samples = 256

                if isinstance(model.max_samples, (int, float)):
                    current = model.max_samples
                    # Increase max_samples slightly to regularize
                    new_value = current + int(current * reg_adjustment)

                    # Log the change
                    logger.info(
                        f"Adjusting {model_name} max_samples: {current} -> {new_value} "
                        f"(adjustment: {reg_adjustment})"
                    )

                    # Apply the change
                    model.max_samples = new_value
            else:
                logger.warning(
                    f"Model {model_name} does not have a max_samples parameter"
                )

        # Save the updated model
        # Generate a new filename with timestamp
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        new_model_path = os.path.join(
            tenant_models_dir, f"{model_name}_updated_{timestamp}.joblib"
        )

        # Save the updated model
        joblib.dump(model_data, new_model_path)

        # Update model info file if it exists
        info_path = model_path.replace(".joblib", "_info.json")
        if os.path.exists(info_path):
            import json

            with open(info_path, "r") as f:
                info_data = json.load(f)

            # Update info with adjustment history
            if "parameter_adjustments" not in info_data:
                info_data["parameter_adjustments"] = []

            info_data["parameter_adjustments"].append(
                {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "params": params,
                    "reason": "feedback_based_learning",
                }
            )

            # Save updated info
            new_info_path = new_model_path.replace(".joblib", "_info.json")
            with open(new_info_path, "w") as f:
                json.dump(info_data, f, indent=2)

        return {
            "model_name": model_name,
            "model_path": new_model_path,
            "parameters_updated": list(params.keys()),
            "timestamp": datetime.datetime.now().isoformat(),
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(f"Error updating ML model parameters: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update model parameters: {str(e)}",
        )
