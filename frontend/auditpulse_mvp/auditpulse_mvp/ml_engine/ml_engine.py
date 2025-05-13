"""Machine Learning Engine for AuditPulse MVP.

This module implements an Isolation Forest model for machine learning-based
anomaly detection in financial transactions.
"""

import datetime
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, HTTPException, status
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.database.models import AnomalyType, DataSource, Tenant, Transaction
from auditpulse_mvp.utils.settings import settings

# Configure logging
logger = logging.getLogger(__name__)

# Define the features to use for anomaly detection
NUMERICAL_FEATURES = ["amount", "day_of_week", "hour_of_day", "month", "day_of_month"]
CATEGORICAL_FEATURES = ["category", "merchant_name", "source"]

# Default model parameters
DEFAULT_MODEL_PARAMS = {
    "n_estimators": 100,
    "max_samples": "auto",
    "contamination": 0.05,  # Assuming ~5% of transactions are anomalous
    "random_state": 42,
    "n_jobs": -1,  # Use all available processors
}


class MLEngine:
    """Machine Learning Engine for anomaly detection."""

    def __init__(self, db_session: AsyncSession = Depends(get_db_session)):
        """Initialize the ML Engine.

        Args:
            db_session: Database session for data access.
        """
        self.db_session = db_session
        self._ensure_model_directory()

    def _ensure_model_directory(self) -> None:
        """Create the model directory if it doesn't exist."""
        models_dir = self._get_base_model_directory()
        os.makedirs(models_dir, exist_ok=True)

    def _get_base_model_directory(self) -> str:
        """Get the base directory for storing models.

        Returns:
            str: Path to the base model directory.
        """
        # Use a models directory relative to the current working directory
        return os.path.join(os.getcwd(), "models")

    def _get_tenant_model_directory(self, tenant_id: uuid.UUID) -> str:
        """Get the model directory for a specific tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            str: Path to the tenant's model directory.
        """
        tenant_dir = os.path.join(self._get_base_model_directory(), str(tenant_id))
        os.makedirs(tenant_dir, exist_ok=True)
        return tenant_dir

    def _get_latest_model_path(self, tenant_id: uuid.UUID) -> Optional[str]:
        """Get the path to the latest model for a tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            Optional[str]: Path to the latest model, or None if no model exists.
        """
        tenant_dir = self._get_tenant_model_directory(tenant_id)

        # Find the latest model based on timestamp in filename
        model_files = list(Path(tenant_dir).glob("*.pkl"))
        if not model_files:
            return None

        # Sort by modified timestamp (newest first)
        latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
        return str(latest_model)

    def _get_model_info_path(self, model_path: str) -> str:
        """Get the path to the model info file.

        Args:
            model_path: Path to the model file.

        Returns:
            str: Path to the model info file.
        """
        return model_path.replace(".pkl", "_info.json")

    def _generate_model_path(self, tenant_id: uuid.UUID) -> str:
        """Generate a path for saving a new model.

        Args:
            tenant_id: The tenant ID.

        Returns:
            str: Path for the new model.
        """
        tenant_dir = self._get_tenant_model_directory(tenant_id)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(tenant_dir, f"isolation_forest_{timestamp}.pkl")

    async def _get_training_data(
        self, tenant_id: uuid.UUID, days: int = 365
    ) -> pd.DataFrame:
        """Get training data for a tenant.

        Args:
            tenant_id: The tenant ID.
            days: Number of days of transaction history to include.

        Returns:
            pd.DataFrame: DataFrame containing transaction features.

        Raises:
            HTTPException: If there's not enough data for training.
        """
        # Calculate the date range
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)

        # Query for transactions in the date range
        stmt = (
            select(Transaction)
            .where(
                and_(
                    Transaction.tenant_id == tenant_id,
                    Transaction.transaction_date >= start_date,
                    Transaction.transaction_date <= end_date,
                    Transaction.is_deleted == False,
                )
            )
            .order_by(Transaction.transaction_date.asc())
        )

        result = await self.db_session.execute(stmt)
        transactions = result.scalars().all()

        if len(transactions) < 100:  # Minimum sample size for meaningful training
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Not enough transaction data for ML training. Need at least 100 transactions, found {len(transactions)}.",
            )

        # Convert transactions to DataFrame and extract features
        return self._prepare_features(transactions)

    def _prepare_features(self, transactions: List[Transaction]) -> pd.DataFrame:
        """Extract features from transactions for ML processing.

        Args:
            transactions: List of transactions.

        Returns:
            pd.DataFrame: DataFrame with extracted features.
        """
        # Extract raw data
        data = []
        for txn in transactions:
            # Extract date-based features
            txn_date = txn.transaction_date

            # Create a row for each transaction
            row = {
                "transaction_id": txn.id,
                "amount": float(txn.amount),
                "source": txn.source.value if txn.source else "unknown",
                "merchant_name": txn.merchant_name or "unknown",
                "category": txn.category or "unknown",
                "day_of_week": txn_date.weekday(),
                "hour_of_day": txn_date.hour,
                "month": txn_date.month,
                "day_of_month": txn_date.day,
                "transaction_date": txn_date,
            }
            data.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Set transaction_id as index for easier identification later
        if not df.empty:
            df.set_index("transaction_id", inplace=True)

        return df

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using one-hot encoding.

        Args:
            df: Input DataFrame with categorical columns.

        Returns:
            pd.DataFrame: DataFrame with encoded categorical features.
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()

        # One-hot encode categorical features
        for feature in CATEGORICAL_FEATURES:
            if feature in result_df.columns:
                # Get most frequent categories (top 20) to avoid too many columns
                top_categories = result_df[feature].value_counts().nlargest(20).index

                # Create a column for each top category
                for category in top_categories:
                    col_name = f"{feature}_{category}"
                    result_df[col_name] = (result_df[feature] == category).astype(int)

                # Drop the original categorical column
                result_df = result_df.drop(columns=[feature])

        return result_df

    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
        """Preprocess data for model training or inference.

        Args:
            df: Input DataFrame with raw features.

        Returns:
            Tuple[pd.DataFrame, StandardScaler]: Preprocessed DataFrame and the scaler.
        """
        # Make a copy of the dataframe to avoid modifying the original
        processed_df = df.copy()

        # Drop non-feature columns
        date_col = "transaction_date"
        if date_col in processed_df.columns:
            processed_df = processed_df.drop(columns=[date_col])

        # Encode categorical features
        processed_df = self._encode_categorical_features(processed_df)

        # Replace NaN values
        processed_df = processed_df.fillna(0)

        # Normalize numerical features
        scaler = StandardScaler()
        numeric_cols = [
            col for col in processed_df.columns if col in NUMERICAL_FEATURES
        ]

        if numeric_cols:
            processed_df[numeric_cols] = scaler.fit_transform(
                processed_df[numeric_cols]
            )

        return processed_df, scaler

    async def train_model(
        self, tenant_id: uuid.UUID, model_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Train a new model for a tenant.

        Args:
            tenant_id: The tenant ID.
            model_params: Optional parameters for the IsolationForest model.

        Returns:
            Dict[str, Any]: Information about the trained model.

        Raises:
            HTTPException: If there's not enough data or a training error occurs.
        """
        try:
            # Get training data
            df = await self._get_training_data(tenant_id)

            # Check if we have enough data
            if len(df) < 100:  # Minimum requirement for reliable training
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Not enough transaction data for ML training. Need at least 100 transactions, found {len(df)}.",
                )

            # Preprocess data
            processed_df, scaler = self._preprocess_data(df)

            # Set model parameters
            params = DEFAULT_MODEL_PARAMS.copy()
            if model_params:
                params.update(model_params)

            # Train the model
            model = IsolationForest(**params)
            model.fit(processed_df)

            # Calculate anomaly scores for training data
            anomaly_scores = model.decision_function(processed_df)
            predictions = model.predict(processed_df)

            # Generate statistics for the model
            anomaly_count = sum(1 for p in predictions if p == -1)
            anomaly_rate = anomaly_count / len(predictions) if predictions else 0

            # Generate a path for the new model
            model_path = self._generate_model_path(tenant_id)

            # Save the model and scaler
            model_data = {
                "model": model,
                "scaler": scaler,
                "feature_names": processed_df.columns.tolist(),
            }
            joblib.dump(model_data, model_path)

            # Save model info
            model_info = {
                "tenant_id": str(tenant_id),
                "created_at": datetime.datetime.now().isoformat(),
                "data_start_date": df["transaction_date"].min().isoformat(),
                "data_end_date": df["transaction_date"].max().isoformat(),
                "training_samples": len(df),
                "parameters": params,
                "anomaly_rate": anomaly_rate,
                "feature_importance": self._calculate_feature_importance(
                    model, processed_df.columns
                ),
                "model_version": "isolation_forest_1.0",
            }

            # Save model info as JSON
            info_path = self._get_model_info_path(model_path)
            with open(info_path, "w") as f:
                json.dump(model_info, f, indent=2)

            return {
                "model_path": model_path,
                "info_path": info_path,
                "training_samples": len(df),
                "anomaly_rate": anomaly_rate,
                "anomaly_count": anomaly_count,
                "model_info": model_info,
            }
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.exception(f"Error training model for tenant {tenant_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error training model: {str(e)}",
            )

    def _calculate_feature_importance(
        self, model: IsolationForest, feature_names: List[str]
    ) -> Dict[str, float]:
        """Calculate feature importance from the Isolation Forest model.

        Args:
            model: Trained IsolationForest model.
            feature_names: List of feature names.

        Returns:
            Dict[str, float]: Mapping of feature names to importance scores.
        """
        # Get feature importances from the model
        importances = model.feature_importances_

        # Create a dictionary of feature name to importance
        importance_dict = dict(zip(feature_names, importances))

        # Sort by importance (descending)
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    async def load_model(self, tenant_id: uuid.UUID) -> Dict[str, Any]:
        """Load the latest model for a tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            Dict[str, Any]: Dictionary containing the model and related data.

        Raises:
            HTTPException: If no model exists or there's an error loading it.
        """
        try:
            # Get the path to the latest model
            model_path = self._get_latest_model_path(tenant_id)
            if not model_path:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No trained model found for tenant {tenant_id}",
                )

            # Load the model data
            model_data = joblib.load(model_path)

            # Load model info
            info_path = self._get_model_info_path(model_path)
            model_info = {}
            if os.path.exists(info_path):
                with open(info_path, "r") as f:
                    model_info = json.load(f)

            return {
                "model": model_data["model"],
                "scaler": model_data["scaler"],
                "feature_names": model_data["feature_names"],
                "model_path": model_path,
                "model_info": model_info,
            }
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.exception(f"Error loading model for tenant {tenant_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error loading model: {str(e)}",
            )

    async def score_transaction(
        self, tenant_id: uuid.UUID, transaction: Transaction
    ) -> float:
        """Score a transaction using the trained model.

        Args:
            tenant_id: The tenant ID.
            transaction: The transaction to score.

        Returns:
            float: Anomaly score between 0 and 1, where higher means more anomalous.

        Raises:
            HTTPException: If there's an error scoring the transaction.
        """
        try:
            # Load the latest model
            model_data = await self.load_model(tenant_id)
            model = model_data["model"]
            scaler = model_data["scaler"]
            feature_names = model_data["feature_names"]

            # Convert transaction to DataFrame and extract features
            txn_df = self._prepare_features([transaction])

            # Preprocess the transaction data
            processed_df = self._encode_categorical_features(txn_df)

            # Handle missing columns - ensure all expected features are present
            for feature in feature_names:
                if feature not in processed_df.columns:
                    processed_df[feature] = 0

            # Keep only the columns that the model expects
            processed_df = processed_df[feature_names]

            # Scale numerical features
            numeric_cols = [
                col for col in processed_df.columns if col in NUMERICAL_FEATURES
            ]
            if numeric_cols:
                processed_df[numeric_cols] = scaler.transform(
                    processed_df[numeric_cols]
                )

            # Get the raw anomaly score from the model
            # The decision_function returns the raw score (distance from normal)
            raw_score = model.decision_function(processed_df)[0]

            # Convert to an anomaly score between 0 and 1
            # Lower raw scores are more anomalous, so we need to invert
            anomaly_score = 1.0 - (raw_score + 0.5)  # Normalize to 0-1 range

            # Clip to ensure it's between 0 and 1
            return max(0.0, min(1.0, anomaly_score))
        except HTTPException:
            # If no model exists, return a neutral score
            if "404" in str(e):
                logger.warning(
                    f"No model found for tenant {tenant_id}, returning neutral score"
                )
                return 0.5
            # Re-raise other HTTP exceptions
            raise
        except Exception as e:
            logger.exception(f"Error scoring transaction for tenant {tenant_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error scoring transaction: {str(e)}",
            )

    async def batch_score_transactions(
        self, tenant_id: uuid.UUID, transactions: List[Transaction]
    ) -> Dict[uuid.UUID, float]:
        """Score multiple transactions in batch.

        Args:
            tenant_id: The tenant ID.
            transactions: List of transactions to score.

        Returns:
            Dict[uuid.UUID, float]: Mapping of transaction IDs to anomaly scores.

        Raises:
            HTTPException: If there's an error scoring the transactions.
        """
        try:
            if not transactions:
                return {}

            # Load the latest model
            model_data = await self.load_model(tenant_id)
            model = model_data["model"]
            scaler = model_data["scaler"]
            feature_names = model_data["feature_names"]

            # Convert transactions to DataFrame and extract features
            txns_df = self._prepare_features(transactions)

            # Preprocess the transaction data
            processed_df = self._encode_categorical_features(txns_df)

            # Handle missing columns - ensure all expected features are present
            for feature in feature_names:
                if feature not in processed_df.columns:
                    processed_df[feature] = 0

            # Keep only the columns that the model expects
            processed_df = processed_df[feature_names]

            # Scale numerical features
            numeric_cols = [
                col for col in processed_df.columns if col in NUMERICAL_FEATURES
            ]
            if numeric_cols:
                processed_df[numeric_cols] = scaler.transform(
                    processed_df[numeric_cols]
                )

            # Get the raw anomaly scores from the model
            raw_scores = model.decision_function(processed_df)

            # Convert to anomaly scores between 0 and 1
            anomaly_scores = [1.0 - (raw_score + 0.5) for raw_score in raw_scores]
            anomaly_scores = [max(0.0, min(1.0, score)) for score in anomaly_scores]

            # Create a dictionary mapping transaction IDs to scores
            return dict(zip([txn.id for txn in transactions], anomaly_scores))
        except HTTPException as e:
            # If no model exists, return neutral scores for all transactions
            if "404" in str(e):
                logger.warning(
                    f"No model found for tenant {tenant_id}, returning neutral scores"
                )
                return {txn.id: 0.5 for txn in transactions}
            # Re-raise other HTTP exceptions
            raise
        except Exception as e:
            logger.exception(
                f"Error batch scoring transactions for tenant {tenant_id}: {e}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error batch scoring transactions: {str(e)}",
            )

    async def get_tenant_models(self, tenant_id: uuid.UUID) -> List[Dict[str, Any]]:
        """Get a list of all models for a tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            List[Dict[str, Any]]: List of model information dictionaries.
        """
        tenant_dir = self._get_tenant_model_directory(tenant_id)
        models = []

        try:
            # Find all model info files
            info_files = list(Path(tenant_dir).glob("*_info.json"))

            for info_file in info_files:
                try:
                    with open(info_file, "r") as f:
                        model_info = json.load(f)

                    # Add file information
                    model_path = str(info_file).replace("_info.json", ".pkl")
                    if os.path.exists(model_path):
                        model_stat = os.stat(model_path)

                        model_info["file_size"] = model_stat.st_size
                        model_info["last_modified"] = datetime.datetime.fromtimestamp(
                            model_stat.st_mtime
                        ).isoformat()
                        model_info["filename"] = os.path.basename(model_path)

                        models.append(model_info)
                except Exception as e:
                    logger.warning(f"Error loading model info {info_file}: {e}")
                    continue

            # Sort by creation date (newest first)
            models.sort(key=lambda x: x.get("created_at", ""), reverse=True)

            return models
        except Exception as e:
            logger.exception(f"Error getting models for tenant {tenant_id}: {e}")
            return []

    async def delete_model(self, tenant_id: uuid.UUID, model_filename: str) -> bool:
        """Delete a specific model.

        Args:
            tenant_id: The tenant ID.
            model_filename: The filename of the model to delete.

        Returns:
            bool: True if successful, False otherwise.
        """
        tenant_dir = self._get_tenant_model_directory(tenant_id)
        model_path = os.path.join(tenant_dir, model_filename)
        info_path = self._get_model_info_path(model_path)

        try:
            # Check if files exist before deleting
            model_exists = os.path.exists(model_path)
            info_exists = os.path.exists(info_path)

            # Delete the files if they exist
            if model_exists:
                os.remove(model_path)

            if info_exists:
                os.remove(info_path)

            return model_exists or info_exists
        except Exception as e:
            logger.exception(
                f"Error deleting model {model_filename} for tenant {tenant_id}: {e}"
            )
            return False

    async def retrain_all_tenant_models(self) -> Dict[uuid.UUID, Dict[str, Any]]:
        """Retrain models for all tenants.

        This is used for scheduled nightly retraining.

        Returns:
            Dict[uuid.UUID, Dict[str, Any]]: Mapping of tenant IDs to training results.
        """
        # Query for all active tenants
        try:
            stmt = select(Tenant).where(Tenant.is_active == True)
            result = await self.db_session.execute(stmt)
            tenants = result.scalars().all()

            results = {}
            for tenant in tenants:
                try:
                    # Skip tenants that have ML engine disabled
                    if not tenant.enable_ml_engine:
                        logger.info(
                            f"ML engine disabled for tenant {tenant.id}, skipping retraining"
                        )
                        results[tenant.id] = {
                            "status": "skipped",
                            "reason": "ML engine disabled",
                        }
                        continue

                    # Retrain the model
                    training_result = await self.train_model(tenant.id)
                    results[tenant.id] = {
                        "status": "success",
                        "model_path": training_result["model_path"],
                        "training_samples": training_result["training_samples"],
                        "anomaly_rate": training_result["anomaly_rate"],
                    }
                    logger.info(f"Successfully retrained model for tenant {tenant.id}")
                except HTTPException as e:
                    results[tenant.id] = {"status": "error", "message": str(e.detail)}
                    logger.warning(
                        f"Error retraining model for tenant {tenant.id}: {e.detail}"
                    )
                except Exception as e:
                    results[tenant.id] = {"status": "error", "message": str(e)}
                    logger.error(
                        f"Unexpected error retraining model for tenant {tenant.id}: {e}"
                    )

            return results
        except Exception as e:
            logger.exception(f"Error retraining tenant models: {e}")
            return {}
