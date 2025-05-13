"""ML Engine for AuditPulse MVP.

This module implements the ML-based anomaly detection for transactions using IsolationForest.
Features:
- Per-tenant training pipeline
- Model persistence with versioning
- Real-time inference
- Nightly retraining job
"""

import datetime
import logging
import os
import pickle
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, HTTPException, status
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.database.models import DataSource, Tenant, Transaction

# Configure logging
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = Path("models")
DEFAULT_FEATURES = [
    "amount",
    "day_of_week",
    "month",
    "hour",
    "is_weekend",
    "is_end_of_month",
]
DEFAULT_CONTAMINATION = 0.05  # Expected ratio of anomalies (5%)
DEFAULT_MAX_SAMPLES = 256  # For performance reasons
MIN_SAMPLES_REQUIRED = 50  # Minimum number of transactions needed for training


class MLEngine:
    """ML Engine for anomaly detection using IsolationForest."""

    def __init__(self, db_session: AsyncSession = Depends(get_db_session)):
        """Initialize the ML Engine.

        Args:
            db_session: The database session.
        """
        self.db_session = db_session
        self._ensure_model_directory()

    def _ensure_model_directory(self):
        """Ensure the model directory structure exists."""
        if not MODEL_DIR.exists():
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created model directory: {MODEL_DIR}")

    def _get_tenant_model_dir(self, tenant_id: uuid.UUID) -> Path:
        """Get the model directory for a specific tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            Path: The tenant's model directory path.
        """
        tenant_dir = MODEL_DIR / str(tenant_id)
        if not tenant_dir.exists():
            tenant_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created tenant model directory: {tenant_dir}")
        return tenant_dir

    def _get_latest_model_path(self, tenant_id: uuid.UUID) -> Optional[Path]:
        """Get the path to the latest model for a tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            Optional[Path]: The path to the latest model, or None if no model exists.
        """
        tenant_dir = self._get_tenant_model_dir(tenant_id)
        model_files = sorted(tenant_dir.glob("*.pkl"), reverse=True)
        return model_files[0] if model_files else None

    def _get_model_path(
        self, tenant_id: uuid.UUID, timestamp: Optional[datetime.datetime] = None
    ) -> Path:
        """Get the path for a model at a specific timestamp.

        Args:
            tenant_id: The tenant ID.
            timestamp: Optional timestamp for the model. If None, current time is used.

        Returns:
            Path: The model path.
        """
        timestamp = timestamp or datetime.datetime.now()
        tenant_dir = self._get_tenant_model_dir(tenant_id)
        return tenant_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}.pkl"

    async def get_training_data(
        self, tenant_id: uuid.UUID, days: int = 180, limit: int = 10000
    ) -> pd.DataFrame:
        """Get training data for a tenant.

        Args:
            tenant_id: The tenant ID.
            days: The number of days of historical data to use.
            limit: Maximum number of transactions to retrieve.

        Returns:
            pd.DataFrame: The training data.

        Raises:
            ValueError: If insufficient data is available for training.
        """
        # Calculate the start date
        start_date = datetime.datetime.now() - datetime.timedelta(days=days)

        # Query transactions
        stmt = (
            select(Transaction)
            .where(
                and_(
                    Transaction.tenant_id == tenant_id,
                    Transaction.transaction_date >= start_date,
                    Transaction.is_deleted == False,
                )
            )
            .order_by(Transaction.transaction_date.desc())
            .limit(limit)
        )

        result = await self.db_session.execute(stmt)
        transactions = list(result.scalars().all())

        if len(transactions) < MIN_SAMPLES_REQUIRED:
            raise ValueError(
                f"Insufficient training data. Got {len(transactions)} transactions, "
                f"but need at least {MIN_SAMPLES_REQUIRED}."
            )

        # Convert to DataFrame and extract features
        return self._prepare_features(transactions)

    def _prepare_features(self, transactions: List[Transaction]) -> pd.DataFrame:
        """Extract and prepare features from transactions.

        Args:
            transactions: The list of transactions.

        Returns:
            pd.DataFrame: The prepared features.
        """
        # Create base dataframe
        df = pd.DataFrame([self._transaction_to_dict(txn) for txn in transactions])

        # Extract temporal features
        df["day_of_week"] = df["transaction_date"].dt.dayofweek
        df["month"] = df["transaction_date"].dt.month
        df["hour"] = df["transaction_date"].dt.hour
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_end_of_month"] = (df["transaction_date"].dt.day >= 25).astype(int)

        # Create account-specific features
        # Create dummy variables for source_account_id
        if "source_account_id" in df.columns:
            account_dummies = pd.get_dummies(
                df["source_account_id"], prefix="account", dummy_na=True
            )
            df = pd.concat([df, account_dummies], axis=1)

        # Merchant features (optional, can be very sparse)
        # df["merchant_count"] = df.groupby("merchant_name")["merchant_name"].transform("count")

        # Select features for model training
        return df

    def _transaction_to_dict(self, transaction: Transaction) -> Dict[str, Any]:
        """Convert a transaction to a dictionary for feature extraction.

        Args:
            transaction: The transaction.

        Returns:
            Dict[str, Any]: The transaction as a dictionary.
        """
        return {
            "id": transaction.id,
            "transaction_id": transaction.transaction_id,
            "source": transaction.source,
            "source_account_id": transaction.source_account_id,
            "amount": transaction.amount,
            "currency": transaction.currency,
            "description": transaction.description,
            "category": transaction.category,
            "merchant_name": transaction.merchant_name,
            "transaction_date": transaction.transaction_date,
            "posting_date": transaction.posting_date,
        }

    async def train_model(
        self,
        tenant_id: uuid.UUID,
        features: Optional[List[str]] = None,
        contamination: float = DEFAULT_CONTAMINATION,
        max_samples: int = DEFAULT_MAX_SAMPLES,
    ) -> Path:
        """Train an isolation forest model for a tenant.

        Args:
            tenant_id: The tenant ID.
            features: Optional list of features to use. If None, defaults are used.
            contamination: The expected ratio of anomalies.
            max_samples: The maximum number of samples to use.

        Returns:
            Path: The path to the saved model.

        Raises:
            ValueError: If there is insufficient data for training.
            HTTPException: If an error occurs during training.
        """
        try:
            # Use default features if none provided
            features_to_use = features or DEFAULT_FEATURES

            # Get training data
            df = await self.get_training_data(tenant_id)

            # Check for missing features
            missing_features = [f for f in features_to_use if f not in df.columns]
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                features_to_use = [f for f in features_to_use if f in df.columns]

            if not features_to_use:
                raise ValueError("No valid features available for training")

            # Select features and handle missing values
            X = df[features_to_use].fillna(0)

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train the model
            model = IsolationForest(
                contamination=contamination,
                max_samples=min(max_samples, len(X)),
                random_state=42,
                n_jobs=-1,  # Use all available CPU cores
            )

            model.fit(X_scaled)

            # Save the model and metadata
            model_path = self._get_model_path(tenant_id)
            model_data = {
                "model": model,
                "scaler": scaler,
                "features": features_to_use,
                "trained_at": datetime.datetime.now(),
                "num_samples": len(X),
                "contamination": contamination,
            }

            joblib.dump(model_data, model_path)
            logger.info(f"Trained and saved model to {model_path}")

            return model_path

        except ValueError as e:
            logger.error(f"Value error during model training: {e}")
            raise
        except Exception as e:
            logger.error(f"Error training model for tenant {tenant_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error training model: {str(e)}",
            )

    async def load_model(self, tenant_id: uuid.UUID) -> Dict[str, Any]:
        """Load the latest model for a tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            Dict[str, Any]: The loaded model data.

        Raises:
            ValueError: If no model exists for the tenant.
        """
        model_path = self._get_latest_model_path(tenant_id)
        if not model_path:
            # Try to train a new model
            try:
                model_path = await self.train_model(tenant_id)
            except ValueError:
                raise ValueError(
                    f"No model exists for tenant {tenant_id} and insufficient data to train one"
                )

        try:
            model_data = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
            return model_data
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise ValueError(f"Failed to load model: {str(e)}")

    def _extract_features_from_transaction(
        self, transaction: Transaction, feature_list: List[str]
    ) -> pd.DataFrame:
        """Extract features from a single transaction.

        Args:
            transaction: The transaction.
            feature_list: The list of features to extract.

        Returns:
            pd.DataFrame: The extracted features.
        """
        # Convert transaction to dictionary
        txn_dict = self._transaction_to_dict(transaction)

        # Create a DataFrame with a single row
        df = pd.DataFrame([txn_dict])

        # Extract temporal features if needed
        if "day_of_week" in feature_list and "transaction_date" in df:
            df["day_of_week"] = df["transaction_date"].dt.dayofweek

        if "month" in feature_list and "transaction_date" in df:
            df["month"] = df["transaction_date"].dt.month

        if "hour" in feature_list and "transaction_date" in df:
            df["hour"] = df["transaction_date"].dt.hour

        if "is_weekend" in feature_list and "transaction_date" in df:
            df["is_weekend"] = (df["transaction_date"].dt.dayofweek >= 5).astype(int)

        if "is_end_of_month" in feature_list and "transaction_date" in df:
            df["is_end_of_month"] = (df["transaction_date"].dt.day >= 25).astype(int)

        # Account dummies (only for the specific account in this transaction)
        if (
            any(f.startswith("account_") for f in feature_list)
            and "source_account_id" in df
        ):
            for feature in feature_list:
                if feature.startswith("account_"):
                    account_id = feature.replace("account_", "")
                    df[feature] = (df["source_account_id"] == account_id).astype(int)

        # Fill missing features with 0
        for feature in feature_list:
            if feature not in df.columns:
                df[feature] = 0

        # Select only the requested features
        return df[feature_list].fillna(0)

    async def score(self, tenant_id: uuid.UUID, transaction: Transaction) -> float:
        """Score a transaction for anomaly detection.

        Args:
            tenant_id: The tenant ID.
            transaction: The transaction to score.

        Returns:
            float: The anomaly score between 0 and 1, where higher values indicate higher anomaly likelihood.

        Raises:
            ValueError: If no model is available or scoring fails.
        """
        try:
            # Load the model
            model_data = await self.load_model(tenant_id)
            model = model_data["model"]
            scaler = model_data["scaler"]
            features = model_data["features"]

            # Extract features from the transaction
            X = self._extract_features_from_transaction(transaction, features)

            # Scale the features
            X_scaled = scaler.transform(X)

            # Get the decision function value (-ve = anomaly, +ve = normal)
            decision_value = model.decision_function(X_scaled)[0]

            # Convert to a 0-1 score (higher = more anomalous)
            # We use 0.5 - X to flip the scale, then clamp to [0, 1]
            score = min(1.0, max(0.0, 0.5 - (decision_value / 2)))

            return score

        except ValueError as e:
            logger.error(f"Value error during scoring: {e}")
            # Return a moderate score if no model exists
            return 0.5
        except Exception as e:
            logger.error(
                f"Error scoring transaction {transaction.id} for tenant {tenant_id}: {e}"
            )
            # Return a moderate score in case of error
            return 0.5

    async def batch_train_models(
        self, tenant_ids: Optional[List[uuid.UUID]] = None
    ) -> Dict[str, Any]:
        """Train models for multiple tenants.

        Args:
            tenant_ids: Optional list of tenant IDs. If None, all active tenants are used.

        Returns:
            Dict[str, Any]: The training results.
        """
        if tenant_ids is None:
            # Get all active tenants
            stmt = select(Tenant).where(Tenant.is_active == True)
            result = await self.db_session.execute(stmt)
            tenants = list(result.scalars().all())
            tenant_ids = [tenant.id for tenant in tenants]

        results = {
            "success": [],
            "error": [],
            "total": len(tenant_ids),
        }

        for tenant_id in tenant_ids:
            try:
                model_path = await self.train_model(tenant_id)
                results["success"].append(
                    {
                        "tenant_id": str(tenant_id),
                        "model_path": str(model_path),
                    }
                )
            except Exception as e:
                logger.error(f"Error training model for tenant {tenant_id}: {e}")
                results["error"].append(
                    {
                        "tenant_id": str(tenant_id),
                        "error": str(e),
                    }
                )

        logger.info(
            f"Batch training completed. Success: {len(results['success'])}, Error: {len(results['error'])}"
        )
        return results


# Singleton instance for dependency injection
ml_engine = MLEngine()


def get_ml_engine() -> MLEngine:
    """Dependency function for FastAPI to get the ML engine.

    Returns:
        MLEngine: The ML engine instance.
    """
    return ml_engine
