"""Test suite for the ML Engine module.

This module tests the ML Engine's ability to train isolation forest models,
perform inference, and manage model persistence.
"""

import datetime
import json
import os
import shutil
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from sklearn.ensemble import IsolationForest

from auditpulse_mvp.database.models import (
    AnomalyType,
    DataSource,
    Tenant,
    Transaction,
)
from auditpulse_mvp.ml_engine.ml_engine import DEFAULT_MODEL_PARAMS, MLEngine


@pytest.fixture
def mock_tenant_id():
    """Create a mock tenant ID."""
    return uuid.uuid4()


@pytest.fixture
def mock_tenant(mock_tenant_id):
    """Create a mock tenant."""
    tenant = MagicMock(spec=Tenant)
    tenant.id = mock_tenant_id
    tenant.name = "Test Tenant"
    tenant.is_active = True
    tenant.enable_ml_engine = True
    return tenant


@pytest.fixture
def mock_transactions(mock_tenant_id):
    """Create a list of mock transactions for training."""
    transactions = []

    # Generate 150 regular transactions (75% of data)
    categories = ["Food", "Transport", "Office Supplies", "Software", "Travel"]
    merchants = ["Restaurant A", "Taxi Co", "Office Store", "SaaS Provider", "Airline"]

    # Transaction template generation function
    def create_transaction(amount, category, merchant, days_ago, is_anomaly=False):
        txn_date = datetime.datetime.now() - datetime.timedelta(days=days_ago)
        return MagicMock(
            spec=Transaction,
            id=uuid.uuid4(),
            tenant_id=mock_tenant_id,
            amount=Decimal(str(amount)),
            transaction_date=txn_date,
            category=category,
            merchant_name=merchant,
            source=DataSource.QUICKBOOKS,
            description=f"{merchant} purchase",
            is_deleted=False,
        )

    # Generate normal transactions
    for i in range(120):
        category_idx = i % len(categories)
        merchant_idx = i % len(merchants)
        # Normal transactions between $10 and $500
        amount = round(10 + (490 * np.random.random()), 2)
        days_ago = np.random.randint(1, 360)  # Spread over last year

        txn = create_transaction(
            amount=amount,
            category=categories[category_idx],
            merchant=merchants[merchant_idx],
            days_ago=days_ago,
        )
        transactions.append(txn)

    # Generate anomalous transactions (25% of data)
    for i in range(30):
        category_idx = i % len(categories)
        merchant_idx = i % len(merchants)
        # Anomalous transactions with very large amounts
        amount = round(2000 + (8000 * np.random.random()), 2)
        days_ago = np.random.randint(1, 360)

        txn = create_transaction(
            amount=amount,
            category=categories[category_idx],
            merchant=merchants[merchant_idx],
            days_ago=days_ago,
            is_anomaly=True,
        )
        transactions.append(txn)

    # Add a few transactions with unusual merchants
    for i in range(10):
        unusual_merchant = f"Unknown Vendor {i}"
        amount = round(100 + (900 * np.random.random()), 2)
        days_ago = np.random.randint(1, 360)

        txn = create_transaction(
            amount=amount,
            category="Miscellaneous",
            merchant=unusual_merchant,
            days_ago=days_ago,
            is_anomaly=True,
        )
        transactions.append(txn)

    return transactions


@pytest.fixture
def mock_db_session(mock_transactions, mock_tenant):
    """Mock database session for testing."""
    session = MagicMock()

    # Setup mock responses for common queries
    async def mock_execute(stmt):
        result_mock = MagicMock()

        # For tenant query
        if "Tenant" in str(stmt):
            result_mock.scalars.return_value.all.return_value = [mock_tenant]

        # For transaction query
        if "Transaction" in str(stmt):
            result_mock.scalars.return_value.all.return_value = mock_transactions

        return result_mock

    session.execute = mock_execute
    return session


@pytest.fixture
def mock_model_dir(monkeypatch, tmp_path):
    """Set up a temporary directory for model storage."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Monkeypatch the base directory method to return our temp dir
    def mock_get_base_dir():
        return str(models_dir)

    monkeypatch.setattr(MLEngine, "_get_base_model_directory", mock_get_base_dir)

    yield models_dir

    # Cleanup - remove temp dir
    if models_dir.exists():
        shutil.rmtree(models_dir)


@pytest.fixture
def ml_engine(mock_db_session):
    """Create an MLEngine instance with a mock session."""
    return MLEngine(db_session=mock_db_session)


class TestMLEngineInitialization:
    """Test MLEngine initialization."""

    def test_init_creates_model_directory(self, ml_engine, mock_model_dir):
        """Test that initializing MLEngine creates the model directory."""
        assert mock_model_dir.exists()


class TestMLEngineDirectoryManagement:
    """Test MLEngine directory management functions."""

    def test_get_tenant_model_directory(
        self, ml_engine, mock_tenant_id, mock_model_dir
    ):
        """Test getting the tenant model directory."""
        tenant_dir = ml_engine._get_tenant_model_directory(mock_tenant_id)
        expected_dir = os.path.join(str(mock_model_dir), str(mock_tenant_id))
        assert tenant_dir == expected_dir
        assert os.path.exists(tenant_dir)

    def test_get_latest_model_path_no_models(self, ml_engine, mock_tenant_id):
        """Test getting the latest model path when no models exist."""
        latest_path = ml_engine._get_latest_model_path(mock_tenant_id)
        assert latest_path is None

    def test_get_latest_model_path_with_models(
        self, ml_engine, mock_tenant_id, mock_model_dir
    ):
        """Test getting the latest model path when models exist."""
        # Create tenant directory
        tenant_dir = os.path.join(str(mock_model_dir), str(mock_tenant_id))
        os.makedirs(tenant_dir, exist_ok=True)

        # Create some model files with different timestamps
        older_model = os.path.join(tenant_dir, "isolation_forest_20230101_120000.pkl")
        newer_model = os.path.join(tenant_dir, "isolation_forest_20230102_120000.pkl")

        # Create the files
        Path(older_model).touch()
        # Sleep to ensure different modification times
        import time

        time.sleep(0.1)
        Path(newer_model).touch()

        # Get latest model path
        latest_path = ml_engine._get_latest_model_path(mock_tenant_id)
        assert latest_path == newer_model


class TestMLEngineDataProcessing:
    """Test MLEngine data processing functions."""

    def test_prepare_features(self, ml_engine, mock_transactions):
        """Test feature preparation from transactions."""
        df = ml_engine._prepare_features(mock_transactions[:5])

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

        # Check expected columns
        expected_columns = [
            "amount",
            "source",
            "merchant_name",
            "category",
            "day_of_week",
            "hour_of_day",
            "month",
            "day_of_month",
            "transaction_date",
        ]
        for col in expected_columns:
            assert col in df.columns

    def test_encode_categorical_features(self, ml_engine):
        """Test encoding of categorical features."""
        # Create a test DataFrame
        test_df = pd.DataFrame(
            {
                "amount": [100.0, 200.0, 300.0, 400.0],
                "category": ["Food", "Transport", "Food", "Software"],
                "merchant_name": ["Vendor A", "Vendor B", "Vendor A", "Vendor C"],
                "source": ["QUICKBOOKS", "PLAID", "QUICKBOOKS", "QUICKBOOKS"],
            }
        )

        # Encode the features
        encoded_df = ml_engine._encode_categorical_features(test_df)

        # Check that original categorical columns are dropped
        assert "category" not in encoded_df.columns
        assert "merchant_name" not in encoded_df.columns
        assert "source" not in encoded_df.columns

        # Check that new encoded columns are created
        assert "category_Food" in encoded_df.columns
        assert "merchant_name_Vendor A" in encoded_df.columns
        assert "source_QUICKBOOKS" in encoded_df.columns

        # Check encoding is correct
        assert encoded_df["category_Food"].tolist() == [1, 0, 1, 0]
        assert encoded_df["merchant_name_Vendor A"].tolist() == [1, 0, 1, 0]
        assert encoded_df["source_QUICKBOOKS"].tolist() == [1, 0, 1, 1]

    def test_preprocess_data(self, ml_engine):
        """Test data preprocessing for model training."""
        # Create a test DataFrame
        test_date = datetime.datetime.now()
        test_df = pd.DataFrame(
            {
                "amount": [100.0, 200.0, 300.0, 400.0],
                "day_of_week": [0, 1, 2, 3],
                "hour_of_day": [9, 10, 11, 12],
                "month": [1, 2, 3, 4],
                "day_of_month": [10, 15, 20, 25],
                "category": ["Food", "Transport", "Food", "Software"],
                "merchant_name": ["Vendor A", "Vendor B", "Vendor A", "Vendor C"],
                "source": ["QUICKBOOKS", "PLAID", "QUICKBOOKS", "QUICKBOOKS"],
                "transaction_date": [test_date] * 4,
            }
        )

        # Preprocess the data
        processed_df, scaler = ml_engine._preprocess_data(test_df)

        # Check that date column is dropped
        assert "transaction_date" not in processed_df.columns

        # Check that numerical features are scaled
        for col in ["amount", "day_of_week", "hour_of_day", "month", "day_of_month"]:
            # Standard scaling should result in mean ≈ 0 and std ≈ 1
            assert -3 < processed_df[col].mean() < 3  # Loose check due to small sample
            assert 0 < processed_df[col].std() < 2  # Loose check due to small sample

        # Check that categorical features are encoded
        assert "category_Food" in processed_df.columns
        assert "merchant_name_Vendor A" in processed_df.columns
        assert "source_QUICKBOOKS" in processed_df.columns


@pytest.mark.asyncio
class TestMLEngineModelTraining:
    """Test MLEngine model training functions."""

    async def test_get_training_data(
        self, ml_engine, mock_tenant_id, mock_transactions
    ):
        """Test getting training data from the database."""
        df = await ml_engine._get_training_data(mock_tenant_id)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(mock_transactions)

        # Check expected columns
        expected_columns = [
            "amount",
            "source",
            "merchant_name",
            "category",
            "day_of_week",
            "hour_of_day",
            "month",
            "day_of_month",
            "transaction_date",
        ]
        for col in expected_columns:
            assert col in df.columns

    async def test_get_training_data_not_enough_data(
        self, ml_engine, mock_tenant_id, mock_db_session
    ):
        """Test exception when not enough data for training."""
        # Mock empty transaction list
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        mock_db_session.execute.return_value = result_mock

        with pytest.raises(HTTPException) as excinfo:
            await ml_engine._get_training_data(mock_tenant_id)

        assert "Not enough transaction data" in str(excinfo.value)

    async def test_train_model(self, ml_engine, mock_tenant_id, mock_model_dir):
        """Test model training functionality."""
        # Train a model
        result = await ml_engine.train_model(mock_tenant_id)

        # Check result structure
        assert "model_path" in result
        assert "info_path" in result
        assert "training_samples" in result
        assert "anomaly_rate" in result
        assert "anomaly_count" in result
        assert "model_info" in result

        # Check that files were created
        assert os.path.exists(result["model_path"])
        assert os.path.exists(result["info_path"])

        # Check model info content
        with open(result["info_path"], "r") as f:
            model_info = json.load(f)

        assert model_info["tenant_id"] == str(mock_tenant_id)
        assert "created_at" in model_info
        assert "data_start_date" in model_info
        assert "data_end_date" in model_info
        assert "training_samples" in model_info
        assert "parameters" in model_info
        assert "anomaly_rate" in model_info
        assert "feature_importance" in model_info

        # Load the model file to verify it's valid
        model_data = joblib.load(result["model_path"])
        assert "model" in model_data
        assert "scaler" in model_data
        assert "feature_names" in model_data
        assert isinstance(model_data["model"], IsolationForest)

    def test_calculate_feature_importance(self, ml_engine):
        """Test feature importance calculation."""
        # Create a simple model
        X = np.random.random((100, 5))
        model = IsolationForest(random_state=42).fit(X)
        feature_names = ["feature1", "feature2", "feature3", "feature4", "feature5"]

        importance = ml_engine._calculate_feature_importance(model, feature_names)

        # Check result structure
        assert isinstance(importance, dict)
        assert len(importance) == 5
        assert all(feature in importance for feature in feature_names)
        assert all(isinstance(importance[feature], float) for feature in feature_names)

        # Check that values sum to approximately 1
        assert sum(importance.values()) > 0.99 and sum(importance.values()) < 1.01


@pytest.mark.asyncio
class TestMLEngineModelManagement:
    """Test MLEngine model management functions."""

    async def test_load_model_success(self, ml_engine, mock_tenant_id, mock_model_dir):
        """Test loading a trained model."""
        # First train a model
        train_result = await ml_engine.train_model(mock_tenant_id)

        # Then load it
        model_data = await ml_engine.load_model(mock_tenant_id)

        # Check the loaded data
        assert "model" in model_data
        assert "scaler" in model_data
        assert "feature_names" in model_data
        assert "model_path" in model_data
        assert "model_info" in model_data

        assert isinstance(model_data["model"], IsolationForest)
        assert model_data["model_path"] == train_result["model_path"]

    async def test_load_model_not_found(self, ml_engine, mock_tenant_id):
        """Test exception when no model exists."""
        # Try to load a model for a tenant with no trained models
        with pytest.raises(HTTPException) as excinfo:
            await ml_engine.load_model(uuid.uuid4())  # Use a different tenant ID

        assert "No trained model found" in str(excinfo.value)

    async def test_get_tenant_models(self, ml_engine, mock_tenant_id):
        """Test retrieving all models for a tenant."""
        # First train two models
        await ml_engine.train_model(mock_tenant_id)
        await ml_engine.train_model(mock_tenant_id)

        # Get all models
        models = await ml_engine.get_tenant_models(mock_tenant_id)

        # Check result
        assert isinstance(models, list)
        assert len(models) == 2

        # Check model info structure
        for model_info in models:
            assert "tenant_id" in model_info
            assert "created_at" in model_info
            assert "training_samples" in model_info
            assert "filename" in model_info

    async def test_delete_model(self, ml_engine, mock_tenant_id):
        """Test deleting a model."""
        # First train a model
        train_result = await ml_engine.train_model(mock_tenant_id)
        model_filename = os.path.basename(train_result["model_path"])

        # Delete the model
        result = await ml_engine.delete_model(mock_tenant_id, model_filename)

        # Check result
        assert result is True
        assert not os.path.exists(train_result["model_path"])
        assert not os.path.exists(train_result["info_path"])


@pytest.mark.asyncio
class TestMLEngineInference:
    """Test MLEngine inference functions."""

    async def test_score_transaction(
        self, ml_engine, mock_tenant_id, mock_transactions
    ):
        """Test scoring a single transaction."""
        # First train a model
        await ml_engine.train_model(mock_tenant_id)

        # Score a normal transaction
        normal_txn = [t for t in mock_transactions if float(t.amount) < 1000][0]
        normal_score = await ml_engine.score_transaction(mock_tenant_id, normal_txn)

        # Score an anomalous transaction
        anomalous_txn = [t for t in mock_transactions if float(t.amount) > 5000][0]
        anomalous_score = await ml_engine.score_transaction(
            mock_tenant_id, anomalous_txn
        )

        # Check scores
        assert 0 <= normal_score <= 1
        assert 0 <= anomalous_score <= 1

        # Anomalous transaction should have a higher score (more anomalous)
        assert anomalous_score > normal_score

    async def test_batch_score_transactions(
        self, ml_engine, mock_tenant_id, mock_transactions
    ):
        """Test scoring multiple transactions in batch."""
        # First train a model
        await ml_engine.train_model(mock_tenant_id)

        # Select a mix of normal and anomalous transactions
        transactions_to_score = mock_transactions[:20]

        # Score the batch
        scores = await ml_engine.batch_score_transactions(
            mock_tenant_id, transactions_to_score
        )

        # Check result structure
        assert isinstance(scores, dict)
        assert len(scores) == len(transactions_to_score)

        # Check that all transaction IDs are in the result
        txn_ids = [t.id for t in transactions_to_score]
        assert all(txn_id in scores for txn_id in txn_ids)

        # Check score values
        assert all(0 <= score <= 1 for score in scores.values())

        # Check that high-amount transactions get higher anomaly scores
        high_amount_txns = [t for t in transactions_to_score if float(t.amount) > 1000]
        low_amount_txns = [t for t in transactions_to_score if float(t.amount) < 1000]

        if high_amount_txns and low_amount_txns:
            high_scores = [scores[t.id] for t in high_amount_txns]
            low_scores = [scores[t.id] for t in low_amount_txns]

            assert sum(high_scores) / len(high_scores) > sum(low_scores) / len(
                low_scores
            )


@pytest.mark.asyncio
class TestMLEngineRetraining:
    """Test MLEngine retraining functions."""

    async def test_retrain_all_tenant_models(
        self, ml_engine, mock_tenant, mock_db_session
    ):
        """Test retraining models for all tenants."""
        # Run the retraining
        results = await ml_engine.retrain_all_tenant_models()

        # Check results
        assert isinstance(results, dict)
        assert mock_tenant.id in results
        assert results[mock_tenant.id]["status"] == "success"
        assert "model_path" in results[mock_tenant.id]
        assert "training_samples" in results[mock_tenant.id]
        assert "anomaly_rate" in results[mock_tenant.id]

    async def test_retrain_tenant_ml_disabled(
        self, ml_engine, mock_tenant, mock_db_session
    ):
        """Test skipping retraining for tenants with ML disabled."""
        # Disable ML for the tenant
        mock_tenant.enable_ml_engine = False

        # Run the retraining
        results = await ml_engine.retrain_all_tenant_models()

        # Check results
        assert mock_tenant.id in results
        assert results[mock_tenant.id]["status"] == "skipped"
        assert results[mock_tenant.id]["reason"] == "ML engine disabled"

    async def test_retrain_tenant_not_enough_data(
        self, ml_engine, mock_tenant, mock_db_session
    ):
        """Test handling retraining error when not enough data."""
        # Mock to return empty transaction list
        empty_result_mock = MagicMock()
        empty_result_mock.scalars.return_value.all.return_value = []

        # Keep tenant query result but change transaction query result
        original_execute = mock_db_session.execute

        async def modified_execute(stmt):
            if "Tenant" in str(stmt):
                return await original_execute(stmt)
            else:
                return empty_result_mock

        mock_db_session.execute = modified_execute

        # Run the retraining
        results = await ml_engine.retrain_all_tenant_models()

        # Check results
        assert mock_tenant.id in results
        assert results[mock_tenant.id]["status"] == "error"
        assert "Not enough transaction data" in results[mock_tenant.id]["message"]
