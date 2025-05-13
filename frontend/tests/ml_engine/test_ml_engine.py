"""Tests for the ML Engine module.

This module contains tests for the ML-based anomaly detection,
covering model training, scoring, and persistence.
"""

import datetime
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.models import DataSource, Tenant, Transaction
from auditpulse_mvp.ml_engine.ml_engine import (
    DEFAULT_CONTAMINATION,
    DEFAULT_FEATURES,
    DEFAULT_MAX_SAMPLES,
    MIN_SAMPLES_REQUIRED,
    MLEngine,
    get_ml_engine,
)


@pytest.fixture
def test_model_dir():
    """Create a temporary directory for storing test models."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Clean up after the test
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_transaction() -> Transaction:
    """Create a mock transaction for testing."""
    return Transaction(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        transaction_id="test-txn-001",
        source=DataSource.QUICKBOOKS,
        source_account_id="test-account",
        amount=5000.0,
        currency="USD",
        description="Test transaction",
        category="Office Supplies",
        merchant_name="Test Vendor",
        transaction_date=datetime.datetime.now(),
    )


@pytest.fixture
def mock_transactions() -> list[Transaction]:
    """Create a list of mock transactions for testing."""
    tenant_id = uuid.uuid4()
    base_date = datetime.datetime.now() - datetime.timedelta(days=30)
    return [
        Transaction(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            transaction_id=f"test-txn-{i}",
            source=DataSource.QUICKBOOKS,
            source_account_id=f"test-account-{i % 3}",  # 3 different accounts
            amount=1000.0 + (i * 100),  # Varying amounts
            currency="USD",
            description=f"Test transaction {i}",
            category="Office Supplies",
            merchant_name="Test Vendor",
            transaction_date=base_date + datetime.timedelta(days=i),
        )
        for i in range(MIN_SAMPLES_REQUIRED + 10)  # Enough for training
    ]


@pytest.fixture
def mock_anomalous_transaction() -> Transaction:
    """Create a mock anomalous transaction with an unusually large amount."""
    return Transaction(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        transaction_id="test-txn-anomaly",
        source=DataSource.QUICKBOOKS,
        source_account_id="test-account",
        amount=100000.0,  # Unusually large amount
        currency="USD",
        description="Potentially anomalous transaction",
        category="Office Supplies",
        merchant_name="Test Vendor",
        transaction_date=datetime.datetime.now(),
    )


@pytest.fixture
def mock_model_data():
    """Create mock model data for testing."""
    # Train a simple model with synthetic data
    X = np.random.randn(100, 5)  # 100 samples, 5 features
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)

    scaler = StandardScaler()
    scaler.fit(X)

    return {
        "model": model,
        "scaler": scaler,
        "features": ["amount", "day_of_week", "month", "hour", "is_weekend"],
        "trained_at": datetime.datetime.now(),
        "num_samples": 100,
        "contamination": 0.1,
    }


@pytest.mark.asyncio
async def test_ml_engine_initialization(db_session):
    """Test ML Engine initialization."""
    engine = MLEngine(db_session)

    assert engine.db_session == db_session


@pytest.mark.asyncio
async def test_ensure_model_directory(test_model_dir):
    """Test ensuring the model directory exists."""
    engine = MLEngine()

    with patch("auditpulse_mvp.ml_engine.ml_engine.MODEL_DIR", test_model_dir):
        # Remove the directory to test creation
        if test_model_dir.exists():
            shutil.rmtree(test_model_dir)

        engine._ensure_model_directory()

        assert test_model_dir.exists()
        assert test_model_dir.is_dir()


@pytest.mark.asyncio
async def test_get_tenant_model_dir(test_model_dir):
    """Test getting the tenant model directory."""
    engine = MLEngine()
    tenant_id = uuid.uuid4()

    with patch("auditpulse_mvp.ml_engine.ml_engine.MODEL_DIR", test_model_dir):
        tenant_dir = engine._get_tenant_model_dir(tenant_id)

        assert tenant_dir == test_model_dir / str(tenant_id)
        assert tenant_dir.exists()
        assert tenant_dir.is_dir()


@pytest.mark.asyncio
async def test_get_latest_model_path(test_model_dir):
    """Test getting the latest model path."""
    engine = MLEngine()
    tenant_id = uuid.uuid4()

    with patch("auditpulse_mvp.ml_engine.ml_engine.MODEL_DIR", test_model_dir):
        # Create tenant directory
        tenant_dir = test_model_dir / str(tenant_id)
        tenant_dir.mkdir(parents=True, exist_ok=True)

        # No models yet
        assert engine._get_latest_model_path(tenant_id) is None

        # Create some model files
        for days in [10, 5, 1]:
            timestamp = datetime.datetime.now() - datetime.timedelta(days=days)
            model_path = tenant_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}.pkl"
            model_path.touch()

        # Should return the most recent model
        latest_path = engine._get_latest_model_path(tenant_id)
        assert latest_path is not None
        assert latest_path.name.startswith(
            (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y%m%d")
        )


@pytest.mark.asyncio
async def test_get_model_path(test_model_dir):
    """Test getting a model path with a specific timestamp."""
    engine = MLEngine()
    tenant_id = uuid.uuid4()
    timestamp = datetime.datetime(2023, 1, 15, 12, 30, 45)

    with patch("auditpulse_mvp.ml_engine.ml_engine.MODEL_DIR", test_model_dir):
        model_path = engine._get_model_path(tenant_id, timestamp)

        assert model_path == test_model_dir / str(tenant_id) / "20230115_123045.pkl"


@pytest.mark.asyncio
async def test_get_training_data_sufficient(db_session, mock_transactions):
    """Test getting training data with sufficient transactions."""
    engine = MLEngine(db_session)
    tenant_id = mock_transactions[0].tenant_id

    # Mock the DB query to return our mock transactions
    with patch.object(db_session, "execute") as mock_execute:
        mock_result = AsyncMock()
        mock_result.scalars.return_value.all.return_value = mock_transactions
        mock_execute.return_value = mock_result

        df = await engine.get_training_data(tenant_id)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(mock_transactions)

        # Check that the prepared features include the defaults
        for feature in DEFAULT_FEATURES:
            if feature != "amount":  # 'amount' is in the original transaction
                assert feature in df.columns


@pytest.mark.asyncio
async def test_get_training_data_insufficient(db_session):
    """Test getting training data with insufficient transactions."""
    engine = MLEngine(db_session)
    tenant_id = uuid.uuid4()

    # Mock the DB query to return too few transactions
    with patch.object(db_session, "execute") as mock_execute:
        mock_result = AsyncMock()
        mock_result.scalars.return_value.all.return_value = [
            MagicMock() for _ in range(MIN_SAMPLES_REQUIRED - 1)
        ]
        mock_execute.return_value = mock_result

        with pytest.raises(ValueError) as excinfo:
            await engine.get_training_data(tenant_id)

        assert "Insufficient training data" in str(excinfo.value)


@pytest.mark.asyncio
async def test_prepare_features(mock_transactions):
    """Test preparing features from transactions."""
    engine = MLEngine()
    df = engine._prepare_features(mock_transactions)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(mock_transactions)

    # Check temporal features
    assert "day_of_week" in df.columns
    assert "month" in df.columns
    assert "hour" in df.columns
    assert "is_weekend" in df.columns
    assert "is_end_of_month" in df.columns

    # Check account dummies
    account_columns = [col for col in df.columns if col.startswith("account_")]
    assert len(account_columns) > 0


@pytest.mark.asyncio
async def test_transaction_to_dict(mock_transaction):
    """Test converting a transaction to a dictionary."""
    engine = MLEngine()
    result = engine._transaction_to_dict(mock_transaction)

    assert isinstance(result, dict)
    assert result["id"] == mock_transaction.id
    assert result["amount"] == mock_transaction.amount
    assert result["merchant_name"] == mock_transaction.merchant_name


@pytest.mark.asyncio
async def test_train_model_success(db_session, mock_transactions, test_model_dir):
    """Test successfully training a model."""
    engine = MLEngine(db_session)
    tenant_id = mock_transactions[0].tenant_id

    with patch(
        "auditpulse_mvp.ml_engine.ml_engine.MODEL_DIR", test_model_dir
    ), patch.object(engine, "get_training_data") as mock_get_data, patch(
        "joblib.dump"
    ) as mock_dump:

        # Mock the training data
        df = pd.DataFrame(
            {
                "amount": np.random.randn(100),
                "day_of_week": np.random.randint(0, 7, 100),
                "month": np.random.randint(1, 13, 100),
                "hour": np.random.randint(0, 24, 100),
                "is_weekend": np.random.randint(0, 2, 100),
                "is_end_of_month": np.random.randint(0, 2, 100),
            }
        )
        mock_get_data.return_value = df

        # Train the model
        model_path = await engine.train_model(tenant_id)

        assert model_path is not None
        assert str(tenant_id) in str(model_path)
        assert mock_dump.called


@pytest.mark.asyncio
async def test_train_model_no_valid_features(db_session):
    """Test training a model with no valid features."""
    engine = MLEngine(db_session)
    tenant_id = uuid.uuid4()

    with patch.object(engine, "get_training_data") as mock_get_data:
        # Mock the training data with columns that don't match our features
        df = pd.DataFrame(
            {
                "invalid_feature": np.random.randn(100),
            }
        )
        mock_get_data.return_value = df

        with pytest.raises(ValueError) as excinfo:
            await engine.train_model(tenant_id, features=["missing_feature"])

        assert "No valid features available for training" in str(excinfo.value)


@pytest.mark.asyncio
async def test_train_model_exception(db_session):
    """Test handling exceptions during model training."""
    engine = MLEngine(db_session)
    tenant_id = uuid.uuid4()

    with patch.object(engine, "get_training_data") as mock_get_data:
        mock_get_data.side_effect = RuntimeError("Simulated error")

        with pytest.raises(HTTPException) as excinfo:
            await engine.train_model(tenant_id)

        assert excinfo.value.status_code == 500
        assert "Error training model" in excinfo.value.detail


@pytest.mark.asyncio
async def test_load_model_exists(db_session, mock_model_data, test_model_dir):
    """Test loading an existing model."""
    engine = MLEngine(db_session)
    tenant_id = uuid.uuid4()

    with patch("auditpulse_mvp.ml_engine.ml_engine.MODEL_DIR", test_model_dir), patch(
        "joblib.load"
    ) as mock_load:

        # Create tenant directory and model file
        tenant_dir = test_model_dir / str(tenant_id)
        tenant_dir.mkdir(parents=True, exist_ok=True)
        model_path = tenant_dir / "20230101_120000.pkl"
        model_path.touch()

        mock_load.return_value = mock_model_data

        # Load the model
        model_data = await engine.load_model(tenant_id)

        assert model_data == mock_model_data
        mock_load.assert_called_once_with(model_path)


@pytest.mark.asyncio
async def test_load_model_not_exists_train_new(db_session, test_model_dir):
    """Test loading a model when none exists, triggering training."""
    engine = MLEngine(db_session)
    tenant_id = uuid.uuid4()

    with patch(
        "auditpulse_mvp.ml_engine.ml_engine.MODEL_DIR", test_model_dir
    ), patch.object(engine, "train_model") as mock_train, patch(
        "joblib.load"
    ) as mock_load:

        # Setup mock for training a new model
        mock_model_path = test_model_dir / str(tenant_id) / "20230101_120000.pkl"
        mock_train.return_value = mock_model_path

        # Setup mock for loading the newly trained model
        mock_load.return_value = {"model": MagicMock()}

        # Load the model (should trigger training)
        await engine.load_model(tenant_id)

        mock_train.assert_called_once_with(tenant_id)
        mock_load.assert_called_once_with(mock_model_path)


@pytest.mark.asyncio
async def test_load_model_not_exists_insufficient_data(db_session, test_model_dir):
    """Test loading a model when none exists and training fails."""
    engine = MLEngine(db_session)
    tenant_id = uuid.uuid4()

    with patch(
        "auditpulse_mvp.ml_engine.ml_engine.MODEL_DIR", test_model_dir
    ), patch.object(engine, "train_model") as mock_train:

        # Setup mock for failed training
        mock_train.side_effect = ValueError("Insufficient data")

        # Try to load the model (should raise ValueError)
        with pytest.raises(ValueError) as excinfo:
            await engine.load_model(tenant_id)

        assert "No model exists for tenant" in str(excinfo.value)
        mock_train.assert_called_once_with(tenant_id)


@pytest.mark.asyncio
async def test_extract_features_from_transaction(mock_transaction):
    """Test extracting features from a single transaction."""
    engine = MLEngine()
    features = ["amount", "day_of_week", "is_weekend"]

    df = engine._extract_features_from_transaction(mock_transaction, features)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    for feature in features:
        assert feature in df.columns


@pytest.mark.asyncio
async def test_score_success(db_session, mock_transaction, mock_model_data):
    """Test successfully scoring a transaction."""
    engine = MLEngine(db_session)
    tenant_id = mock_transaction.tenant_id

    with patch.object(engine, "load_model") as mock_load, patch.object(
        IsolationForest, "decision_function"
    ) as mock_decision:

        # Mock the model loading
        mock_load.return_value = mock_model_data

        # Mock the model decision function to return a normal value
        mock_decision.return_value = np.array([0.2])  # Positive = normal

        # Score the transaction
        score = await engine.score(tenant_id, mock_transaction)

        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(0.4)  # 0.5 - (0.2 / 2)
        mock_load.assert_called_once_with(tenant_id)


@pytest.mark.asyncio
async def test_score_anomaly(db_session, mock_anomalous_transaction, mock_model_data):
    """Test scoring an anomalous transaction."""
    engine = MLEngine(db_session)
    tenant_id = mock_anomalous_transaction.tenant_id

    with patch.object(engine, "load_model") as mock_load, patch.object(
        IsolationForest, "decision_function"
    ) as mock_decision:

        # Mock the model loading
        mock_load.return_value = mock_model_data

        # Mock the model decision function to return an anomalous value
        mock_decision.return_value = np.array([-0.5])  # Negative = anomaly

        # Score the transaction
        score = await engine.score(tenant_id, mock_anomalous_transaction)

        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(0.75)  # 0.5 - (-0.5 / 2)
        mock_load.assert_called_once_with(tenant_id)


@pytest.mark.asyncio
async def test_score_no_model(db_session, mock_transaction):
    """Test scoring when no model is available."""
    engine = MLEngine(db_session)
    tenant_id = mock_transaction.tenant_id

    with patch.object(engine, "load_model") as mock_load:
        # Mock load_model to raise ValueError (no model)
        mock_load.side_effect = ValueError("No model exists")

        # Score should return a fallback value
        score = await engine.score(tenant_id, mock_transaction)

        assert score == 0.5  # Fallback moderate score
        mock_load.assert_called_once_with(tenant_id)


@pytest.mark.asyncio
async def test_score_error(db_session, mock_transaction, mock_model_data):
    """Test handling errors during scoring."""
    engine = MLEngine(db_session)
    tenant_id = mock_transaction.tenant_id

    with patch.object(engine, "load_model") as mock_load, patch.object(
        engine, "_extract_features_from_transaction"
    ) as mock_extract:

        # Mock the model loading
        mock_load.return_value = mock_model_data

        # Mock feature extraction to raise an error
        mock_extract.side_effect = RuntimeError("Simulated error")

        # Score should return a fallback value
        score = await engine.score(tenant_id, mock_transaction)

        assert score == 0.5  # Fallback moderate score
        mock_load.assert_called_once_with(tenant_id)


@pytest.mark.asyncio
async def test_batch_train_models(db_session):
    """Test batch training models for multiple tenants."""
    engine = MLEngine(db_session)
    tenant_ids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]

    with patch.object(engine, "train_model") as mock_train:
        # Mock successful training for the first two tenants, error for the third
        mock_train.side_effect = [
            Path(f"models/{tenant_ids[0]}/20230101_120000.pkl"),
            Path(f"models/{tenant_ids[1]}/20230101_120000.pkl"),
            ValueError("Insufficient data"),
        ]

        # Batch train
        results = await engine.batch_train_models(tenant_ids)

        assert results["total"] == 3
        assert len(results["success"]) == 2
        assert len(results["error"]) == 1
        assert mock_train.call_count == 3


@pytest.mark.asyncio
async def test_batch_train_models_all_tenants(db_session):
    """Test batch training models for all active tenants."""
    engine = MLEngine(db_session)
    tenant_ids = [uuid.uuid4(), uuid.uuid4()]

    with patch.object(db_session, "execute") as mock_execute, patch.object(
        engine, "train_model"
    ) as mock_train:

        # Mock DB query to return active tenants
        mock_tenants = [
            Tenant(id=tenant_id, name=f"Tenant {i}", is_active=True)
            for i, tenant_id in enumerate(tenant_ids)
        ]
        mock_result = AsyncMock()
        mock_result.scalars.return_value.all.return_value = mock_tenants
        mock_execute.return_value = mock_result

        # Mock successful training
        mock_train.side_effect = [
            Path(f"models/{tenant_id}/20230101_120000.pkl") for tenant_id in tenant_ids
        ]

        # Batch train all tenants
        results = await engine.batch_train_models()

        assert results["total"] == 2
        assert len(results["success"]) == 2
        assert len(results["error"]) == 0
        assert mock_train.call_count == 2


def test_get_ml_engine():
    """Test the get_ml_engine dependency function."""
    engine = get_ml_engine()
    assert isinstance(engine, MLEngine)
