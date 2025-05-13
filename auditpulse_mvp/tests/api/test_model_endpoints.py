"""Tests for model endpoints."""

import json
import pytest
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from auditpulse_mvp.main import app
from auditpulse_mvp.api.deps import get_current_user, get_db
from auditpulse_mvp.ml_engine.model_manager import ModelManager
from auditpulse_mvp.database.models import ModelVersion, ModelPerformance
from auditpulse_mvp.database.models.user import User


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_current_user():
    """Mock the current user."""
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.email = "test@example.com"
    user.is_superuser = False
    return user


@pytest.fixture
def mock_model_manager():
    """Mock the model manager."""
    manager = AsyncMock(spec=ModelManager)

    # Mock model version
    model_version = MagicMock(spec=ModelVersion)
    model_version.id = uuid.uuid4()
    model_version.model_type = "anomaly_detection"
    model_version.version = "1.0.0"
    model_version.metadata = {
        "algorithm": "random_forest",
        "features": ["amount", "time", "location"],
    }
    model_version.is_active = True
    model_version.created_at = datetime.now()
    model_version.activated_at = datetime.now()
    model_version.deactivated_at = None
    model_version.model_data = {"model_bytes": "base64_encoded_data"}

    # Mock inactive version
    inactive_version = MagicMock(spec=ModelVersion)
    inactive_version.id = uuid.uuid4()
    inactive_version.model_type = "anomaly_detection"
    inactive_version.version = "0.9.0"
    inactive_version.metadata = {
        "algorithm": "random_forest",
        "features": ["amount", "time"],
    }
    inactive_version.is_active = False
    inactive_version.created_at = datetime.now()
    inactive_version.activated_at = None
    inactive_version.deactivated_at = datetime.now()

    # Mock performance record
    performance = MagicMock(spec=ModelPerformance)
    performance.id = uuid.uuid4()
    performance.model_type = "anomaly_detection"
    performance.version = "1.0.0"
    performance.metrics = {
        "accuracy": 0.95,
        "precision": 0.92,
        "recall": 0.89,
        "f1_score": 0.91,
    }
    performance.dataset_size = 1000
    performance.evaluation_time = 5.23
    performance.recorded_at = datetime.now()

    # Set up method returns
    manager.create_version.return_value = model_version
    manager.get_active_version.return_value = model_version
    manager.get_version.return_value = model_version
    manager.list_versions.return_value = [model_version, inactive_version]
    manager.activate_version.return_value = model_version
    manager.rollback_version.return_value = inactive_version
    manager.record_performance.return_value = performance
    manager.get_performance_history.return_value = [performance]
    manager.get_performance_summary.return_value = {
        "avg_evaluation_time": 5.23,
        "avg_dataset_size": 1000,
        "evaluation_count": 1,
        "latest_metrics": performance.metrics,
    }

    return manager


# Override dependencies for testing
app.dependency_overrides[get_current_user] = lambda: mock_current_user()
app.dependency_overrides[ModelManager] = lambda: mock_model_manager()


def test_create_model_version(client, mock_model_manager):
    """Test creating a model version."""
    # Test request data
    data = {
        "model_type": "anomaly_detection",
        "model_data": {
            "model_bytes": "base64_encoded_data",
            "feature_importances": {"amount": 0.8, "time": 0.15, "location": 0.05},
        },
        "metadata": {
            "algorithm": "random_forest",
            "features": ["amount", "time", "location"],
            "hyperparameters": {"n_estimators": 100, "max_depth": 10},
        },
        "is_active": True,
    }

    # Send request
    response = client.post("/api/v1/models/versions", json=data)

    # Verify response
    assert response.status_code == 201
    result = response.json()
    assert result["model_type"] == "anomaly_detection"
    assert result["version"] == "1.0.0"
    assert result["is_active"] == True

    # Verify model manager was called
    mock_model_manager.create_version.assert_called_once()


def test_get_active_version(client, mock_model_manager):
    """Test getting active model version."""
    # Send request
    response = client.get("/api/v1/models/versions/anomaly_detection/active")

    # Verify response
    assert response.status_code == 200
    result = response.json()
    assert result["model_type"] == "anomaly_detection"
    assert result["is_active"] == True

    # Verify model manager was called
    mock_model_manager.get_active_version.assert_called_once()


def test_list_versions(client, mock_model_manager):
    """Test listing model versions."""
    # Send request
    response = client.get("/api/v1/models/versions/anomaly_detection")

    # Verify response
    assert response.status_code == 200
    result = response.json()
    assert len(result) == 2
    assert result[0]["model_type"] == "anomaly_detection"

    # Verify model manager was called
    mock_model_manager.list_versions.assert_called_once()


def test_activate_version(client, mock_model_manager):
    """Test activating a model version."""
    # Send request
    response = client.post("/api/v1/models/versions/anomaly_detection/1.0.0/activate")

    # Verify response
    assert response.status_code == 200
    result = response.json()
    assert result["model_type"] == "anomaly_detection"
    assert result["version"] == "1.0.0"
    assert result["is_active"] == True

    # Verify model manager was called
    mock_model_manager.activate_version.assert_called_once()


def test_rollback_version(client, mock_model_manager):
    """Test rolling back to a previous model version."""
    # Send request
    response = client.post("/api/v1/models/versions/anomaly_detection/0.9.0/rollback")

    # Verify response
    assert response.status_code == 200
    result = response.json()
    assert result["model_type"] == "anomaly_detection"
    assert result["version"] == "0.9.0"

    # Verify model manager was called
    mock_model_manager.rollback_version.assert_called_once()


def test_record_performance(client, mock_model_manager):
    """Test recording model performance."""
    # Test request data
    data = {
        "model_type": "anomaly_detection",
        "version": "1.0.0",
        "metrics": {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.89,
            "f1_score": 0.91,
            "roc_auc": 0.97,
        },
        "dataset_size": 1000,
        "evaluation_time": 5.23,
    }

    # Send request
    response = client.post("/api/v1/models/performance", json=data)

    # Verify response
    assert response.status_code == 201
    result = response.json()
    assert result["model_type"] == "anomaly_detection"
    assert result["version"] == "1.0.0"
    assert "metrics" in result

    # Verify model manager was called
    mock_model_manager.record_performance.assert_called_once()


def test_get_performance_history(client, mock_model_manager):
    """Test getting performance history."""
    # Send request
    response = client.get("/api/v1/models/performance/anomaly_detection?limit=10")

    # Verify response
    assert response.status_code == 200
    result = response.json()
    assert len(result) == 1
    assert result[0]["model_type"] == "anomaly_detection"
    assert "metrics" in result[0]

    # Verify model manager was called
    mock_model_manager.get_performance_history.assert_called_once()


def test_get_performance_summary(client, mock_model_manager):
    """Test getting performance summary."""
    # Send request
    response = client.get("/api/v1/models/performance/anomaly_detection/summary")

    # Verify response
    assert response.status_code == 200
    result = response.json()
    assert "avg_evaluation_time" in result
    assert "latest_metrics" in result

    # Verify model manager was called
    mock_model_manager.get_performance_summary.assert_called_once()


def test_validate_model(client, mock_model_manager):
    """Test validating a model."""
    # Test request data
    data = {
        "model_type": "anomaly_detection",
        "version": "1.0.0",
        "validation_data": [
            {"amount": 100, "time": "2023-01-01T12:00:00", "location": "New York"},
            {"amount": 500, "time": "2023-01-02T15:30:00", "location": "Los Angeles"},
            {"amount": 50, "time": "2023-01-03T09:15:00", "location": "Chicago"},
        ],
        "ground_truth": [0, 1, 0],
    }

    # Send request
    with (
        patch("sklearn.metrics.accuracy_score", return_value=0.95),
        patch("sklearn.metrics.precision_score", return_value=0.92),
        patch("sklearn.metrics.recall_score", return_value=0.89),
        patch("sklearn.metrics.f1_score", return_value=0.91),
    ):
        response = client.post("/api/v1/models/validate", json=data)

    # Verify response
    assert response.status_code == 200
    result = response.json()
    assert result["model_type"] == "anomaly_detection"
    assert result["version"] == "1.0.0"
    assert "metrics" in result
    assert "validation_size" in result
    assert result["validation_success"] == True

    # Verify model manager was called for verification
    mock_model_manager.get_version.assert_called_once()
    # Verify record performance was called with metrics
    mock_model_manager.record_performance.assert_called_once()


def test_predict(client, mock_model_manager):
    """Test making a prediction."""
    # Test request data
    data = {
        "model_type": "anomaly_detection",
        "data": {
            "amount": 1000,
            "time": "2023-01-04T18:45:00",
            "location": "Miami",
            "device": "mobile",
        },
    }

    # Send request
    with (
        patch("random.choice", return_value=1),
        patch("random.uniform", return_value=0.92),
    ):
        response = client.post("/api/v1/models/predict", json=data)

    # Verify response
    assert response.status_code == 200
    result = response.json()
    assert result["model_type"] == "anomaly_detection"
    assert result["version"] == "1.0.0"
    assert "prediction" in result
    assert "confidence" in result
    assert "execution_time_ms" in result
    assert "metadata" in result

    # Verify model manager was called for active version
    mock_model_manager.get_active_version.assert_called_once()


def test_model_health_check(client, mock_model_manager):
    """Test model health check."""
    # Send request
    response = client.get("/api/v1/models/health/anomaly_detection")

    # Verify response
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "healthy"
    assert result["model_type"] == "anomaly_detection"
    assert result["version"] == "1.0.0"
    assert "created_at" in result
    assert "performance_metrics" in result

    # Verify model manager was called
    mock_model_manager.get_active_version.assert_called_once()
    mock_model_manager.get_performance_history.assert_called_once()


def test_predict_with_invalid_model_type(client, mock_model_manager):
    """Test prediction with invalid model type."""
    # Mock get_active_version to return None
    mock_model_manager.get_active_version.return_value = None

    # Test request data
    data = {"model_type": "invalid_model", "data": {"amount": 100}}

    # Send request
    response = client.post("/api/v1/models/predict", json=data)

    # Verify response
    assert response.status_code == 404
    result = response.json()
    assert "detail" in result
    assert "No active version found" in result["detail"]


def test_validate_model_invalid_data(client, mock_model_manager):
    """Test validating a model with invalid data."""
    # Test request data with mismatched lengths
    data = {
        "model_type": "anomaly_detection",
        "version": "1.0.0",
        "validation_data": [{"amount": 100}, {"amount": 500}],
        "ground_truth": [0],  # Only one value
    }

    # Send request
    response = client.post("/api/v1/models/validate", json=data)

    # Verify response
    assert response.status_code == 400
    result = response.json()
    assert "detail" in result
    assert "same length" in result["detail"]


def test_create_model_version_invalid_type(client, mock_model_manager):
    """Test creating a model version with invalid model type."""
    # Test request data
    data = {
        "model_type": "invalid_type",
        "model_data": {"model_bytes": "data"},
        "is_active": False,
    }

    # Send request
    response = client.post("/api/v1/models/versions", json=data)

    # Verify response
    assert response.status_code == 422  # Validation error
    result = response.json()
    assert "detail" in result
