"""Tests for model manager."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from auditpulse_mvp.ml_engine.model_manager import ModelManager
from auditpulse_mvp.database.models import ModelVersion, ModelPerformance


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    return settings


@pytest.fixture
def model_manager(mock_db_session, mock_settings):
    """Create a model manager instance."""
    return ModelManager(db_session=mock_db_session, settings=mock_settings)


@pytest.mark.asyncio
async def test_create_version(model_manager, mock_db_session):
    """Test creating a new model version."""
    # Mock database query result
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db_session.execute.return_value = mock_result

    # Create version
    version = await model_manager.create_version(
        model_type="test_model",
        model_data={"weights": [1, 2, 3]},
        metadata={"accuracy": 0.95},
        is_active=True,
    )

    # Verify version creation
    assert version.model_type == "test_model"
    assert version.version == "1.0.0"
    assert version.model_data == {"weights": [1, 2, 3]}
    assert version.metadata == {"accuracy": 0.95}
    assert version.is_active is True

    # Verify database operations
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()
    mock_db_session.refresh.assert_called_once()


@pytest.mark.asyncio
async def test_get_active_version(model_manager, mock_db_session):
    """Test getting active model version."""
    # Mock active version
    active_version = ModelVersion(
        model_type="test_model",
        version="1.0.0",
        model_data={},
        metadata={},
        is_active=True,
    )

    # Mock database query result
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = active_version
    mock_db_session.execute.return_value = mock_result

    # Get active version
    version = await model_manager.get_active_version("test_model")

    # Verify version
    assert version == active_version
    assert version.is_active is True


@pytest.mark.asyncio
async def test_activate_version(model_manager, mock_db_session):
    """Test activating a model version."""
    # Mock version to activate
    version = ModelVersion(
        model_type="test_model",
        version="1.0.0",
        model_data={},
        metadata={},
        is_active=False,
    )

    # Mock database query results
    mock_version_result = MagicMock()
    mock_version_result.scalar_one_or_none.return_value = version

    mock_current_result = MagicMock()
    mock_current_result.scalar_one_or_none.return_value = None

    mock_db_session.execute.side_effect = [mock_version_result, mock_current_result]

    # Activate version
    activated = await model_manager.activate_version("test_model", "1.0.0")

    # Verify activation
    assert activated.is_active is True
    assert activated.activated_at is not None

    # Verify database operations
    assert mock_db_session.commit.call_count == 1
    assert mock_db_session.refresh.call_count == 1


@pytest.mark.asyncio
async def test_record_performance(model_manager, mock_db_session):
    """Test recording model performance."""
    # Record performance
    performance = await model_manager.record_performance(
        model_type="test_model",
        version="1.0.0",
        metrics={"accuracy": 0.95, "f1_score": 0.92},
        dataset_size=1000,
        evaluation_time=1.5,
    )

    # Verify performance record
    assert performance.model_type == "test_model"
    assert performance.version == "1.0.0"
    assert performance.metrics == {"accuracy": 0.95, "f1_score": 0.92}
    assert performance.dataset_size == 1000
    assert performance.evaluation_time == 1.5
    assert performance.recorded_at is not None

    # Verify database operations
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()
    mock_db_session.refresh.assert_called_once()


@pytest.mark.asyncio
async def test_get_performance_history(model_manager, mock_db_session):
    """Test getting performance history."""
    # Mock performance records
    records = [
        ModelPerformance(
            model_type="test_model",
            version="1.0.0",
            metrics={"accuracy": 0.95},
            dataset_size=1000,
            evaluation_time=1.5,
            recorded_at=datetime.now(),
        ),
        ModelPerformance(
            model_type="test_model",
            version="1.0.0",
            metrics={"accuracy": 0.96},
            dataset_size=1000,
            evaluation_time=1.4,
            recorded_at=datetime.now(),
        ),
    ]

    # Mock database query result
    mock_result = MagicMock()
    mock_result.scalars.return_value = records
    mock_db_session.execute.return_value = mock_result

    # Get performance history
    history = await model_manager.get_performance_history(
        model_type="test_model",
        version="1.0.0",
        limit=10,
    )

    # Verify history
    assert len(history) == 2
    assert all(p.model_type == "test_model" for p in history)
    assert all(p.version == "1.0.0" for p in history)


@pytest.mark.asyncio
async def test_get_performance_summary(model_manager, mock_db_session):
    """Test getting performance summary."""
    # Mock summary data
    mock_summary = MagicMock()
    mock_summary.avg_evaluation_time = 1.45
    mock_summary.avg_dataset_size = 1000.0
    mock_summary.evaluation_count = 2

    # Mock database query results
    mock_summary_result = MagicMock()
    mock_summary_result.first.return_value = mock_summary

    mock_history_result = MagicMock()
    mock_history_result.scalars.return_value = [
        ModelPerformance(
            model_type="test_model",
            version="1.0.0",
            metrics={"accuracy": 0.96},
            dataset_size=1000,
            evaluation_time=1.4,
            recorded_at=datetime.now(),
        )
    ]

    mock_db_session.execute.side_effect = [mock_summary_result, mock_history_result]

    # Get performance summary
    summary = await model_manager.get_performance_summary(
        model_type="test_model",
        version="1.0.0",
    )

    # Verify summary
    assert summary["avg_evaluation_time"] == 1.45
    assert summary["avg_dataset_size"] == 1000.0
    assert summary["evaluation_count"] == 2
    assert summary["latest_metrics"] == {"accuracy": 0.96}
