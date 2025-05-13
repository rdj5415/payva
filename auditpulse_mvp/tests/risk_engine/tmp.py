"""Tests for the Risk Engine module.

This module contains tests for the risk scoring decision layer,
covering risk assessment, configuration management, and fusion logic.
"""

import datetime
import uuid
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .database.models import Anomaly, AnomalyType, DataSource, Tenant, Transaction
from .risk_engine.risk_engine import (
    RiskConfig,
    RiskEngine,
    RiskLevel,
    RiskResult,
    RiskWeights,
    get_risk_engine,
)


@pytest.fixture
def mock_tenant_id() -> uuid.UUID:
    """Return a mock tenant ID."""
    return uuid.UUID("12345678-1234-5678-1234-567812345678")


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
def mock_flags() -> list[dict]:
    """Create mock flags for testing."""
    return [
        {
            "rule_name": "Large Amount",
            "rule_type": "large_amount",
            "score": 0.75,
            "description": "Transaction amount exceeds threshold",
            "weight": 1.0,
        }
    ]


@pytest.fixture
def mock_rules_engine():
    """Create a mock rules engine."""
    engine = AsyncMock()
    engine.flags.return_value = [
        {
            "rule_name": "Large Amount",
            "rule_type": "large_amount",
            "score": 0.75,
            "description": "Transaction amount exceeds threshold",
            "weight": 1.0,
        }
    ]
    engine.score.return_value = 0.75
    return engine


@pytest.fixture
def mock_ml_engine():
    """Create a mock ML engine."""
    engine = AsyncMock()
    engine.score.return_value = 0.6
    return engine


@pytest.fixture
def mock_gpt_engine():
    """Create a mock GPT engine."""
    engine = AsyncMock()
    engine.generate_explanation.return_value = "This is a test explanation."
    return engine


@pytest.fixture
def mock_tenant(mock_tenant_id):
    """Create a mock tenant."""
    return Tenant(
        id=mock_tenant_id,
        name="Test Tenant",
        slug="test-tenant",
        risk_settings={
            "weights": {
                "rules_weight": 50.0,
                "ml_weight": 30.0,
                "gpt_weight": 20.0,
                "low_threshold": 25.0,
                "medium_threshold": 50.0,
                "high_threshold": 75.0,
            },
            "gpt_confidence_threshold": 0.6,
            "min_rule_score": 0.4,
            "min_ml_score": 0.5,
            "baseline_risk": 5.0,
            "use_gpt": True,
            "use_ml": True,
        },
    )


@pytest.mark.asyncio
async def test_risk_weights_validation():
    """Test validation of risk weights."""
    # Valid weights
    weights = RiskWeights(
        rules_weight=40.0,
        ml_weight=40.0,
        gpt_weight=20.0,
        low_threshold=25.0,
        medium_threshold=50.0,
        high_threshold=75.0,
    )
    assert weights.rules_weight == 40.0
    assert weights.ml_weight == 40.0
    assert weights.gpt_weight == 20.0

    # Invalid weights sum
    with pytest.raises(ValidationError):
        RiskWeights(
            rules_weight=30.0,
            ml_weight=30.0,
            gpt_weight=30.0,  # Sum = 90, not 100
        )

    # Invalid thresholds
    with pytest.raises(ValidationError):
        RiskWeights(
            low_threshold=30.0,
            medium_threshold=20.0,  # Medium < Low
            high_threshold=75.0,
        )

    with pytest.raises(ValidationError):
        RiskWeights(
            low_threshold=25.0,
            medium_threshold=60.0,
            high_threshold=50.0,  # High < Medium
        )


@pytest.mark.asyncio
async def test_risk_engine_initialization(
    db_session, mock_rules_engine, mock_ml_engine, mock_gpt_engine
):
    """Test risk engine initialization."""
    engine = RiskEngine(
        db_session=db_session,
        rules_engine=mock_rules_engine,
        ml_engine=mock_ml_engine,
        gpt_engine=mock_gpt_engine,
    )

    assert engine.db_session == db_session
    assert engine.rules_engine == mock_rules_engine
    assert engine.ml_engine == mock_ml_engine
    assert engine.gpt_engine == mock_gpt_engine


@pytest.mark.asyncio
async def test_get_config_from_tenant(db_session, mock_tenant, mock_tenant_id):
    """Test getting risk configuration from a tenant."""
    engine = RiskEngine(db_session)

    # Mock the database query
    with patch.object(db_session, "execute") as mock_execute:
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = mock_tenant
        mock_execute.return_value = mock_result

        # Get the config
        config = await engine.get_config(mock_tenant_id)

        # Check the config
        assert config.tenant_id == mock_tenant_id
        assert config.weights.rules_weight == 50.0
        assert config.weights.ml_weight == 30.0
        assert config.weights.gpt_weight == 20.0
        assert config.gpt_confidence_threshold == 0.6
        assert config.min_rule_score == 0.4
        assert config.min_ml_score == 0.5
        assert config.baseline_risk == 5.0
        assert config.use_gpt is True
        assert config.use_ml is True

        # Check that the config is cached
        assert str(mock_tenant_id) in engine._tenant_configs


@pytest.mark.asyncio
async def test_get_config_tenant_not_found(db_session, mock_tenant_id):
    """Test getting default risk configuration when tenant not found."""
    engine = RiskEngine(db_session)

    # Mock the database query to return None
    with patch.object(db_session, "execute") as mock_execute:
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_execute.return_value = mock_result

        # Get the config
        config = await engine.get_config(mock_tenant_id)

        # Check the config - should be default values
        assert config.tenant_id == mock_tenant_id
        assert config.weights.rules_weight == 40.0  # Default value
        assert config.weights.ml_weight == 40.0  # Default value
        assert config.weights.gpt_weight == 20.0  # Default value


@pytest.mark.asyncio
async def test_update_config(db_session, mock_tenant_id):
    """Test updating risk configuration."""
    engine = RiskEngine(db_session)

    # Create a new config
    config = RiskConfig(
        tenant_id=mock_tenant_id,
        weights=RiskWeights(
            rules_weight=45.0,
            ml_weight=35.0,
            gpt_weight=20.0,
            low_threshold=20.0,
            medium_threshold=40.0,
            high_threshold=70.0,
        ),
        gpt_confidence_threshold=0.7,
        min_rule_score=0.3,
        min_ml_score=0.6,
        baseline_risk=10.0,
        use_gpt=False,
        use_ml=True,
    )

    # Mock the database execute and commit
    with (
        patch.object(db_session, "execute") as mock_execute,
        patch.object(db_session, "commit") as mock_commit,
    ):

        # Update the config
        success = await engine.update_config(config)

        # Check the result
        assert success is True
        mock_execute.assert_called_once()
        mock_commit.assert_called_once()

        # Check that the config is cached
        assert str(mock_tenant_id) in engine._tenant_configs
        assert engine._tenant_configs[str(mock_tenant_id)] == config


@pytest.mark.asyncio
async def test_evaluate_transaction_all_engines(
    db_session,
    mock_tenant_id,
    mock_transaction,
    mock_rules_engine,
    mock_ml_engine,
    mock_gpt_engine,
    mock_tenant,
):
    """Test evaluating a transaction with all engines enabled."""
    engine = RiskEngine(
        db_session=db_session,
        rules_engine=mock_rules_engine,
        ml_engine=mock_ml_engine,
        gpt_engine=mock_gpt_engine,
    )

    # Mock getting the config
    with patch.object(engine, "get_config") as mock_get_config:
        config = RiskConfig(
            tenant_id=mock_tenant_id,
            weights=RiskWeights(
                rules_weight=50.0,
                ml_weight=30.0,
                gpt_weight=20.0,
                low_threshold=25.0,
                medium_threshold=50.0,
                high_threshold=75.0,
            ),
            gpt_confidence_threshold=0.6,
            min_rule_score=0.4,
            min_ml_score=0.5,
            baseline_risk=5.0,
            use_gpt=True,
            use_ml=True,
        )
        mock_get_config.return_value = config

        # Evaluate the transaction
        result = await engine.evaluate_transaction(mock_transaction)

        # Expected score: (0.75 * 50) + (0.6 * 30) + (0.0 * 20) + 5.0 = 55.5
        # Final score is rules_score * rules_weight + ml_score * ml_weight + gpt_weight * 0 + baseline
        # Should be medium risk (above 50%)

        # Check the result
        assert isinstance(result, RiskResult)
        assert result.transaction_id == mock_transaction.id
        assert result.tenant_id == mock_transaction.tenant_id
        assert result.score == 55.5
        assert result.risk_level == RiskLevel.MEDIUM
        assert result.components["rules_score"] == 0.75
        assert result.components["ml_score"] == 0.6
        assert result.components["gpt_confidence"] == 0.0
        assert result.explanation == "This is a test explanation."
        assert result.flags == mock_rules_engine.flags.return_value


@pytest.mark.asyncio
async def test_evaluate_transaction_gpt_disabled(
    db_session,
    mock_tenant_id,
    mock_transaction,
    mock_rules_engine,
    mock_ml_engine,
    mock_gpt_engine,
):
    """Test evaluating a transaction with GPT disabled."""
    engine = RiskEngine(
        db_session=db_session,
        rules_engine=mock_rules_engine,
        ml_engine=mock_ml_engine,
        gpt_engine=mock_gpt_engine,
    )

    # Mock getting the config
    with patch.object(engine, "get_config") as mock_get_config:
        config = RiskConfig(
            tenant_id=mock_tenant_id,
            weights=RiskWeights(
                rules_weight=60.0,
                ml_weight=40.0,
                gpt_weight=0.0,  # GPT weight is 0
                low_threshold=25.0,
                medium_threshold=50.0,
                high_threshold=75.0,
            ),
            gpt_confidence_threshold=0.6,
            min_rule_score=0.4,
            min_ml_score=0.5,
            baseline_risk=5.0,
            use_gpt=False,  # GPT disabled
            use_ml=True,
        )
        mock_get_config.return_value = config

        # Evaluate the transaction
        result = await engine.evaluate_transaction(mock_transaction)

        # Expected score: (0.75 * 60) + (0.6 * 40) + 5.0 = 69.0
        # Should be medium risk (above 50% but below 75%)

        # Check the result
        assert result.score == 69.0
        assert result.risk_level == RiskLevel.MEDIUM
        assert result.components["rules_score"] == 0.75
        assert result.components["ml_score"] == 0.6
        assert result.components["gpt_confidence"] == 0.0
        assert (
            mock_gpt_engine.generate_explanation.call_count == 0
        )  # GPT explanation not called


@pytest.mark.asyncio
async def test_evaluate_transaction_ml_disabled(
    db_session,
    mock_tenant_id,
    mock_transaction,
    mock_rules_engine,
    mock_ml_engine,
    mock_gpt_engine,
):
    """Test evaluating a transaction with ML disabled."""
    engine = RiskEngine(
        db_session=db_session,
        rules_engine=mock_rules_engine,
        ml_engine=mock_ml_engine,
        gpt_engine=mock_gpt_engine,
    )

    # Mock getting the config
    with patch.object(engine, "get_config") as mock_get_config:
        config = RiskConfig(
            tenant_id=mock_tenant_id,
            weights=RiskWeights(
                rules_weight=80.0,
                ml_weight=0.0,  # ML weight is 0
                gpt_weight=20.0,
                low_threshold=25.0,
                medium_threshold=50.0,
                high_threshold=75.0,
            ),
            gpt_confidence_threshold=0.6,
            min_rule_score=0.4,
            min_ml_score=0.5,
            baseline_risk=5.0,
            use_gpt=True,
            use_ml=False,  # ML disabled
        )
        mock_get_config.return_value = config

        # Evaluate the transaction
        result = await engine.evaluate_transaction(mock_transaction)

        # Expected score: (0.75 * 80) + (0.0 * 0) + (0.0 * 20) + 5.0 = 65.0
        # Should be medium risk (above 50% but below 75%)

        # Check the result
        assert result.score == 65.0
        assert result.risk_level == RiskLevel.MEDIUM
        assert result.components["rules_score"] == 0.75
        assert result.components["ml_score"] == 0.0
        assert mock_ml_engine.score.call_count == 0  # ML scoring not called


@pytest.mark.asyncio
async def test_evaluate_transaction_high_risk(
    db_session,
    mock_tenant_id,
    mock_transaction,
    mock_rules_engine,
    mock_ml_engine,
    mock_gpt_engine,
):
    """Test evaluating a transaction that results in high risk."""
    engine = RiskEngine(
        db_session=db_session,
        rules_engine=mock_rules_engine,
        ml_engine=mock_ml_engine,
        gpt_engine=mock_gpt_engine,
    )

    # Override mock returns for higher scores
    mock_rules_engine.score.return_value = 0.9
    mock_ml_engine.score.return_value = 0.8

    # Mock getting the config
    with patch.object(engine, "get_config") as mock_get_config:
        config = RiskConfig(
            tenant_id=mock_tenant_id,
            weights=RiskWeights(
                rules_weight=50.0,
                ml_weight=30.0,
                gpt_weight=20.0,
                low_threshold=25.0,
                medium_threshold=50.0,
                high_threshold=75.0,
            ),
            gpt_confidence_threshold=0.6,
            min_rule_score=0.4,
            min_ml_score=0.5,
            baseline_risk=5.0,
            use_gpt=True,
            use_ml=True,
        )
        mock_get_config.return_value = config

        # Evaluate the transaction
        result = await engine.evaluate_transaction(mock_transaction)

        # Expected score: (0.9 * 50) + (0.8 * 30) + (0.0 * 20) + 5.0 = 74.0
        # Should be high risk (at or above 75%)

        # Check the result
        assert result.score == 74.0
        assert result.risk_level == RiskLevel.MEDIUM  # Just below high

        # Change the rules score to push it into high risk
        mock_rules_engine.score.return_value = 1.0

        # Re-evaluate
        result = await engine.evaluate_transaction(mock_transaction)

        # New expected score: (1.0 * 50) + (0.8 * 30) + (0.0 * 20) + 5.0 = 79.0
        assert result.score == 79.0
        assert result.risk_level == RiskLevel.HIGH


@pytest.mark.asyncio
async def test_get_risk_engine():
    """Test the get_risk_engine factory function."""
    db_session = AsyncMock()
    rules_engine = AsyncMock()
    ml_engine = AsyncMock()
    gpt_engine = AsyncMock()

    engine = get_risk_engine(
        db_session=db_session,
        rules_engine=rules_engine,
        ml_engine=ml_engine,
        gpt_engine=gpt_engine,
    )

    assert isinstance(engine, RiskEngine)
    assert engine.db_session == db_session
    assert engine.rules_engine == rules_engine
    assert engine.ml_engine == ml_engine
    assert engine.gpt_engine == gpt_engine
