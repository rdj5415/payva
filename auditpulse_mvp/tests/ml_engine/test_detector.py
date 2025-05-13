"""Test suite for the ML anomaly detector module.

This module tests the integration between ML predictions and the rules engine
for comprehensive anomaly detection.
"""

import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, status

from auditpulse_mvp.database.models import (
    Anomaly,
    AnomalyType,
    DataSource,
    Transaction,
)
from auditpulse_mvp.ml_engine.detector import (
    MLAnomalyDetector,
    DEFAULT_ML_SCORE_THRESHOLD,
)
from auditpulse_mvp.rules_engine.rule_engine import RuleResult


@pytest.fixture
def mock_transaction():
    """Create a mock transaction for testing."""
    return MagicMock(
        spec=Transaction,
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        amount=Decimal("1000.00"),
        transaction_date=MagicMock(),
        source=DataSource.QUICKBOOKS,
        is_deleted=False,
    )


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = AsyncMock()

    # Mock the select execution for empty results
    execute_result = AsyncMock()
    execute_result.scalars.return_value.first.return_value = None
    session.execute.return_value = execute_result

    return session


@pytest.fixture
def mock_ml_engine():
    """Create a mock ML engine."""
    ml_engine = AsyncMock()

    # Default behavior: return a moderately anomalous score
    ml_engine.score_transaction.return_value = 0.6
    ml_engine.batch_score_transactions.return_value = {}

    return ml_engine


@pytest.fixture
def mock_rule_engine():
    """Create a mock rule engine."""
    rule_engine = AsyncMock()

    # Default behavior: no anomaly detected
    rule_engine.evaluate_transaction.return_value = RuleResult(
        anomaly_detected=False,
        rule_name=None,
        anomaly_type=None,
        description=None,
        confidence=0.0,
    )

    return rule_engine


@pytest.fixture
def detector(mock_db_session, mock_ml_engine, mock_rule_engine):
    """Create an MLAnomalyDetector with mocked dependencies."""
    detector = MLAnomalyDetector(db_session=mock_db_session)

    # Replace the engines with mocks
    detector.ml_engine = mock_ml_engine
    detector.rule_engine = mock_rule_engine

    return detector


@pytest.mark.asyncio
class TestMLAnomalyDetector:
    """Test the MLAnomalyDetector class."""

    async def test_process_transaction_no_anomaly(self, detector, mock_transaction):
        """Test processing a transaction with no anomaly."""
        # Set up mock ML engine to return a non-anomalous score
        detector.ml_engine.score_transaction.return_value = 0.3

        # Process the transaction
        ml_score, anomaly = await detector.process_transaction(mock_transaction)

        # Check results
        assert ml_score == 0.3
        assert anomaly is None

        # Verify ML engine was called
        detector.ml_engine.score_transaction.assert_called_once_with(
            mock_transaction.tenant_id, mock_transaction
        )

        # Verify rule engine was called since ML didn't detect anomaly
        detector.rule_engine.evaluate_transaction.assert_called_once_with(
            mock_transaction
        )

    async def test_process_transaction_ml_anomaly(
        self, detector, mock_transaction, mock_db_session
    ):
        """Test processing a transaction with ML-detected anomaly."""
        # Set up mock ML engine to return an anomalous score
        detector.ml_engine.score_transaction.return_value = 0.8

        # Set up mock DB session to create anomaly
        mock_anomaly = MagicMock(spec=Anomaly)
        mock_db_session.add = AsyncMock()
        mock_db_session.commit = AsyncMock()
        mock_db_session.refresh = AsyncMock()

        # Process the transaction
        ml_score, anomaly = await detector.process_transaction(mock_transaction)

        # Check results
        assert ml_score == 0.8
        assert anomaly is not None

        # Verify ML engine was called
        detector.ml_engine.score_transaction.assert_called_once_with(
            mock_transaction.tenant_id, mock_transaction
        )

        # Verify rule engine was NOT called since ML detected anomaly
        detector.rule_engine.evaluate_transaction.assert_not_called()

        # Verify anomaly was created
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()

    async def test_process_transaction_rule_anomaly(
        self, detector, mock_transaction, mock_db_session
    ):
        """Test processing a transaction with rule-detected anomaly."""
        # Set up mock ML engine to return a non-anomalous score
        detector.ml_engine.score_transaction.return_value = 0.3

        # Set up mock rule engine to detect an anomaly
        detector.rule_engine.evaluate_transaction.return_value = RuleResult(
            anomaly_detected=True,
            rule_name="TestRule",
            anomaly_type=AnomalyType.LARGE_AMOUNT,
            description="Test anomaly",
            confidence=0.9,
        )

        # Set up mock DB session to create anomaly
        mock_anomaly = MagicMock(spec=Anomaly)
        mock_db_session.add = AsyncMock()
        mock_db_session.commit = AsyncMock()
        mock_db_session.refresh = AsyncMock()

        # Process the transaction
        ml_score, anomaly = await detector.process_transaction(mock_transaction)

        # Check results
        assert ml_score == 0.3
        assert anomaly is not None

        # Verify ML engine was called
        detector.ml_engine.score_transaction.assert_called_once_with(
            mock_transaction.tenant_id, mock_transaction
        )

        # Verify rule engine was called
        detector.rule_engine.evaluate_transaction.assert_called_once_with(
            mock_transaction
        )

        # Verify anomaly was created
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()

    async def test_process_transaction_ml_error(
        self, detector, mock_transaction, mock_db_session
    ):
        """Test handling ML errors gracefully."""
        # Set up mock ML engine to raise an error
        detector.ml_engine.score_transaction.side_effect = Exception("Test error")

        # Set up mock rule engine to detect an anomaly
        detector.rule_engine.evaluate_transaction.return_value = RuleResult(
            anomaly_detected=True,
            rule_name="TestRule",
            anomaly_type=AnomalyType.LARGE_AMOUNT,
            description="Test anomaly",
            confidence=0.9,
        )

        # Process the transaction
        ml_score, anomaly = await detector.process_transaction(mock_transaction)

        # Check results - should fall back to rules with zero ML score
        assert ml_score == 0.0
        assert anomaly is not None

        # Verify rule engine was called despite ML error
        detector.rule_engine.evaluate_transaction.assert_called_once_with(
            mock_transaction
        )

    async def test_process_transaction_no_ml_model(
        self, detector, mock_transaction, mock_db_session
    ):
        """Test handling case when no ML model exists."""
        # Set up mock ML engine to raise 404 (no model found)
        detector.ml_engine.score_transaction.side_effect = HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No trained model found",
        )

        # Set up mock rule engine to detect an anomaly
        detector.rule_engine.evaluate_transaction.return_value = RuleResult(
            anomaly_detected=True,
            rule_name="TestRule",
            anomaly_type=AnomalyType.LARGE_AMOUNT,
            description="Test anomaly",
            confidence=0.9,
        )

        # Process the transaction
        ml_score, anomaly = await detector.process_transaction(mock_transaction)

        # Check results - should fall back to rules with zero ML score
        assert ml_score == 0.0
        assert anomaly is not None

        # Verify rule engine was called despite no ML model
        detector.rule_engine.evaluate_transaction.assert_called_once_with(
            mock_transaction
        )

    async def test_process_transaction_skip_rules(self, detector, mock_transaction):
        """Test processing a transaction with rules checking disabled."""
        # Set up mock ML engine to return a non-anomalous score
        detector.ml_engine.score_transaction.return_value = 0.3

        # Process the transaction with rules disabled
        ml_score, anomaly = await detector.process_transaction(
            mock_transaction, check_rules=False
        )

        # Check results
        assert ml_score == 0.3
        assert anomaly is None

        # Verify ML engine was called
        detector.ml_engine.score_transaction.assert_called_once_with(
            mock_transaction.tenant_id, mock_transaction
        )

        # Verify rule engine was NOT called because rules checking is disabled
        detector.rule_engine.evaluate_transaction.assert_not_called()

    async def test_create_anomaly_new(
        self, detector, mock_transaction, mock_db_session
    ):
        """Test creating a new anomaly."""
        # Set up mock DB session
        mock_db_session.add = AsyncMock()
        mock_db_session.commit = AsyncMock()
        mock_db_session.refresh = AsyncMock()

        # Create a new anomaly
        anomaly = await detector._create_anomaly(
            transaction=mock_transaction,
            ml_score=0.8,
            anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
            description="Test anomaly",
            confidence=0.9,
        )

        # Verify database operations
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()

    async def test_create_anomaly_existing(
        self, detector, mock_transaction, mock_db_session
    ):
        """Test updating an existing anomaly."""
        # Set up an existing anomaly
        existing_anomaly = MagicMock(spec=Anomaly)
        existing_anomaly.confidence = 0.5
        existing_anomaly.detection_metadata = {}

        # Set up mock DB session to return existing anomaly
        execute_result = AsyncMock()
        execute_result.scalars.return_value.first.return_value = existing_anomaly
        mock_db_session.execute.return_value = execute_result

        # Create/update anomaly with higher confidence
        anomaly = await detector._create_anomaly(
            transaction=mock_transaction,
            ml_score=0.8,
            anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
            description="Test anomaly",
            confidence=0.9,
            rule_name="TestRule",
        )

        # Verify result is the existing anomaly
        assert anomaly is existing_anomaly

        # Verify anomaly was updated
        assert existing_anomaly.anomaly_type == AnomalyType.STATISTICAL_OUTLIER
        assert existing_anomaly.description == "Test anomaly"
        assert existing_anomaly.confidence == 0.9
        assert existing_anomaly.ml_score == 0.8
        assert existing_anomaly.detection_metadata == {
            "rule_name": "TestRule",
            "ml_score": 0.8,
        }

        # Verify commit was called to save updates
        mock_db_session.commit.assert_called_once()

        # Verify add and refresh weren't called for existing anomaly
        mock_db_session.add.assert_not_called()
        mock_db_session.refresh.assert_not_called()

    async def test_create_anomaly_existing_lower_confidence(
        self, detector, mock_transaction, mock_db_session
    ):
        """Test not updating an existing anomaly when new confidence is lower."""
        # Set up an existing anomaly with high confidence
        existing_anomaly = MagicMock(spec=Anomaly)
        existing_anomaly.confidence = 0.9
        existing_anomaly.detection_metadata = {"rule_name": "BetterRule"}

        # Set up mock DB session to return existing anomaly
        execute_result = AsyncMock()
        execute_result.scalars.return_value.first.return_value = existing_anomaly
        mock_db_session.execute.return_value = execute_result

        # Try to update with lower confidence
        anomaly = await detector._create_anomaly(
            transaction=mock_transaction,
            ml_score=0.3,
            anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
            description="Lower confidence anomaly",
            confidence=0.6,
        )

        # Verify result is the existing anomaly
        assert anomaly is existing_anomaly

        # Verify anomaly was NOT updated
        assert existing_anomaly.confidence == 0.9
        assert existing_anomaly.detection_metadata == {"rule_name": "BetterRule"}

        # Verify commit was NOT called
        mock_db_session.commit.assert_not_called()

    async def test_process_transactions_batch(
        self, detector, mock_transaction, mock_db_session
    ):
        """Test batch processing of transactions."""
        # Create multiple transactions
        transaction1 = mock_transaction
        transaction2 = MagicMock(
            spec=Transaction,
            id=uuid.uuid4(),
            tenant_id=transaction1.tenant_id,  # Same tenant
            amount=Decimal("500.00"),
            transaction_date=MagicMock(),
            source=DataSource.QUICKBOOKS,
            is_deleted=False,
        )

        # Set up mock ML engine to return scores
        detector.ml_engine.batch_score_transactions.return_value = {
            transaction1.id: 0.8,  # Anomalous
            transaction2.id: 0.3,  # Normal
        }

        # Set up mock rule engine for the second transaction
        detector.rule_engine.evaluate_transaction.return_value = RuleResult(
            anomaly_detected=True,
            rule_name="TestRule",
            anomaly_type=AnomalyType.UNUSUAL_VENDOR,
            description="Unusual vendor detected",
            confidence=0.7,
        )

        # Set up mock DB session
        mock_db_session.add = AsyncMock()
        mock_db_session.commit = AsyncMock()
        mock_db_session.refresh = AsyncMock()

        # Process transactions
        results = await detector.process_transactions([transaction1, transaction2])

        # Check results
        assert len(results) == 2
        assert transaction1.id in results
        assert transaction2.id in results

        score1, anomaly1 = results[transaction1.id]
        score2, anomaly2 = results[transaction2.id]

        # First transaction: ML anomaly
        assert score1 == 0.8
        assert anomaly1 is not None

        # Second transaction: Rule anomaly
        assert score2 == 0.3
        assert anomaly2 is not None

        # Verify ML engine was called once for batch
        detector.ml_engine.batch_score_transactions.assert_called_once_with(
            transaction1.tenant_id, [transaction1, transaction2]
        )

        # Verify rule engine was called for second transaction only
        # (first had ML anomaly already)
        detector.rule_engine.evaluate_transaction.assert_called_once_with(transaction2)

        # Verify anomalies were created (2 adds, 2 commits, 2 refreshes)
        assert mock_db_session.add.call_count == 2
        assert mock_db_session.commit.call_count == 2
        assert mock_db_session.refresh.call_count == 2
