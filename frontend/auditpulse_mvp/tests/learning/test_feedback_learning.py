"""Test cases for the feedback learning module."""

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text

from auditpulse_mvp.database.models import (
    Anomaly,
    FeedbackType,
    AnomalyType,
    TenantConfiguration,
)
from auditpulse_mvp.learning.feedback_learning import (
    FeedbackLearner,
    update_thresholds_from_feedback,
    get_learning_statistics,
)
from auditpulse_mvp.api.api_v1.endpoints.config import (
    SensitivityConfiguration,
    SensitivityLevel,
)


# ---- Fixtures ----


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    mock_session = AsyncMock(spec=AsyncSession)
    return mock_session


@pytest.fixture
def sample_anomalies():
    """Create a sample list of anomalies with feedback."""
    tenant_id = uuid.uuid4()
    anomalies = []

    # Create some rule-based anomalies
    for i in range(10):
        is_false_positive = i < 6  # 60% false positives
        anomaly = MagicMock(spec=Anomaly)
        anomaly.id = uuid.uuid4()
        anomaly.tenant_id = tenant_id
        anomaly.anomaly_type = AnomalyType.RULES_BASED
        anomaly.feedback = (
            FeedbackType.FALSE_POSITIVE.value
            if is_false_positive
            else FeedbackType.TRUE_POSITIVE.value
        )
        anomaly.detection_metadata = {"rule_name": "Large Amount"}
        anomaly.created_at = datetime.utcnow() - timedelta(days=5)
        anomaly.updated_at = datetime.utcnow() - timedelta(days=2)
        anomalies.append(anomaly)

    # Create some ML-based anomalies
    for i in range(8):
        is_false_positive = i < 3  # ~37.5% false positives
        anomaly = MagicMock(spec=Anomaly)
        anomaly.id = uuid.uuid4()
        anomaly.tenant_id = tenant_id
        anomaly.anomaly_type = AnomalyType.ML_BASED
        anomaly.feedback = (
            FeedbackType.FALSE_POSITIVE.value
            if is_false_positive
            else FeedbackType.TRUE_POSITIVE.value
        )
        anomaly.detection_metadata = {"model_name": "isolation_forest"}
        anomaly.created_at = datetime.utcnow() - timedelta(days=10)
        anomaly.updated_at = datetime.utcnow() - timedelta(days=3)
        anomalies.append(anomaly)

    return anomalies


@pytest.fixture
def sample_sensitivity_config():
    """Create a sample sensitivity configuration."""
    config = SensitivityConfiguration(
        sensitivity_level=SensitivityLevel.MEDIUM,
        risk_engine={
            "ml_threshold": 0.8,
            "rules_threshold": 0.7,
            "ml_score_weight": 0.6,
            "rules_score_weight": 0.4,
        },
        rules={
            "large_transaction": {
                "enabled": True,
                "threshold": 10000,
                "weight": 0.8,
                "parameters": {},
            },
            "irregular_amount": {
                "enabled": True,
                "threshold": 0.9,
                "weight": 0.7,
                "parameters": {"std_dev_threshold": 2.5},
            },
            "weekend_transaction": {
                "enabled": True,
                "threshold": 0.6,
                "weight": 0.5,
                "parameters": {},
            },
            "round_number_transaction": {
                "enabled": True,
                "threshold": 0.7,
                "weight": 0.4,
                "parameters": {},
            },
        },
    )
    return config


# ---- Test FeedbackLearner class ----


class TestFeedbackLearner:
    """Test suite for the FeedbackLearner class."""

    @pytest.mark.asyncio
    async def test_gather_feedback_data(self, mock_db_session, sample_anomalies):
        """Test gathering feedback data."""
        # Arrange
        tenant_id = sample_anomalies[0].tenant_id
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = (
            sample_anomalies
        )

        # Act
        learner = FeedbackLearner(db_session=mock_db_session)
        result = await learner.gather_feedback_data(tenant_id, days=30)

        # Assert
        assert len(result["anomalies"]) == len(sample_anomalies)
        assert len(result["ml_anomalies"]) == 8
        assert len(result["rule_anomalies"]) == 10
        assert result["feedback_counts"] == {
            FeedbackType.TRUE_POSITIVE.value: 9,
            FeedbackType.FALSE_POSITIVE.value: 9,
            FeedbackType.IGNORE.value: 0,
        }
        assert result["rule_false_positives"]["Large Amount"] == 6

    def test_analyze_feedback(self, sample_anomalies):
        """Test analyzing feedback data."""
        # Arrange
        learner = FeedbackLearner()
        feedback_data = {
            "anomalies": sample_anomalies,
            "feedback_counts": {
                FeedbackType.TRUE_POSITIVE.value: 9,
                FeedbackType.FALSE_POSITIVE.value: 9,
                FeedbackType.IGNORE.value: 0,
            },
            "ml_anomalies": [
                a for a in sample_anomalies if a.anomaly_type == AnomalyType.ML_BASED
            ],
            "rule_anomalies": [
                a for a in sample_anomalies if a.anomaly_type == AnomalyType.RULES_BASED
            ],
            "rule_false_positives": {"Large Amount": 6},
        }

        # Act
        result = learner.analyze_feedback(feedback_data)

        # Assert
        assert result["total_feedback"] == 18
        assert result["false_positive_count"] == 9
        assert result["true_positive_count"] == 9
        assert result["false_positive_rate"] == 0.5
        assert result["true_positive_rate"] == 0.5
        assert result["ml_false_positive_rate"] == 0.375
        assert result["rule_false_positive_rate"] == 0.6
        assert "Large Amount" in result["rule_analysis"]
        assert result["rule_analysis"]["Large Amount"]["false_positive_rate"] == 0.6

    @pytest.mark.asyncio
    async def test_update_sensitivity_config_high_fp(
        self, mock_db_session, sample_sensitivity_config
    ):
        """Test updating sensitivity config with high false positive rate."""
        # Arrange
        tenant_id = uuid.uuid4()
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = (
            MagicMock(spec=TenantConfiguration)
        )

        with patch(
            "auditpulse_mvp.learning.feedback_learning.get_tenant_sensitivity_config",
            return_value=sample_sensitivity_config,
        ), patch(
            "auditpulse_mvp.learning.feedback_learning.get_preset_configuration",
            return_value=SensitivityConfiguration(
                sensitivity_level=SensitivityLevel.LOW,
                risk_engine={
                    "ml_threshold": 0.85,
                    "rules_threshold": 0.75,
                    "ml_score_weight": 0.5,
                    "rules_score_weight": 0.5,
                },
                rules={},
            ),
        ):
            learner = FeedbackLearner(db_session=mock_db_session)
            feedback_analysis = {
                "false_positive_rate": 0.75,  # High false positive rate
                "true_positive_rate": 0.25,
                "total_feedback": 20,
                "ml_false_positive_rate": 0.7,
                "rule_false_positive_rate": 0.8,
                "rule_analysis": {},
            }

            # Act
            result = await learner.update_sensitivity_config(
                tenant_id, feedback_analysis
            )

            # Assert
            assert result.sensitivity_level == SensitivityLevel.LOW
            assert mock_db_session.execute.call_count == 1
            assert mock_db_session.commit.call_count == 1

    @pytest.mark.asyncio
    async def test_update_sensitivity_config_low_fp(
        self, mock_db_session, sample_sensitivity_config
    ):
        """Test updating sensitivity config with low false positive rate."""
        # Arrange
        tenant_id = uuid.uuid4()
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = (
            MagicMock(spec=TenantConfiguration)
        )

        with patch(
            "auditpulse_mvp.learning.feedback_learning.get_tenant_sensitivity_config",
            return_value=sample_sensitivity_config,
        ), patch(
            "auditpulse_mvp.learning.feedback_learning.get_preset_configuration",
            return_value=SensitivityConfiguration(
                sensitivity_level=SensitivityLevel.HIGH,
                risk_engine={
                    "ml_threshold": 0.75,
                    "rules_threshold": 0.65,
                    "ml_score_weight": 0.7,
                    "rules_score_weight": 0.3,
                },
                rules={},
            ),
        ):
            learner = FeedbackLearner(db_session=mock_db_session)
            feedback_analysis = {
                "false_positive_rate": 0.15,  # Low false positive rate
                "true_positive_rate": 0.85,
                "total_feedback": 25,
                "ml_false_positive_rate": 0.2,
                "rule_false_positive_rate": 0.1,
                "rule_analysis": {},
            }

            # Act
            result = await learner.update_sensitivity_config(
                tenant_id, feedback_analysis
            )

            # Assert
            assert result.sensitivity_level == SensitivityLevel.HIGH
            assert mock_db_session.execute.call_count == 1
            assert mock_db_session.commit.call_count == 1

    @pytest.mark.asyncio
    async def test_adjust_rule_thresholds(self, sample_sensitivity_config):
        """Test adjusting rule thresholds."""
        # Arrange
        learner = FeedbackLearner()
        config = sample_sensitivity_config
        feedback_analysis = {
            "rule_analysis": {
                "Large Amount": {
                    "false_positive_rate": 0.8,
                    "total_count": 10,
                },
                "Weekend Transaction": {
                    "false_positive_rate": 0.9,
                    "total_count": 15,
                },
            },
        }

        # Act
        learner._adjust_rule_thresholds(config, feedback_analysis)

        # Assert
        assert config.rules.large_transaction.threshold == 12000  # Increased by 20%
        assert (
            config.rules.weekend_transaction.enabled is False
        )  # Disabled due to high FP rate

    @pytest.mark.asyncio
    async def test_process_recent_feedback_success(
        self, mock_db_session, sample_anomalies, sample_sensitivity_config
    ):
        """Test successful processing of recent feedback."""
        # Arrange
        tenant_id = uuid.uuid4()

        # Setup mocks
        learner = FeedbackLearner(db_session=mock_db_session)

        with patch.object(
            learner,
            "gather_feedback_data",
            return_value={"anomalies": sample_anomalies},
        ), patch.object(
            learner,
            "analyze_feedback",
            return_value={
                "false_positive_rate": 0.5,
                "true_positive_rate": 0.5,
                "total_feedback": 18,
            },
        ), patch.object(
            learner, "update_sensitivity_config", return_value=sample_sensitivity_config
        ), patch.object(
            learner.ml_engine,
            "train_model",
            return_value={
                "model_path": "/path/to/model.joblib",
                "training_samples": 1000,
                "anomaly_rate": 0.05,
            },
        ):
            # Act
            result = await learner.process_recent_feedback(tenant_id, days=30)

            # Assert
            assert result["status"] == "success"
            assert result["processed_count"] == len(sample_anomalies)
            assert result["false_positive_rate"] == 0.5
            assert result["model_retrained"] is True
            assert result["model_info"]["path"] == "/path/to/model.joblib"

    @pytest.mark.asyncio
    async def test_process_recent_feedback_no_data(self, mock_db_session):
        """Test processing with no feedback data."""
        # Arrange
        tenant_id = uuid.uuid4()

        # Setup mocks
        learner = FeedbackLearner(db_session=mock_db_session)

        with patch.object(
            learner, "gather_feedback_data", return_value={"anomalies": []}
        ):
            # Act
            result = await learner.process_recent_feedback(tenant_id, days=30)

            # Assert
            assert result["status"] == "skipped"
            assert result["reason"] == "No recent feedback found"
            assert result["processed_count"] == 0

    @pytest.mark.asyncio
    async def test_process_recent_feedback_model_error(
        self, mock_db_session, sample_anomalies, sample_sensitivity_config
    ):
        """Test handling model training error."""
        # Arrange
        tenant_id = uuid.uuid4()

        # Setup mocks
        learner = FeedbackLearner(db_session=mock_db_session)

        with patch.object(
            learner,
            "gather_feedback_data",
            return_value={"anomalies": sample_anomalies},
        ), patch.object(
            learner,
            "analyze_feedback",
            return_value={
                "false_positive_rate": 0.5,
                "true_positive_rate": 0.5,
                "total_feedback": 18,
            },
        ), patch.object(
            learner, "update_sensitivity_config", return_value=sample_sensitivity_config
        ), patch.object(
            learner.ml_engine,
            "train_model",
            side_effect=HTTPException(status_code=400, detail="Not enough data"),
        ):
            # Act
            result = await learner.process_recent_feedback(tenant_id, days=30)

            # Assert
            assert result["status"] == "success"
            assert result["model_retrained"] is False
            assert "error" in result["model_info"]


# ---- Test helper functions ----


@pytest.mark.asyncio
async def test_update_thresholds_from_feedback(mock_db_session):
    """Test updating thresholds based on feedback."""
    # Arrange
    tenant_id = uuid.uuid4()

    # Mock row objects for query results
    class MockRow:
        def __init__(self, rule_name, feedback, count):
            self.rule_name = rule_name
            self.feedback = feedback
            self.count = count

    # Setup rule query results (high false positive rate)
    rule_rows = [
        MockRow("Large Amount", FeedbackType.FALSE_POSITIVE.value, 8),
        MockRow("Large Amount", FeedbackType.TRUE_POSITIVE.value, 2),
    ]

    # Setup ML query results (low false positive rate)
    ml_rows = [
        MockRow("isolation_forest", FeedbackType.FALSE_POSITIVE.value, 1),
        MockRow("isolation_forest", FeedbackType.TRUE_POSITIVE.value, 9),
    ]

    # Configure mock execution results
    mock_db_session.execute.side_effect = [
        AsyncMock(return_value=rule_rows),
        AsyncMock(return_value=ml_rows),
    ]

    # Mock the update functions
    with patch(
        "auditpulse_mvp.learning.feedback_learning.update_rule_thresholds",
        return_value=None,
    ) as mock_update_rule, patch(
        "auditpulse_mvp.learning.feedback_learning.update_ml_model_parameters",
        return_value=None,
    ) as mock_update_ml:
        # Act
        result = await update_thresholds_from_feedback(
            tenant_id, mock_db_session, days_lookback=30, min_feedback_count=5
        )

        # Assert
        assert result["rules_updated"] == 1
        assert result["ml_models_updated"] == 1
        assert "Large Amount" in result["rule_adjustments"]
        assert "isolation_forest" in result["ml_adjustments"]
        assert mock_update_rule.call_count == 1
        assert mock_update_ml.call_count == 1


@pytest.mark.asyncio
async def test_get_learning_statistics(mock_db_session):
    """Test getting learning statistics."""
    # Arrange
    tenant_id = uuid.uuid4()

    # Mock row objects for query results
    class MockRow:
        def __init__(self, anomaly_type, feedback, count):
            self.anomaly_type = anomaly_type
            self.feedback = feedback
            self.count = count

    # Setup statistics query results
    stat_rows = [
        MockRow(AnomalyType.RULES_BASED, FeedbackType.FALSE_POSITIVE.value, 15),
        MockRow(AnomalyType.RULES_BASED, FeedbackType.TRUE_POSITIVE.value, 10),
        MockRow(AnomalyType.ML_BASED, FeedbackType.FALSE_POSITIVE.value, 5),
        MockRow(AnomalyType.ML_BASED, FeedbackType.TRUE_POSITIVE.value, 20),
    ]

    # Configure mock execution results
    mock_db_session.execute.return_value = stat_rows

    # Act
    stats = await get_learning_statistics(tenant_id, mock_db_session, days_lookback=90)

    # Assert
    assert stats["total_feedback"] == 50
    assert stats["rules_based"]["false_positive"] == 15
    assert stats["ml_based"]["true_positive"] == 20
    assert stats["rules_based"]["false_positive_rate"] == 0.6  # 15/25
    assert stats["ml_based"]["true_positive_rate"] == 0.8  # 20/25
