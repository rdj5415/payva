"""Integration tests for the feedback learning system.

This module tests the end-to-end functionality of the feedback learning system,
including the interaction between feedback processing, rule updates, and model retraining.
"""
import asyncio
from uuid import uuid4
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.models import Anomaly, Tenant, SensitivityConfig, FeedbackType, AnomalyType
from auditpulse_mvp.api.api_v1.endpoints.anomalies import AnomalyFeedback
from auditpulse_mvp.learning.feedback_learning import FeedbackLearner, update_thresholds_from_feedback
from auditpulse_mvp.learning.scheduler import FeedbackLearningScheduler
from auditpulse_mvp.ml.models import update_ml_model_parameters
from auditpulse_mvp.rules.rule_engine import update_rule_thresholds


@pytest.fixture
async def test_tenant():
    """Create a test tenant."""
    return Tenant(
        id=str(uuid4()),
        name="Test Tenant",
        description="Test tenant for integration tests",
    )


@pytest.fixture
async def test_sensitivity_config(test_tenant):
    """Create a test sensitivity configuration."""
    return SensitivityConfig(
        id=str(uuid4()),
        tenant_id=test_tenant.id,
        rule_type="amount_threshold",
        sensitivity=0.5,  # Starting with medium sensitivity
        last_updated=datetime.utcnow() - timedelta(days=7),
    )


@pytest.fixture
async def test_anomalies(test_tenant):
    """Create test anomalies with various feedback states."""
    base_time = datetime.utcnow() - timedelta(days=2)
    
    # Create 10 anomalies with various feedback states
    anomalies = []
    
    # 4 confirmed anomalies (true positives)
    for i in range(4):
        anomalies.append(
            Anomaly(
                id=str(uuid4()),
                tenant_id=test_tenant.id,
                transaction_id=f"tx-{uuid4()}",
                anomaly_type="amount_threshold",
                description="Unusually large transaction amount",
                confidence=0.85,
                risk_score=0.75,
                feedback="confirm",
                feedback_timestamp=base_time + timedelta(hours=i),
                created_at=base_time - timedelta(hours=i),
            )
        )
    
    # 3 dismissed anomalies (false positives)
    for i in range(3):
        anomalies.append(
            Anomaly(
                id=str(uuid4()),
                tenant_id=test_tenant.id,
                transaction_id=f"tx-{uuid4()}",
                anomaly_type="amount_threshold",
                description="Unusually large transaction amount",
                confidence=0.65,
                risk_score=0.55,
                feedback="dismiss",
                feedback_timestamp=base_time + timedelta(hours=i),
                created_at=base_time - timedelta(hours=i),
            )
        )
    
    # 3 anomalies without feedback
    for i in range(3):
        anomalies.append(
            Anomaly(
                id=str(uuid4()),
                tenant_id=test_tenant.id,
                transaction_id=f"tx-{uuid4()}",
                anomaly_type="amount_threshold",
                description="Unusually large transaction amount",
                confidence=0.70,
                risk_score=0.60,
                feedback=None,
                created_at=base_time - timedelta(hours=i),
            )
        )
    
    return anomalies


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.fixture
def test_tenant_id():
    """Generate a test tenant ID."""
    return uuid4()


@pytest.fixture
def sample_anomalies(test_tenant_id):
    """Create sample anomalies with feedback."""
    # Create rule-based anomalies
    rule_anomalies = []
    for i in range(10):
        is_false_positive = i < 7  # 70% false positives
        anomaly = MagicMock(spec=Anomaly)
        anomaly.id = uuid4()
        anomaly.tenant_id = test_tenant_id
        anomaly.anomaly_type = AnomalyType.RULES_BASED
        anomaly.feedback = (
            FeedbackType.FALSE_POSITIVE.value if is_false_positive
            else FeedbackType.TRUE_POSITIVE.value
        )
        anomaly.detection_metadata = {"rule_name": "Large Amount"}
        rule_anomalies.append(anomaly)
    
    # Create ML-based anomalies
    ml_anomalies = []
    for i in range(15):
        is_false_positive = i < 5  # 33% false positives
        anomaly = MagicMock(spec=Anomaly)
        anomaly.id = uuid4()
        anomaly.tenant_id = test_tenant_id
        anomaly.anomaly_type = AnomalyType.ML_BASED
        anomaly.feedback = (
            FeedbackType.FALSE_POSITIVE.value if is_false_positive
            else FeedbackType.TRUE_POSITIVE.value
        )
        anomaly.detection_metadata = {"model_name": "isolation_forest"}
        ml_anomalies.append(anomaly)
    
    return rule_anomalies + ml_anomalies


@pytest.mark.asyncio
class TestFeedbackLearningIntegration:
    """Integration tests for the feedback learning system."""
    
    async def test_end_to_end_learning_process(self, mock_db_session, test_tenant_id, sample_anomalies):
        """Test the complete feedback learning process from anomaly feedback to model updates."""
        # Setup mock database responses
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = sample_anomalies
        
        # Mock the rule and ML model update functions
        with patch("auditpulse_mvp.learning.feedback_learning.update_rule_thresholds") as mock_update_rules, \
             patch("auditpulse_mvp.learning.feedback_learning.update_ml_model_parameters") as mock_update_ml:
            
            # Execute the update process
            results = await update_thresholds_from_feedback(
                test_tenant_id, mock_db_session, days_lookback=30, min_feedback_count=5
            )
            
            # Verify that the update functions were called
            assert mock_update_rules.call_count > 0
            assert mock_update_ml.call_count > 0
            
            # Verify the results contain the expected data
            assert results["rules_updated"] > 0
            assert results["ml_models_updated"] > 0
            assert "rule_adjustments" in results
            assert "ml_adjustments" in results
    
    async def test_learner_process_updates_sensitivity(self, mock_db_session, test_tenant_id, sample_anomalies):
        """Test that the FeedbackLearner updates sensitivity configuration based on feedback."""
        # Setup mock database responses
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = sample_anomalies
        
        # Create a learner instance
        learner = FeedbackLearner(db_session=mock_db_session)
        
        # Mock the update_sensitivity_config method to track calls
        original_update_config = learner.update_sensitivity_config
        config_updated = False
        
        async def mock_update_sensitivity(*args, **kwargs):
            nonlocal config_updated
            config_updated = True
            return await original_update_config(*args, **kwargs)
        
        with patch.object(learner, "update_sensitivity_config", side_effect=mock_update_sensitivity), \
             patch.object(learner.ml_engine, "train_model", return_value={"status": "success"}):
            
            # Process the feedback
            await learner.process_recent_feedback(test_tenant_id)
            
            # Verify that the sensitivity config was updated
            assert config_updated is True
    
    async def test_scheduler_runs_feedback_learning(self, mock_db_session, test_tenant_id):
        """Test that the scheduler runs feedback learning for all tenants."""
        # Mock the get_all_tenant_ids function
        tenant_ids = [test_tenant_id, uuid4()]
        
        # Mock the feedback learner
        mock_learner = AsyncMock()
        mock_learner.process_recent_feedback.return_value = {"status": "success"}
        
        with patch("auditpulse_mvp.learning.scheduler.get_all_tenant_ids", return_value=tenant_ids), \
             patch("auditpulse_mvp.learning.scheduler.get_feedback_learner", return_value=mock_learner):
            
            # Create a scheduler instance
            scheduler = FeedbackLearningScheduler(db_session=mock_db_session)
            
            # Run the feedback learning process
            await scheduler.run_feedback_learning()
            
            # Verify that the learner was called for each tenant
            assert mock_learner.process_recent_feedback.call_count == len(tenant_ids)
            for tenant_id in tenant_ids:
                mock_learner.process_recent_feedback.assert_any_call(tenant_id, days=30)
    
    async def test_rule_threshold_update_integration(self, mock_db_session, test_tenant_id):
        """Test that rule thresholds are updated correctly based on feedback."""
        # Mock the rule update function
        with patch("auditpulse_mvp.rules.rule_engine.update_rule_thresholds", return_value=None) as mock_update:
            # Call the actual function
            adjustment = 0.2
            rule_name = "Large Amount"
            
            await update_rule_thresholds(rule_name, adjustment, test_tenant_id, mock_db_session)
            
            # Verify the mock was called with the correct parameters
            mock_update.assert_called_once_with(rule_name, adjustment, test_tenant_id, mock_db_session)
    
    async def test_ml_model_update_integration(self, test_tenant_id):
        """Test that ML model parameters are updated correctly based on feedback."""
        # Mock the ML model update function
        params = {
            "threshold": 0.15,
            "regularization": 0.05,
        }
        model_name = "isolation_forest"
        
        with patch("auditpulse_mvp.ml.models.update_ml_model_parameters", return_value=None) as mock_update:
            # Call the actual function
            await update_ml_model_parameters(model_name, params, test_tenant_id)
            
            # Verify the mock was called with the correct parameters
            mock_update.assert_called_once_with(model_name, params, test_tenant_id)


@pytest.mark.asyncio
async def test_feedback_learning_with_real_db(mock_db_session, test_tenant_id):
    """
    Test the feedback learning system with a more realistic database setup.
    
    This test creates a more complex scenario to test the interaction between different
    components of the feedback learning system.
    """
    # Mock Tenant and Anomaly data
    tenants = [MagicMock(id=test_tenant_id), MagicMock(id=uuid4())]
    
    # Create a mix of anomalies with different feedback types
    anomalies = []
    
    # Add rule-based anomalies with high false positive rate for tenant 1
    for i in range(15):
        anomaly = MagicMock(spec=Anomaly)
        anomaly.id = uuid4()
        anomaly.tenant_id = test_tenant_id
        anomaly.anomaly_type = AnomalyType.RULES_BASED
        anomaly.feedback = FeedbackType.FALSE_POSITIVE.value if i < 12 else FeedbackType.TRUE_POSITIVE.value
        anomaly.detection_metadata = {"rule_name": "Large Amount" if i < 10 else "Weekend Transaction"}
        anomalies.append(anomaly)
    
    # Add ML-based anomalies with mixed feedback for tenant 1
    for i in range(20):
        anomaly = MagicMock(spec=Anomaly)
        anomaly.id = uuid4()
        anomaly.tenant_id = test_tenant_id
        anomaly.anomaly_type = AnomalyType.ML_BASED
        anomaly.feedback = FeedbackType.FALSE_POSITIVE.value if i < 8 else FeedbackType.TRUE_POSITIVE.value
        anomaly.detection_metadata = {"model_name": "isolation_forest"}
        anomalies.append(anomaly)
    
    # Mock the database queries to return our test data
    mock_db_session.execute.side_effect = [
        # First call: get_all_tenant_ids
        MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=tenants)))),
        # Second call: gather_feedback_data
        MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=anomalies)))),
        # Additional calls as needed
        MagicMock(),
        MagicMock(),
        MagicMock(),
    ]
    
    # Mock the update functions to track how they're called
    with patch("auditpulse_mvp.learning.feedback_learning.update_rule_thresholds") as mock_update_rules, \
         patch("auditpulse_mvp.learning.feedback_learning.update_ml_model_parameters") as mock_update_ml, \
         patch("auditpulse_mvp.api.api_v1.endpoints.config.get_tenant_sensitivity_config") as mock_get_config, \
         patch("auditpulse_mvp.api.api_v1.endpoints.config.get_preset_configuration") as mock_get_preset:
        
        # Mock the sensitivity configuration
        mock_config = MagicMock()
        mock_config.sensitivity_level = "high"
        mock_config.dict.return_value = {
            "sensitivity_level": "high",
            "risk_engine": {
                "ml_threshold": 0.7,
                "rules_threshold": 0.6,
                "ml_score_weight": 0.6,
                "rules_score_weight": 0.4,
            },
            "rules": {
                "large_transaction": {
                    "enabled": True,
                    "threshold": 10000.0,
                    "weight": 0.8,
                    "parameters": {},
                },
                "weekend_transaction": {
                    "enabled": True,
                    "threshold": 0.7,
                    "weight": 0.5,
                    "parameters": {},
                },
            },
        }
        mock_get_config.return_value = mock_config
        
        # Mock the preset configuration
        mock_preset = MagicMock()
        mock_preset.sensitivity_level = "medium"
        mock_preset.dict.return_value = {
            "sensitivity_level": "medium",
            "risk_engine": {
                "ml_threshold": 0.75,
                "rules_threshold": 0.65,
                "ml_score_weight": 0.5,
                "rules_score_weight": 0.5,
            },
            "rules": {},
        }
        mock_get_preset.return_value = mock_preset
        
        # Create and start the scheduler
        scheduler = FeedbackLearningScheduler(db_session=mock_db_session)
        
        # Run the feedback learning process
        await scheduler.run_feedback_learning()
        
        # Verify that the rule update was called with expected parameters
        assert mock_update_rules.call_count > 0
        # Large Amount rule should have its threshold increased due to high false positive rate
        mock_update_rules.assert_any_call("Large Amount", 0.2, test_tenant_id, mock_db_session)
        
        # Verify that the ML model update was called
        assert mock_update_ml.call_count > 0 