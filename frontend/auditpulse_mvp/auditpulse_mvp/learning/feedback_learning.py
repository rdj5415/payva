"""Feedback-based learning for anomaly detection.

This module implements continuous learning mechanisms that update machine learning
models and rule engine parameters based on user feedback on anomaly detections.
"""
import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

from fastapi import Depends, HTTPException, status
from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text

from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.database.models import (
    Anomaly,
    FeedbackType,
    Tenant,
    AnomalyType,
    TenantConfiguration,
)
from auditpulse_mvp.ml_engine.ml_engine import MLEngine
from auditpulse_mvp.rules_engine.rules_engine import RulesEngine
from auditpulse_mvp.api.api_v1.endpoints.config import (
    SensitivityConfiguration,
    get_tenant_sensitivity_config,
    get_preset_configuration,
    SensitivityLevel,
)
from auditpulse_mvp.ml.models import update_ml_model_parameters
from auditpulse_mvp.rules.rule_engine import update_rule_thresholds
from auditpulse_mvp.utils.settings import settings

# Configure logging
logger = logging.getLogger(__name__)


class FeedbackLearner:
    """Class for implementing feedback-based learning.
    
    This class processes anomaly feedback to adjust machine learning model
    sensitivity and rule engine parameters for continuous improvement.
    """
    
    def __init__(self, db_session: AsyncSession = Depends(get_db_session)):
        """Initialize the feedback learner.
        
        Args:
            db_session: Database session
        """
        self.db_session = db_session
        self.ml_engine = MLEngine(db_session=db_session)
        self.rules_engine = RulesEngine(db_session=db_session)
    
    async def process_recent_feedback(
        self, tenant_id: uuid.UUID, days: int = 30
    ) -> Dict[str, Any]:
        """Process recent feedback to adjust detection parameters.
        
        This function analyzes recent anomaly feedback and adjusts ML thresholds
        and rule parameters accordingly.
        
        Args:
            tenant_id: Tenant ID
            days: Number of days of feedback to consider
            
        Returns:
            Dict containing the results of the learning process
        """
        try:
            # Gather recent feedback data
            feedback_data = await self.gather_feedback_data(tenant_id, days)
            
            if not feedback_data["anomalies"]:
                logger.info(f"No recent feedback found for tenant {tenant_id}")
                return {
                    "status": "skipped",
                    "reason": "No recent feedback found",
                    "processed_count": 0,
                }
            
            # Analyze feedback to calculate false positive/negative rates
            feedback_analysis = self.analyze_feedback(feedback_data)
            
            # Update sensitivity configuration based on feedback
            updated_config = await self.update_sensitivity_config(tenant_id, feedback_analysis)
            
            # Retrain ML model for the tenant if enough data exists
            try:
                retrain_result = await self.ml_engine.train_model(tenant_id)
                model_retrained = True
                model_info = {
                    "path": retrain_result["model_path"],
                    "samples": retrain_result["training_samples"],
                    "anomaly_rate": retrain_result["anomaly_rate"],
                }
            except HTTPException as e:
                if e.status_code == status.HTTP_400_BAD_REQUEST:
                    # Not enough data for retraining
                    model_retrained = False
                    model_info = {"error": str(e.detail)}
                else:
                    raise
            
            # Return the results
            return {
                "status": "success",
                "processed_count": len(feedback_data["anomalies"]),
                "false_positive_rate": feedback_analysis["false_positive_rate"],
                "true_positive_rate": feedback_analysis["true_positive_rate"],
                "config_updated": True,
                "model_retrained": model_retrained,
                "model_info": model_info if model_retrained else None,
                "updated_config": updated_config.dict(),
            }
        except Exception as e:
            logger.exception(f"Error processing feedback for tenant {tenant_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
            }
    
    async def gather_feedback_data(
        self, tenant_id: uuid.UUID, days: int = 30
    ) -> Dict[str, Any]:
        """Gather recent feedback data for a tenant.
        
        Args:
            tenant_id: Tenant ID
            days: Number of days of feedback to consider
            
        Returns:
            Dict containing feedback data
        """
        # Calculate the date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Query for anomalies with feedback in the date range
        stmt = (
            select(Anomaly)
            .where(
                and_(
                    Anomaly.tenant_id == tenant_id,
                    Anomaly.feedback.isnot(None),
                    Anomaly.updated_at >= start_date,
                    Anomaly.updated_at <= end_date,
                )
            )
        )
        
        result = await self.db_session.execute(stmt)
        anomalies = result.scalars().all()
        
        # Count feedback by type
        feedback_counts = {
            FeedbackType.TRUE_POSITIVE.value: 0,
            FeedbackType.FALSE_POSITIVE.value: 0,
            FeedbackType.IGNORE.value: 0,
        }
        
        for anomaly in anomalies:
            if anomaly.feedback in feedback_counts:
                feedback_counts[anomaly.feedback] += 1
        
        # Group anomalies by type
        ml_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.ML_BASED]
        rule_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.RULES_BASED]
        
        # Group false positives by rule name (if available)
        rule_false_positives = {}
        for anomaly in anomalies:
            if anomaly.feedback == FeedbackType.FALSE_POSITIVE.value:
                if anomaly.detection_metadata and "rule_name" in anomaly.detection_metadata:
                    rule_name = anomaly.detection_metadata["rule_name"]
                    if rule_name not in rule_false_positives:
                        rule_false_positives[rule_name] = 0
                    rule_false_positives[rule_name] += 1
        
        return {
            "anomalies": anomalies,
            "feedback_counts": feedback_counts,
            "ml_anomalies": ml_anomalies,
            "rule_anomalies": rule_anomalies,
            "rule_false_positives": rule_false_positives,
            "start_date": start_date,
            "end_date": end_date,
        }
    
    def analyze_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feedback data to determine false positive/negative rates.
        
        Args:
            feedback_data: Feedback data from gather_feedback_data()
            
        Returns:
            Dict containing analysis results
        """
        feedback_counts = feedback_data["feedback_counts"]
        ml_anomalies = feedback_data["ml_anomalies"]
        rule_anomalies = feedback_data["rule_anomalies"]
        
        # Calculate overall rates
        total_feedback = sum(feedback_counts.values())
        false_positive_count = feedback_counts[FeedbackType.FALSE_POSITIVE.value]
        true_positive_count = feedback_counts[FeedbackType.TRUE_POSITIVE.value]
        ignore_count = feedback_counts[FeedbackType.IGNORE.value]
        
        false_positive_rate = false_positive_count / total_feedback if total_feedback > 0 else 0
        true_positive_rate = true_positive_count / total_feedback if total_feedback > 0 else 0
        
        # Calculate ML-specific rates
        ml_total = len(ml_anomalies)
        ml_false_positives = sum(1 for a in ml_anomalies if a.feedback == FeedbackType.FALSE_POSITIVE.value)
        ml_true_positives = sum(1 for a in ml_anomalies if a.feedback == FeedbackType.TRUE_POSITIVE.value)
        
        ml_false_positive_rate = ml_false_positives / ml_total if ml_total > 0 else 0
        ml_true_positive_rate = ml_true_positives / ml_total if ml_total > 0 else 0
        
        # Calculate rule-specific rates
        rule_total = len(rule_anomalies)
        rule_false_positives = sum(1 for a in rule_anomalies if a.feedback == FeedbackType.FALSE_POSITIVE.value)
        rule_true_positives = sum(1 for a in rule_anomalies if a.feedback == FeedbackType.TRUE_POSITIVE.value)
        
        rule_false_positive_rate = rule_false_positives / rule_total if rule_total > 0 else 0
        rule_true_positive_rate = rule_true_positives / rule_total if rule_total > 0 else 0
        
        # Analyze rule-specific false positives
        rule_fp_analysis = {}
        for rule_name, count in feedback_data["rule_false_positives"].items():
            # Calculate the false positive rate for this rule
            rule_anomalies_count = sum(
                1 for a in rule_anomalies 
                if a.detection_metadata and a.detection_metadata.get("rule_name") == rule_name
            )
            
            rule_fp_analysis[rule_name] = {
                "false_positive_count": count,
                "total_count": rule_anomalies_count,
                "false_positive_rate": count / rule_anomalies_count if rule_anomalies_count > 0 else 0,
            }
        
        return {
            "total_feedback": total_feedback,
            "false_positive_count": false_positive_count,
            "true_positive_count": true_positive_count,
            "ignore_count": ignore_count,
            "false_positive_rate": false_positive_rate,
            "true_positive_rate": true_positive_rate,
            "ml_false_positive_rate": ml_false_positive_rate,
            "ml_true_positive_rate": ml_true_positive_rate,
            "rule_false_positive_rate": rule_false_positive_rate,
            "rule_true_positive_rate": rule_true_positive_rate,
            "rule_analysis": rule_fp_analysis,
        }
    
    async def update_sensitivity_config(
        self, tenant_id: uuid.UUID, feedback_analysis: Dict[str, Any]
    ) -> SensitivityConfiguration:
        """Update sensitivity configuration based on feedback analysis.
        
        Args:
            tenant_id: Tenant ID
            feedback_analysis: Results from analyze_feedback()
            
        Returns:
            Updated SensitivityConfiguration
        """
        # Get current configuration
        current_config = await get_tenant_sensitivity_config(self.db_session, tenant_id)
        
        # Make a copy of the current configuration
        updated_config = SensitivityConfiguration.parse_obj(current_config.dict())
        
        # Determine if we need to adjust sensitivity based on false positive rate
        fp_rate = feedback_analysis["false_positive_rate"]
        
        # High false positive rate suggests we need to decrease sensitivity
        if fp_rate > 0.5:  # More than 50% false positives
            if current_config.sensitivity_level == SensitivityLevel.HIGH:
                updated_config = get_preset_configuration(SensitivityLevel.MEDIUM)
            elif current_config.sensitivity_level == SensitivityLevel.MEDIUM:
                updated_config = get_preset_configuration(SensitivityLevel.LOW)
            else:
                # Already at low sensitivity, make custom adjustments
                updated_config.sensitivity_level = SensitivityLevel.CUSTOM
                
                # Increase ML threshold (make it harder to flag anomalies)
                updated_config.risk_engine.ml_threshold = min(
                    0.9, current_config.risk_engine.ml_threshold + 0.05
                )
                
                # Update rule thresholds based on specific rule false positive rates
                self._adjust_rule_thresholds(updated_config, feedback_analysis)
        
        # Low false positive rate with sufficient volume suggests we can increase sensitivity
        elif fp_rate < 0.2 and feedback_analysis["total_feedback"] >= 20:
            if current_config.sensitivity_level == SensitivityLevel.LOW:
                updated_config = get_preset_configuration(SensitivityLevel.MEDIUM)
            elif current_config.sensitivity_level == SensitivityLevel.MEDIUM:
                updated_config = get_preset_configuration(SensitivityLevel.HIGH)
            # If already at high sensitivity, no change needed
        
        # If custom configuration is needed, adjust specific parameters
        if updated_config.sensitivity_level == SensitivityLevel.CUSTOM:
            # Adjust ML-specific parameters
            ml_fp_rate = feedback_analysis["ml_false_positive_rate"]
            if ml_fp_rate > 0.5:
                # Decrease ML weight in scoring
                updated_config.risk_engine.ml_score_weight = max(
                    0.2, current_config.risk_engine.ml_score_weight - 0.05
                )
                updated_config.risk_engine.rules_score_weight = 1.0 - updated_config.risk_engine.ml_score_weight
            
            # Further tune rule-specific thresholds
            self._adjust_rule_thresholds(updated_config, feedback_analysis)
        
        # Save the updated configuration
        config_dict = updated_config.dict()
        
        # Check if configuration exists
        query = (
            select(TenantConfiguration)
            .where(
                and_(
                    TenantConfiguration.tenant_id == tenant_id,
                    TenantConfiguration.key == "sensitivity_config"
                )
            )
        )
        
        result = await self.db_session.execute(query)
        existing_config = result.scalar_one_or_none()
        
        if existing_config:
            stmt = (
                update(TenantConfiguration)
                .where(
                    and_(
                        TenantConfiguration.tenant_id == tenant_id,
                        TenantConfiguration.key == "sensitivity_config"
                    )
                )
                .values(value=config_dict)
            )
            await self.db_session.execute(stmt)
        else:
            # Create new configuration
            new_config = TenantConfiguration(
                tenant_id=tenant_id,
                key="sensitivity_config",
                value=config_dict,
            )
            self.db_session.add(new_config)
        
        # Commit changes
        await self.db_session.commit()
        
        return updated_config
    
    def _adjust_rule_thresholds(
        self, config: SensitivityConfiguration, feedback_analysis: Dict[str, Any]
    ) -> None:
        """Adjust rule thresholds based on feedback analysis.
        
        Args:
            config: Configuration to update
            feedback_analysis: Feedback analysis results
            
        Returns:
            None - updates config in place
        """
        rule_analysis = feedback_analysis.get("rule_analysis", {})
        
        # Adjust Large Transaction rule
        if "Large Amount" in rule_analysis and rule_analysis["Large Amount"]["false_positive_rate"] > 0.5:
            # Increase threshold if too many false positives
            if hasattr(config.rules, "large_transaction"):
                if config.rules.large_transaction.threshold:
                    config.rules.large_transaction.threshold *= 1.2  # Increase by 20%
        
        # Adjust Statistical Outlier rule
        if "Statistical Outlier" in rule_analysis and rule_analysis["Statistical Outlier"]["false_positive_rate"] > 0.5:
            # Increase standard deviation threshold
            if hasattr(config.rules, "irregular_amount"):
                if config.rules.irregular_amount.parameters and "std_dev_threshold" in config.rules.irregular_amount.parameters:
                    config.rules.irregular_amount.parameters["std_dev_threshold"] *= 1.2  # Increase by 20%
        
        # Adjust Weekend Transaction rule
        if "Weekend Transaction" in rule_analysis and rule_analysis["Weekend Transaction"]["false_positive_rate"] > 0.7:
            # If very high false positive rate, disable the rule
            if hasattr(config.rules, "weekend_transaction"):
                config.rules.weekend_transaction.enabled = False
        
        # Adjust Round Number Transaction rule
        if "Round Number Transaction" in rule_analysis and rule_analysis["Round Number Transaction"]["false_positive_rate"] > 0.7:
            # If very high false positive rate, disable the rule
            if hasattr(config.rules, "round_number_transaction"):
                config.rules.round_number_transaction.enabled = False
        
        # Ensure each rule has the required parameters
        for rule_attr in ["large_transaction", "irregular_amount", "weekend_transaction", "round_number_transaction"]:
            if hasattr(config.rules, rule_attr):
                rule = getattr(config.rules, rule_attr)
                if rule.parameters is None:
                    rule.parameters = {}


async def update_thresholds_from_feedback(tenant_id: uuid.UUID, db_session: AsyncSession) -> Dict[str, Any]:
    """Update thresholds and ML models based on feedback for a tenant.
    
    This function is called to process feedback and update detection parameters.
    It can be called on-demand or scheduled to run nightly.
    
    Args:
        tenant_id: Tenant ID
        db_session: Database session
        
    Returns:
        Dict with results of the update process
    """
    learner = FeedbackLearner(db_session=db_session)
    return await learner.process_recent_feedback(tenant_id)


# Singleton instance
_feedback_learner = None


async def get_feedback_learner(db_session: AsyncSession = Depends(get_db_session)) -> FeedbackLearner:
    """Get the global feedback learner instance.
    
    Args:
        db_session: Database session
        
    Returns:
        FeedbackLearner: The feedback learner instance
    """
    global _feedback_learner
    if _feedback_learner is None:
        _feedback_learner = FeedbackLearner(db_session=db_session)
    return _feedback_learner 


async def update_thresholds_from_feedback(
    tenant_id: uuid.UUID,
    db: AsyncSession,
    days_lookback: int = 30,
    min_feedback_count: int = 5,
) -> Dict[str, Any]:
    """Update rule thresholds and ML model parameters based on user feedback.
    
    This function analyzes recent feedback for false positives and true positives,
    then adjusts rule thresholds and ML model parameters accordingly.
    
    Args:
        tenant_id: The tenant ID to analyze feedback for
        db: Database session
        days_lookback: Number of days of feedback to consider
        min_feedback_count: Minimum number of feedback entries needed to make adjustments
        
    Returns:
        Dict containing information about the learning process and any adjustments made
    """
    logger.info(f"Starting feedback learning process for tenant {tenant_id}")
    
    # Time threshold for recent feedback
    cutoff_date = datetime.utcnow() - timedelta(days=days_lookback)
    
    # Results container
    results = {
        "rules_updated": 0,
        "ml_models_updated": 0,
        "rule_adjustments": {},
        "ml_adjustments": {},
        "no_action_reason": None,
    }
    
    # Get counts of feedback by type for each rule/ML model
    # Start with rules-based anomalies
    rule_query = text("""
        SELECT 
            detection_metadata->>'rule_name' as rule_name,
            feedback,
            COUNT(*) as count
        FROM 
            anomalies
        WHERE 
            tenant_id = :tenant_id
            AND feedback IS NOT NULL
            AND updated_at >= :cutoff_date
            AND anomaly_type = 'rules_based'
            AND detection_metadata ? 'rule_name'
        GROUP BY 
            detection_metadata->>'rule_name', feedback
    """)
    
    # Execute query
    rule_result = await db.execute(
        rule_query,
        {
            "tenant_id": str(tenant_id),
            "cutoff_date": cutoff_date,
        }
    )
    
    # Process rule feedback
    rule_feedback = {}
    for row in rule_result:
        rule_name = row.rule_name
        feedback_type = row.feedback
        count = row.count
        
        if rule_name not in rule_feedback:
            rule_feedback[rule_name] = {
                "false_positive": 0,
                "true_positive": 0,
                "total": 0,
            }
        
        if feedback_type == FeedbackType.FALSE_POSITIVE.value:
            rule_feedback[rule_name]["false_positive"] += count
        elif feedback_type == FeedbackType.TRUE_POSITIVE.value:
            rule_feedback[rule_name]["true_positive"] += count
            
        rule_feedback[rule_name]["total"] += count
    
    # Update rule thresholds based on feedback
    for rule_name, counts in rule_feedback.items():
        if counts["total"] < min_feedback_count:
            continue
        
        # Calculate false positive rate
        fp_rate = counts["false_positive"] / counts["total"]
        
        # Determine threshold adjustment
        if fp_rate > 0.7:  # High false positive rate
            # Increase threshold significantly
            adjustment = 0.2
            logger.info(f"High false positive rate ({fp_rate:.2f}) for rule {rule_name}")
        elif fp_rate > 0.5:  # Moderate false positive rate
            # Increase threshold moderately
            adjustment = 0.1
            logger.info(f"Moderate false positive rate ({fp_rate:.2f}) for rule {rule_name}")
        elif fp_rate > 0.3:  # Low false positive rate
            # Increase threshold slightly
            adjustment = 0.05
            logger.info(f"Low false positive rate ({fp_rate:.2f}) for rule {rule_name}")
        elif fp_rate < 0.1 and counts["true_positive"] > 0:
            # High true positive rate, decrease threshold slightly
            adjustment = -0.05
            logger.info(f"High true positive rate for rule {rule_name}")
        else:
            # No adjustment needed
            adjustment = 0
            logger.info(f"No adjustment needed for rule {rule_name}")
        
        if adjustment != 0:
            try:
                # Update rule threshold
                await update_rule_thresholds(rule_name, adjustment, tenant_id)
                results["rules_updated"] += 1
                results["rule_adjustments"][rule_name] = {
                    "adjustment": adjustment,
                    "false_positive_rate": fp_rate,
                    "feedback_count": counts["total"],
                }
                logger.info(f"Updated threshold for rule {rule_name} by {adjustment}")
            except Exception as e:
                logger.error(f"Failed to update threshold for rule {rule_name}: {e}")
    
    # Now handle ML-based anomalies
    ml_query = text("""
        SELECT 
            detection_metadata->>'model_name' as model_name,
            feedback,
            COUNT(*) as count
        FROM 
            anomalies
        WHERE 
            tenant_id = :tenant_id
            AND feedback IS NOT NULL
            AND updated_at >= :cutoff_date
            AND anomaly_type = 'ml_based'
            AND detection_metadata ? 'model_name'
        GROUP BY 
            detection_metadata->>'model_name', feedback
    """)
    
    # Execute query
    ml_result = await db.execute(
        ml_query,
        {
            "tenant_id": str(tenant_id),
            "cutoff_date": cutoff_date,
        }
    )
    
    # Process ML model feedback
    ml_feedback = {}
    for row in ml_result:
        model_name = row.model_name
        feedback_type = row.feedback
        count = row.count
        
        if model_name not in ml_feedback:
            ml_feedback[model_name] = {
                "false_positive": 0,
                "true_positive": 0,
                "total": 0,
            }
        
        if feedback_type == FeedbackType.FALSE_POSITIVE.value:
            ml_feedback[model_name]["false_positive"] += count
        elif feedback_type == FeedbackType.TRUE_POSITIVE.value:
            ml_feedback[model_name]["true_positive"] += count
            
        ml_feedback[model_name]["total"] += count
    
    # Update ML model parameters based on feedback
    for model_name, counts in ml_feedback.items():
        if counts["total"] < min_feedback_count:
            continue
        
        # Calculate false positive rate
        fp_rate = counts["false_positive"] / counts["total"]
        
        # Determine parameter adjustments
        if fp_rate > 0.7:  # High false positive rate
            # Significant adjustment to reduce false positives
            params = {
                "threshold": 0.2,  # Increase detection threshold
                "regularization": 0.1,  # Increase regularization
            }
            logger.info(f"High false positive rate ({fp_rate:.2f}) for model {model_name}")
        elif fp_rate > 0.5:  # Moderate false positive rate
            params = {
                "threshold": 0.1,  # Moderate threshold increase
                "regularization": 0.05,
            }
            logger.info(f"Moderate false positive rate ({fp_rate:.2f}) for model {model_name}")
        elif fp_rate > 0.3:  # Low false positive rate
            params = {
                "threshold": 0.05,  # Slight threshold increase
                "regularization": 0.02,
            }
            logger.info(f"Low false positive rate ({fp_rate:.2f}) for model {model_name}")
        elif fp_rate < 0.1 and counts["true_positive"] > 0:
            # High true positive rate, decrease threshold slightly
            params = {
                "threshold": -0.05,  # Decrease threshold slightly
                "regularization": 0,
            }
            logger.info(f"High true positive rate for model {model_name}")
        else:
            # No adjustment needed
            params = None
            logger.info(f"No adjustment needed for model {model_name}")
        
        if params:
            try:
                # Update ML model parameters
                await update_ml_model_parameters(model_name, params, tenant_id)
                results["ml_models_updated"] += 1
                results["ml_adjustments"][model_name] = {
                    "parameters": params,
                    "false_positive_rate": fp_rate,
                    "feedback_count": counts["total"],
                }
                logger.info(f"Updated parameters for model {model_name}: {params}")
            except Exception as e:
                logger.error(f"Failed to update parameters for model {model_name}: {e}")
    
    # If no updates were made, set reason
    if results["rules_updated"] == 0 and results["ml_models_updated"] == 0:
        results["no_action_reason"] = "Insufficient feedback or no significant patterns detected"
    
    logger.info(
        f"Completed feedback learning for tenant {tenant_id}: "
        f"{results['rules_updated']} rules and {results['ml_models_updated']} ML models updated"
    )
    
    return results


async def get_learning_statistics(
    tenant_id: uuid.UUID,
    db: AsyncSession,
    days_lookback: int = 90,
) -> Dict[str, Any]:
    """Get statistics about feedback and learning effectiveness.
    
    Args:
        tenant_id: The tenant ID to analyze
        db: Database session
        days_lookback: Number of days to look back
        
    Returns:
        Dict containing statistics about feedback and learning effectiveness
    """
    cutoff_date = datetime.utcnow() - timedelta(days=days_lookback)
    
    # Query for feedback statistics
    stats_query = text("""
        SELECT 
            anomaly_type,
            feedback,
            COUNT(*) as count
        FROM 
            anomalies
        WHERE 
            tenant_id = :tenant_id
            AND feedback IS NOT NULL
            AND created_at >= :cutoff_date
        GROUP BY 
            anomaly_type, feedback
    """)
    
    result = await db.execute(
        stats_query,
        {
            "tenant_id": str(tenant_id),
            "cutoff_date": cutoff_date,
        }
    )
    
    # Process statistics
    stats = {
        "rules_based": {
            "false_positive": 0,
            "true_positive": 0,
            "needs_investigation": 0,
            "other": 0,
            "total": 0,
        },
        "ml_based": {
            "false_positive": 0,
            "true_positive": 0, 
            "needs_investigation": 0,
            "other": 0,
            "total": 0,
        },
        "combined": {
            "false_positive": 0,
            "true_positive": 0,
            "needs_investigation": 0,
            "other": 0,
            "total": 0,
        },
        "total_feedback": 0,
    }
    
    for row in result:
        anomaly_type = row.anomaly_type
        feedback_type = row.feedback
        count = row.count
        
        if anomaly_type not in stats:
            continue
            
        if feedback_type == FeedbackType.FALSE_POSITIVE.value:
            stats[anomaly_type]["false_positive"] += count
        elif feedback_type == FeedbackType.TRUE_POSITIVE.value:
            stats[anomaly_type]["true_positive"] += count
        elif feedback_type == FeedbackType.NEEDS_INVESTIGATION.value:
            stats[anomaly_type]["needs_investigation"] += count
        else:
            stats[anomaly_type]["other"] += count
            
        stats[anomaly_type]["total"] += count
        stats["total_feedback"] += count
    
    # Calculate false positive rates
    for anomaly_type in ["rules_based", "ml_based", "combined"]:
        if stats[anomaly_type]["total"] > 0:
            stats[anomaly_type]["false_positive_rate"] = (
                stats[anomaly_type]["false_positive"] / stats[anomaly_type]["total"]
            )
            stats[anomaly_type]["true_positive_rate"] = (
                stats[anomaly_type]["true_positive"] / stats[anomaly_type]["total"]
            )
        else:
            stats[anomaly_type]["false_positive_rate"] = 0
            stats[anomaly_type]["true_positive_rate"] = 0
    
    # Add learning adjustments history
    # In a real implementation, this would query a table of model/rule adjustment history
    # For now, we'll just return a placeholder
    stats["learning_adjustments"] = {
        "rules_adjusted": 0,
        "ml_models_adjusted": 0,
        "last_adjustment_date": None,
    }
    
    return stats 