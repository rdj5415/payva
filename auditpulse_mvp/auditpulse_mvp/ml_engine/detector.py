"""ML-based anomaly detector.

This module provides integration between the ML Engine and the rules engine
for comprehensive anomaly detection combining both approaches.
"""
import logging
import uuid
from typing import Dict, List, Optional, Tuple, Union

from fastapi import Depends, HTTPException, status
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.database.models import Anomaly, AnomalyType, DataSource, Tenant, Transaction
from auditpulse_mvp.ml_engine.ml_engine import MLEngine
from auditpulse_mvp.rules_engine.rule_engine import RuleEngine
from auditpulse_mvp.utils.settings import settings

# Configure logging
logger = logging.getLogger(__name__)

# ML Score threshold for flagging anomalies
DEFAULT_ML_SCORE_THRESHOLD = 0.7  # 0-1 scale, higher is more anomalous


class MLAnomalyDetector:
    """ML-based anomaly detector.
    
    This class integrates ML model predictions with the rules engine to
    provide comprehensive anomaly detection for transactions.
    """

    def __init__(
        self,
        db_session: AsyncSession = Depends(get_db_session),
        ml_score_threshold: float = DEFAULT_ML_SCORE_THRESHOLD,
    ):
        """Initialize the ML anomaly detector.
        
        Args:
            db_session: Database session for data access.
            ml_score_threshold: Threshold for ML scores to flag as anomalous.
        """
        self.db_session = db_session
        self.ml_engine = MLEngine(db_session=db_session)
        self.rule_engine = RuleEngine(db_session=db_session)
        self.ml_score_threshold = ml_score_threshold
    
    async def process_transaction(
        self, transaction: Transaction, check_rules: bool = True
    ) -> Tuple[float, Optional[Anomaly]]:
        """Process a transaction for anomaly detection using ML and rules.
        
        Args:
            transaction: The transaction to process.
            check_rules: Whether to also check rules engine for anomalies.
            
        Returns:
            Tuple[float, Optional[Anomaly]]: ML score and anomaly if detected.
        """
        tenant_id = transaction.tenant_id
        
        # First, try to score with the ML model
        ml_score = 0.0
        ml_anomaly_detected = False
        
        try:
            ml_score = await self.ml_engine.score_transaction(tenant_id, transaction)
            ml_anomaly_detected = ml_score >= self.ml_score_threshold
            
            logger.debug(
                f"ML score for transaction {transaction.id}: {ml_score} "
                f"(threshold: {self.ml_score_threshold}, anomaly: {ml_anomaly_detected})"
            )
        except HTTPException as e:
            if e.status_code == status.HTTP_404_NOT_FOUND:
                # No model found, this is OK, we'll rely on rules only
                logger.info(f"No ML model found for tenant {tenant_id}, using rules only")
            else:
                # Other error, log it but continue with rules
                logger.error(f"Error scoring transaction with ML: {e.detail}")
        except Exception as e:
            # Log error but continue with rules
            logger.exception(f"Unexpected error scoring transaction with ML: {e}")
        
        # If ML detected an anomaly, create it immediately
        anomaly = None
        if ml_anomaly_detected:
            # Create anomaly from ML detection
            anomaly = await self._create_anomaly(
                transaction=transaction,
                ml_score=ml_score,
                anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                description=f"ML model detected statistical outlier (score: {ml_score:.2f})",
                confidence=ml_score,
            )
            logger.info(
                f"ML model detected anomaly for transaction {transaction.id} "
                f"with score {ml_score:.2f}"
            )
        
        # If requested, also check rules-based detection
        if check_rules and not anomaly:
            try:
                # Run rules engine on the transaction
                rule_result = await self.rule_engine.evaluate_transaction(transaction)
                
                if rule_result.anomaly_detected:
                    # Rules detected an anomaly, create it
                    anomaly = await self._create_anomaly(
                        transaction=transaction,
                        ml_score=ml_score,  # Still include ML score for reference
                        anomaly_type=rule_result.anomaly_type,
                        description=rule_result.description,
                        confidence=rule_result.confidence,
                        rule_name=rule_result.rule_name,
                    )
                    logger.info(
                        f"Rules engine detected anomaly for transaction {transaction.id}: "
                        f"{rule_result.rule_name} ({rule_result.anomaly_type})"
                    )
            except Exception as e:
                logger.exception(f"Error evaluating transaction with rules engine: {e}")
        
        return ml_score, anomaly
    
    async def process_transactions(
        self, transactions: List[Transaction], check_rules: bool = True
    ) -> Dict[uuid.UUID, Tuple[float, Optional[Anomaly]]]:
        """Process multiple transactions for anomaly detection.
        
        Args:
            transactions: The transactions to process.
            check_rules: Whether to also check rules engine for anomalies.
            
        Returns:
            Dict[uuid.UUID, Tuple[float, Optional[Anomaly]]]: Mapping of transaction IDs
                to (ML score, detected anomaly) tuples.
        """
        if not transactions:
            return {}
        
        # Group transactions by tenant for efficient batch scoring
        transactions_by_tenant = {}
        for txn in transactions:
            tenant_id = txn.tenant_id
            if tenant_id not in transactions_by_tenant:
                transactions_by_tenant[tenant_id] = []
            transactions_by_tenant[tenant_id].append(txn)
        
        # Process each tenant's transactions
        results = {}
        for tenant_id, tenant_txns in transactions_by_tenant.items():
            # Score transactions with ML model
            ml_scores = {}
            try:
                ml_scores = await self.ml_engine.batch_score_transactions(tenant_id, tenant_txns)
            except HTTPException as e:
                if e.status_code == status.HTTP_404_NOT_FOUND:
                    # No model found, this is OK, we'll use default scores
                    logger.info(f"No ML model found for tenant {tenant_id}, using rules only")
                    ml_scores = {txn.id: 0.5 for txn in tenant_txns}  # Neutral scores
                else:
                    # Other error, log it but continue with neutral scores
                    logger.error(f"Error batch scoring transactions with ML: {e.detail}")
                    ml_scores = {txn.id: 0.5 for txn in tenant_txns}
            except Exception as e:
                # Log error but continue with neutral scores
                logger.exception(f"Unexpected error batch scoring transactions with ML: {e}")
                ml_scores = {txn.id: 0.5 for txn in tenant_txns}
            
            # Process each transaction
            for txn in tenant_txns:
                ml_score = ml_scores.get(txn.id, 0.5)  # Default to neutral if not in results
                ml_anomaly_detected = ml_score >= self.ml_score_threshold
                
                anomaly = None
                if ml_anomaly_detected:
                    # Create anomaly from ML detection
                    anomaly = await self._create_anomaly(
                        transaction=txn,
                        ml_score=ml_score,
                        anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                        description=f"ML model detected statistical outlier (score: {ml_score:.2f})",
                        confidence=ml_score,
                    )
                
                # If requested and no ML anomaly, check rules-based detection
                if check_rules and not anomaly:
                    try:
                        # Run rules engine on the transaction
                        rule_result = await self.rule_engine.evaluate_transaction(txn)
                        
                        if rule_result.anomaly_detected:
                            # Rules detected an anomaly, create it
                            anomaly = await self._create_anomaly(
                                transaction=txn,
                                ml_score=ml_score,  # Still include ML score for reference
                                anomaly_type=rule_result.anomaly_type,
                                description=rule_result.description,
                                confidence=rule_result.confidence,
                                rule_name=rule_result.rule_name,
                            )
                    except Exception as e:
                        logger.exception(f"Error evaluating transaction with rules engine: {e}")
                
                # Store the result
                results[txn.id] = (ml_score, anomaly)
        
        return results
    
    async def _create_anomaly(
        self,
        transaction: Transaction,
        ml_score: float,
        anomaly_type: AnomalyType,
        description: str,
        confidence: float,
        rule_name: Optional[str] = None,
    ) -> Anomaly:
        """Create an anomaly record in the database.
        
        Args:
            transaction: The transaction with the anomaly.
            ml_score: ML model score for the transaction.
            anomaly_type: Type of anomaly detected.
            description: Description of the anomaly.
            confidence: Confidence score for the anomaly (0-1).
            rule_name: Name of the rule that detected the anomaly (if any).
            
        Returns:
            Anomaly: The created anomaly record.
        """
        # Check if there's already an anomaly for this transaction
        stmt = select(Anomaly).where(
            and_(
                Anomaly.transaction_id == transaction.id,
                Anomaly.is_resolved == False,
            )
        )
        result = await self.db_session.execute(stmt)
        existing_anomaly = result.scalars().first()
        
        if existing_anomaly:
            # Update the existing anomaly if new one has higher confidence
            if confidence > existing_anomaly.confidence:
                existing_anomaly.anomaly_type = anomaly_type
                existing_anomaly.description = description
                existing_anomaly.confidence = confidence
                existing_anomaly.ml_score = ml_score
                
                if rule_name:
                    existing_anomaly.detection_metadata = {
                        **(existing_anomaly.detection_metadata or {}),
                        "rule_name": rule_name,
                        "ml_score": ml_score,
                    }
                else:
                    existing_anomaly.detection_metadata = {
                        **(existing_anomaly.detection_metadata or {}),
                        "ml_score": ml_score,
                    }
                
                await self.db_session.commit()
            
            return existing_anomaly
        
        # Create a new anomaly
        detection_metadata = {"ml_score": ml_score}
        if rule_name:
            detection_metadata["rule_name"] = rule_name
        
        anomaly = Anomaly(
            transaction_id=transaction.id,
            tenant_id=transaction.tenant_id,
            anomaly_type=anomaly_type,
            description=description,
            confidence=confidence,
            is_resolved=False,
            ml_score=ml_score,
            detection_metadata=detection_metadata,
        )
        
        self.db_session.add(anomaly)
        await self.db_session.commit()
        await self.db_session.refresh(anomaly)
        
        return anomaly


async def get_ml_anomaly_detector(
    db_session: AsyncSession = Depends(get_db_session),
) -> MLAnomalyDetector:
    """Get an ML anomaly detector instance.
    
    Args:
        db_session: Database session for data access.
        
    Returns:
        MLAnomalyDetector: The ML anomaly detector instance.
    """
    return MLAnomalyDetector(db_session=db_session) 