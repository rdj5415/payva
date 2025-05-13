"""Feedback service for AuditPulse MVP.

This module provides functionality for collecting and processing user feedback
on anomalies, which is used to improve the ML models and risk scoring.
"""

import datetime
import logging
from typing import Dict, List, Optional, Set, Union
from uuid import UUID

from fastapi import HTTPException
from pydantic import BaseModel, UUID4
from sqlalchemy.orm import Session

from auditpulse_mvp.database.models import Anomaly, FeedbackType, Transaction
from auditpulse_mvp.ml_engine.ml_engine import get_ml_engine
from auditpulse_mvp.risk_engine.risk_engine import get_risk_engine

# Configure logging
logger = logging.getLogger(__name__)


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""

    anomaly_id: UUID4
    feedback_type: FeedbackType
    resolution_notes: str
    additional_context: Optional[Dict] = None


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""

    success: bool
    message: str
    anomaly_id: UUID4
    feedback_type: FeedbackType
    timestamp: datetime.datetime


class FeedbackService:
    """Service for handling user feedback on anomalies."""

    def __init__(self, db_session: Session):
        """Initialize the feedback service.

        Args:
            db_session: Database session.
        """
        self.db_session = db_session
        self.ml_engine = get_ml_engine()
        self.risk_engine = get_risk_engine()

    async def submit_feedback(
        self,
        tenant_id: UUID4,
        feedback: FeedbackRequest,
    ) -> FeedbackResponse:
        """Submit feedback for an anomaly.

        Args:
            tenant_id: Tenant ID.
            feedback: Feedback request.

        Returns:
            Feedback response.

        Raises:
            HTTPException: If anomaly not found or feedback submission fails.
        """
        # Get anomaly
        anomaly = (
            self.db_session.query(Anomaly)
            .filter(
                Anomaly.id == feedback.anomaly_id,
                Anomaly.tenant_id == tenant_id,
            )
            .first()
        )

        if not anomaly:
            raise HTTPException(status_code=404, detail="Anomaly not found")

        try:
            # Update anomaly
            anomaly.is_resolved = True
            anomaly.feedback_type = feedback.feedback_type
            anomaly.resolution_notes = feedback.resolution_notes
            anomaly.resolved_at = datetime.datetime.now()

            # Add feedback to ML training data
            await self._add_to_training_data(anomaly, feedback)

            # Update risk scoring if needed
            if feedback.feedback_type in [
                FeedbackType.FALSE_POSITIVE,
                FeedbackType.FALSE_NEGATIVE,
            ]:
                await self._update_risk_scoring(anomaly, feedback)

            # Commit changes
            self.db_session.commit()

            return FeedbackResponse(
                success=True,
                message="Feedback submitted successfully",
                anomaly_id=anomaly.id,
                feedback_type=feedback.feedback_type,
                timestamp=anomaly.resolved_at,
            )

        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Failed to submit feedback: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to submit feedback",
            )

    async def get_feedback_stats(
        self,
        tenant_id: UUID4,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
    ) -> Dict:
        """Get feedback statistics for a tenant.

        Args:
            tenant_id: Tenant ID.
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            Dictionary containing feedback statistics.
        """
        # Build query
        query = self.db_session.query(Anomaly).filter(
            Anomaly.tenant_id == tenant_id,
            Anomaly.is_resolved == True,
        )

        if start_date:
            query = query.filter(Anomaly.resolved_at >= start_date)

        if end_date:
            query = query.filter(Anomaly.resolved_at <= end_date)

        # Get all resolved anomalies
        anomalies = query.all()

        # Calculate statistics
        total = len(anomalies)
        feedback_types = {
            feedback_type: len(
                [a for a in anomalies if a.feedback_type == feedback_type]
            )
            for feedback_type in FeedbackType
        }

        # Calculate accuracy metrics
        true_positives = feedback_types[FeedbackType.TRUE_POSITIVE]
        false_positives = feedback_types[FeedbackType.FALSE_POSITIVE]
        true_negatives = feedback_types[FeedbackType.TRUE_NEGATIVE]
        false_negatives = feedback_types[FeedbackType.FALSE_NEGATIVE]

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "total_feedback": total,
            "feedback_types": feedback_types,
            "accuracy_metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
            },
        }

    async def _add_to_training_data(
        self,
        anomaly: Anomaly,
        feedback: FeedbackRequest,
    ) -> None:
        """Add anomaly and feedback to training data.

        Args:
            anomaly: Anomaly to add to training data.
            feedback: Feedback for the anomaly.
        """
        # Skip if transaction is None
        if anomaly.transaction is None:
            logger.warning(
                f"Cannot add anomaly {anomaly.id} to training data: transaction is None"
            )
            return

        # Prepare training example
        example = {
            "transaction": {
                "amount": anomaly.transaction.amount,
                "description": anomaly.transaction.description,
                "merchant_name": anomaly.transaction.merchant_name,
                "category": anomaly.transaction.category,
                "transaction_date": anomaly.transaction.transaction_date,
            },
            "anomaly": {
                "type": str(anomaly.anomaly_type),
                "score": anomaly.score,
                "risk_level": str(anomaly.risk_level),
            },
            "feedback": {
                "type": str(feedback.feedback_type),
                "notes": feedback.resolution_notes,
                "additional_context": feedback.additional_context,
            },
        }

        # Currently, MLEngine doesn't have a direct add_training_example method
        # We would need to either extend MLEngine or store this data for later use
        # For now, just log that we received feedback
        logger.info(
            f"Received feedback for anomaly {anomaly.id}: {feedback.feedback_type}"
        )

        # TODO: Implement proper feedback collection for ML training
        # This could involve:
        # 1. Storing feedback in a dedicated table
        # 2. Creating a training dataset file
        # 3. Extending the MLEngine to include an add_training_example method

    async def _update_risk_scoring(
        self,
        anomaly: Anomaly,
        feedback: FeedbackRequest,
    ) -> None:
        """Update risk scoring based on feedback.

        Args:
            anomaly: Anomaly to update scoring for.
            feedback: Feedback for the anomaly.
        """
        # Get transaction
        transaction = anomaly.transaction

        # Skip if transaction is None
        if transaction is None:
            logger.warning(
                f"Cannot update risk scoring for anomaly {anomaly.id}: transaction is None"
            )
            return

        # Update risk scoring
        if feedback.feedback_type == FeedbackType.FALSE_POSITIVE:
            # Decrease risk score for similar transactions
            await self.risk_engine.adjust_risk_scoring(
                transaction=transaction,
                adjustment_factor=0.5,  # Reduce risk score by 50%
            )

        elif feedback.feedback_type == FeedbackType.FALSE_NEGATIVE:
            # Increase risk score for similar transactions
            await self.risk_engine.adjust_risk_scoring(
                transaction=transaction,
                adjustment_factor=2.0,  # Double risk score
            )
