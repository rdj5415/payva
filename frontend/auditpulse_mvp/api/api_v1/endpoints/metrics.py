"""Metrics API endpoints for AuditPulse MVP.

This module provides API endpoints for retrieving metrics and statistics
for the dashboard, including anomaly counts, risk levels, and trends.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from auditpulse_mvp.api.api_v1.auth import get_current_user
from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.database.models import Anomaly, AnomalyStatus, AnomalyRiskLevel

# Create router
router = APIRouter()


@router.get("/metrics")
async def get_metrics(
    tenant_id: str,
    db: Session = Depends(get_db_session),
    current_user = Depends(get_current_user),
) -> Dict:
    """Get key metrics for the dashboard.
    
    Args:
        tenant_id: Tenant ID
        db: Database session
        current_user: Current user
        
    Returns:
        Dictionary containing key metrics
    """
    try:
        # Get current date range
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        
        # Get total anomalies
        total_anomalies = db.query(func.count(Anomaly.id)).filter(
            Anomaly.tenant_id == tenant_id,
        ).scalar()
        
        # Get anomaly change
        previous_total = db.query(func.count(Anomaly.id)).filter(
            and_(
                Anomaly.tenant_id == tenant_id,
                Anomaly.created_at < week_ago,
            ),
        ).scalar()
        anomaly_change = total_anomalies - previous_total
        
        # Get high risk count
        high_risk_count = db.query(func.count(Anomaly.id)).filter(
            and_(
                Anomaly.tenant_id == tenant_id,
                Anomaly.risk_level == AnomalyRiskLevel.HIGH,
            ),
        ).scalar()
        
        # Get high risk change
        previous_high_risk = db.query(func.count(Anomaly.id)).filter(
            and_(
                Anomaly.tenant_id == tenant_id,
                Anomaly.risk_level == AnomalyRiskLevel.HIGH,
                Anomaly.created_at < week_ago,
            ),
        ).scalar()
        high_risk_change = high_risk_count - previous_high_risk
        
        # Calculate accuracy
        total_feedback = db.query(func.count(Anomaly.id)).filter(
            and_(
                Anomaly.tenant_id == tenant_id,
                Anomaly.feedback_type.isnot(None),
            ),
        ).scalar()
        
        true_positives = db.query(func.count(Anomaly.id)).filter(
            and_(
                Anomaly.tenant_id == tenant_id,
                Anomaly.feedback_type == "true_positive",
            ),
        ).scalar()
        
        accuracy = (true_positives / total_feedback * 100) if total_feedback > 0 else 0
        
        # Calculate accuracy change
        previous_true_positives = db.query(func.count(Anomaly.id)).filter(
            and_(
                Anomaly.tenant_id == tenant_id,
                Anomaly.feedback_type == "true_positive",
                Anomaly.created_at < week_ago,
            ),
        ).scalar()
        
        previous_total_feedback = db.query(func.count(Anomaly.id)).filter(
            and_(
                Anomaly.tenant_id == tenant_id,
                Anomaly.feedback_type.isnot(None),
                Anomaly.created_at < week_ago,
            ),
        ).scalar()
        
        previous_accuracy = (
            previous_true_positives / previous_total_feedback * 100
        ) if previous_total_feedback > 0 else 0
        
        accuracy_change = accuracy - previous_accuracy
        
        # Calculate average response time
        resolved_anomalies = db.query(Anomaly).filter(
            and_(
                Anomaly.tenant_id == tenant_id,
                Anomaly.status == AnomalyStatus.RESOLVED,
            ),
        ).all()
        
        total_response_time = timedelta()
        for anomaly in resolved_anomalies:
            if anomaly.resolved_at and anomaly.created_at:
                total_response_time += anomaly.resolved_at - anomaly.created_at
        
        avg_response_time = (
            total_response_time.total_seconds() / len(resolved_anomalies) / 3600
        ) if resolved_anomalies else 0
        
        # Calculate response time change
        previous_resolved = db.query(Anomaly).filter(
            and_(
                Anomaly.tenant_id == tenant_id,
                Anomaly.status == AnomalyStatus.RESOLVED,
                Anomaly.created_at < week_ago,
            ),
        ).all()
        
        previous_total_time = timedelta()
        for anomaly in previous_resolved:
            if anomaly.resolved_at and anomaly.created_at:
                previous_total_time += anomaly.resolved_at - anomaly.created_at
        
        previous_avg_time = (
            previous_total_time.total_seconds() / len(previous_resolved) / 3600
        ) if previous_resolved else 0
        
        response_time_change = avg_response_time - previous_avg_time
        
        return {
            "total_anomalies": total_anomalies,
            "anomaly_change": anomaly_change,
            "high_risk_count": high_risk_count,
            "high_risk_change": high_risk_change,
            "accuracy": accuracy,
            "accuracy_change": accuracy_change,
            "avg_response_time": avg_response_time,
            "response_time_change": response_time_change,
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics: {str(e)}",
        )


@router.get("/metrics/risk")
async def get_risk_metrics(
    tenant_id: str,
    period: str = "7d",
    db: Session = Depends(get_db_session),
    current_user = Depends(get_current_user),
) -> Dict:
    """Get risk metrics and trends.
    
    Args:
        tenant_id: Tenant ID
        period: Time period (e.g., "7d", "30d")
        db: Database session
        current_user: Current user
        
    Returns:
        Dictionary containing risk metrics and trends
    """
    try:
        # Parse period
        days = int(period[:-1])
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get daily risk counts
        risk_trends = []
        current_date = start_date
        
        while current_date <= end_date:
            next_date = current_date + timedelta(days=1)
            
            # Get counts for each risk level
            high_risk = db.query(func.count(Anomaly.id)).filter(
                and_(
                    Anomaly.tenant_id == tenant_id,
                    Anomaly.risk_level == AnomalyRiskLevel.HIGH,
                    Anomaly.created_at >= current_date,
                    Anomaly.created_at < next_date,
                ),
            ).scalar()
            
            medium_risk = db.query(func.count(Anomaly.id)).filter(
                and_(
                    Anomaly.tenant_id == tenant_id,
                    Anomaly.risk_level == AnomalyRiskLevel.MEDIUM,
                    Anomaly.created_at >= current_date,
                    Anomaly.created_at < next_date,
                ),
            ).scalar()
            
            low_risk = db.query(func.count(Anomaly.id)).filter(
                and_(
                    Anomaly.tenant_id == tenant_id,
                    Anomaly.risk_level == AnomalyRiskLevel.LOW,
                    Anomaly.created_at >= current_date,
                    Anomaly.created_at < next_date,
                ),
            ).scalar()
            
            risk_trends.append({
                "timestamp": current_date.isoformat(),
                "high_risk": high_risk,
                "medium_risk": medium_risk,
                "low_risk": low_risk,
            })
            
            current_date = next_date
        
        return {
            "risk_trends": risk_trends,
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get risk metrics: {str(e)}",
        ) 