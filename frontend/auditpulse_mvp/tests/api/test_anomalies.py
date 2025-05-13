"""Tests for anomalies API endpoints.

This module contains tests for the anomalies API endpoints.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.models import Anomaly, Transaction, User, Tenant
from auditpulse_mvp.api.api_v1.endpoints.anomalies import AnomalyResponse


@pytest.fixture
async def mock_transactions(db: AsyncSession, test_tenant: Tenant) -> List[Transaction]:
    """Create sample transactions for testing.

    Args:
        db: Database session
        test_tenant: Test tenant

    Returns:
        List of created transactions
    """
    transactions = []

    # Create 5 sample transactions
    for i in range(5):
        txn = Transaction(
            id=uuid.uuid4(),
            tenant_id=test_tenant.id,
            transaction_id=f"TXN-{i}",
            source="quickbooks",
            source_account_id="ACCT-123",
            amount=1000 * (i + 1),
            currency="USD",
            description=f"Test transaction {i}",
            category="Expenses",
            merchant_name=f"Vendor {i}",
            transaction_date=datetime.utcnow() - timedelta(days=i),
            posting_date=datetime.utcnow() - timedelta(days=i),
            raw_data={"source_data": f"data-{i}"},
        )
        db.add(txn)
        transactions.append(txn)

    await db.commit()

    # Refresh transactions with IDs
    for txn in transactions:
        await db.refresh(txn)

    return transactions


@pytest.fixture
async def mock_anomalies(
    db: AsyncSession, test_tenant: Tenant, mock_transactions: List[Transaction]
) -> List[Anomaly]:
    """Create sample anomalies for testing.

    Args:
        db: Database session
        test_tenant: Test tenant
        mock_transactions: Sample transactions

    Returns:
        List of created anomalies
    """
    anomalies = []

    # Create anomalies for each transaction with different risk scores
    for i, txn in enumerate(mock_transactions):
        anomaly = Anomaly(
            id=uuid.uuid4(),
            tenant_id=test_tenant.id,
            transaction_id=txn.id,
            anomaly_type="large_amount" if i % 2 == 0 else "unusual_vendor",
            description=f"Anomaly for transaction {i}",
            confidence=0.8 + (i * 0.03),
            ml_score=0.7 + (i * 0.05),
            risk_score=20 * (i + 1),  # 20, 40, 60, 80, 100
            risk_level=["low", "medium", "medium", "high", "critical"][i],
            explanation=f"This transaction is {i+1}x larger than usual for this vendor",
            rule_score=25 * (i + 1),
            detection_metadata={
                "rule_name": (
                    "high_value_transaction" if i % 2 == 0 else "unusual_vendor"
                ),
                "threshold": 5000,
                "actual_value": 1000 * (i + 1),
            },
            is_resolved=i >= 3,  # Last two are resolved
            feedback=(
                "false_positive" if i == 3 else ("true_positive" if i == 4 else None)
            ),
            feedback_notes="This is expected" if i == 3 else (None),
        )
        db.add(anomaly)
        anomalies.append(anomaly)

    await db.commit()

    # Refresh anomalies with IDs
    for anomaly in anomalies:
        await db.refresh(anomaly)

    return anomalies


async def test_list_anomalies(
    async_client: AsyncClient,
    test_token: str,
    test_tenant: Tenant,
    mock_anomalies: List[Anomaly],
):
    """Test listing anomalies endpoint.

    Args:
        async_client: Async test client
        test_token: Authentication token
        test_tenant: Test tenant
        mock_anomalies: Sample anomalies
    """
    # Make request
    response = await async_client.get(
        "/api/v1/anomalies",
        headers={
            "Authorization": f"Bearer {test_token}",
            "X-Tenant-ID": str(test_tenant.id),
        },
    )

    # Check response
    assert response.status_code == 200
    data = response.json()

    # Validate response structure
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert "has_more" in data

    # Validate response data
    assert data["total"] == len(mock_anomalies)
    assert data["page"] == 1
    assert len(data["items"]) == len(mock_anomalies)

    # Validate first item fields
    first_item = data["items"][0]
    assert "id" in first_item
    assert "transaction_id" in first_item
    assert "risk_score" in first_item
    assert "risk_level" in first_item
    assert "explanation" in first_item


async def test_list_anomalies_filtering(
    async_client: AsyncClient,
    test_token: str,
    test_tenant: Tenant,
    mock_anomalies: List[Anomaly],
):
    """Test filtering anomalies.

    Args:
        async_client: Async test client
        test_token: Authentication token
        test_tenant: Test tenant
        mock_anomalies: Sample anomalies
    """
    # Test filtering by min risk score
    response = await async_client.get(
        "/api/v1/anomalies?min_risk_score=50",
        headers={
            "Authorization": f"Bearer {test_token}",
            "X-Tenant-ID": str(test_tenant.id),
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3  # Should only return anomalies with risk score >= 50

    # Test filtering by status (new)
    response = await async_client.get(
        "/api/v1/anomalies?statuses=new",
        headers={
            "Authorization": f"Bearer {test_token}",
            "X-Tenant-ID": str(test_tenant.id),
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert (
        data["total"] == 3
    )  # Should only return unresolved anomalies without feedback

    # Test filtering by anomaly type
    response = await async_client.get(
        "/api/v1/anomalies?anomaly_types=rules_based",
        headers={
            "Authorization": f"Bearer {test_token}",
            "X-Tenant-ID": str(test_tenant.id),
        },
    )

    # Will return 0 because our mock data uses string types not from the enum
    assert response.status_code == 200

    # Test pagination
    response = await async_client.get(
        "/api/v1/anomalies?page=1&page_size=2",
        headers={
            "Authorization": f"Bearer {test_token}",
            "X-Tenant-ID": str(test_tenant.id),
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 5  # Total count should still be 5
    assert len(data["items"]) == 2  # But only 2 items per page
    assert data["has_more"] == True  # More items available


async def test_get_anomaly(
    async_client: AsyncClient,
    test_token: str,
    test_tenant: Tenant,
    mock_anomalies: List[Anomaly],
):
    """Test getting a single anomaly.

    Args:
        async_client: Async test client
        test_token: Authentication token
        test_tenant: Test tenant
        mock_anomalies: Sample anomalies
    """
    # Get the first anomaly
    anomaly_id = mock_anomalies[0].id

    # Make request
    response = await async_client.get(
        f"/api/v1/anomalies/{anomaly_id}",
        headers={
            "Authorization": f"Bearer {test_token}",
            "X-Tenant-ID": str(test_tenant.id),
        },
    )

    # Check response
    assert response.status_code == 200
    data = response.json()

    # Validate response data
    assert data["id"] == str(anomaly_id)
    assert data["risk_score"] == mock_anomalies[0].risk_score
    assert data["risk_level"] == mock_anomalies[0].risk_level

    # Test with invalid ID
    response = await async_client.get(
        f"/api/v1/anomalies/{uuid.uuid4()}",
        headers={
            "Authorization": f"Bearer {test_token}",
            "X-Tenant-ID": str(test_tenant.id),
        },
    )

    assert response.status_code == 404


async def test_add_anomaly_feedback(
    async_client: AsyncClient,
    test_token: str,
    test_tenant: Tenant,
    mock_anomalies: List[Anomaly],
):
    """Test adding feedback to an anomaly.

    Args:
        async_client: Async test client
        test_token: Authentication token
        test_tenant: Test tenant
        mock_anomalies: Sample anomalies
    """
    # Get an unresolved anomaly
    anomaly_id = mock_anomalies[0].id

    # Make request to add feedback
    response = await async_client.post(
        f"/api/v1/anomalies/{anomaly_id}/feedback",
        json={
            "feedback_type": "false_positive",
            "notes": "This is an expected transaction",
            "should_notify": False,
        },
        headers={
            "Authorization": f"Bearer {test_token}",
            "X-Tenant-ID": str(test_tenant.id),
        },
    )

    # Check response
    assert response.status_code == 200
    data = response.json()

    # Validate response data
    assert data["id"] == str(anomaly_id)
    assert data["feedback"] == "false_positive"
    assert data["feedback_notes"] == "This is an expected transaction"
    assert (
        data["is_resolved"] == True
    )  # Should be automatically resolved for false positives

    # Test with invalid ID
    response = await async_client.post(
        f"/api/v1/anomalies/{uuid.uuid4()}/feedback",
        json={
            "feedback_type": "false_positive",
            "notes": "This is an expected transaction",
        },
        headers={
            "Authorization": f"Bearer {test_token}",
            "X-Tenant-ID": str(test_tenant.id),
        },
    )

    assert response.status_code == 404
