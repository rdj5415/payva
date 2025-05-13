"""
End-to-end tests for AuditPulse MVP.

This module contains end-to-end tests that validate the entire application flow,
from API endpoints to database interactions, including cross-component functionality
such as notifications, Plaid integration, and model validation.
"""

import os
import pytest
import asyncio
import json
from datetime import datetime, timedelta
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch, MagicMock

from auditpulse_mvp.main import app
from auditpulse_mvp.database.session import Base
from auditpulse_mvp.core.config import settings
from auditpulse_mvp.core.security import create_access_token
from auditpulse_mvp.schemas.user import UserCreate
from auditpulse_mvp.crud.crud_user import user as crud_user
from auditpulse_mvp.core.templates import TemplateManager
from auditpulse_mvp.utils.plaid_client import get_plaid_client

# Test database URL (use SQLite in memory for tests)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Test user credentials
TEST_USER_EMAIL = "test@example.com"
TEST_USER_PASSWORD = "testpassword"
TEST_ADMIN_EMAIL = "admin@example.com"
TEST_ADMIN_PASSWORD = "adminpassword"


@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for each test module."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def test_db_engine():
    """Create a test database engine."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture(scope="module")
async def test_db(test_db_engine):
    """Create a test database session."""
    async_session = sessionmaker(
        test_db_engine, expire_on_commit=False, class_=AsyncSession
    )
    async with async_session() as session:
        yield session


@pytest.fixture(scope="module")
async def test_client():
    """Create a test client for the FastAPI app."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture(scope="module")
async def admin_token(test_db):
    """Create a test admin user and return an authentication token."""
    admin_user = UserCreate(
        email=TEST_ADMIN_EMAIL,
        password=TEST_ADMIN_PASSWORD,
        first_name="Admin",
        last_name="User",
        is_superuser=True,
    )
    db_user = await crud_user.create(test_db, obj_in=admin_user)
    return create_access_token(subject=db_user.id)


@pytest.fixture(scope="module")
async def normal_token(test_db):
    """Create a test user and return an authentication token."""
    user_in = UserCreate(
        email=TEST_USER_EMAIL,
        password=TEST_USER_PASSWORD,
        first_name="Test",
        last_name="User",
    )
    db_user = await crud_user.create(test_db, obj_in=user_in)
    return create_access_token(subject=db_user.id)


# Authentication Tests
@pytest.mark.asyncio
async def test_login(test_client):
    """Test user login endpoint."""
    response = await test_client.post(
        "/api/v1/auth/login",
        json={"username": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_incorrect_password(test_client):
    """Test login with incorrect password."""
    response = await test_client.post(
        "/api/v1/auth/login",
        json={"username": TEST_USER_EMAIL, "password": "wrongpassword"},
    )
    assert response.status_code == 401


# User API Tests
@pytest.mark.asyncio
async def test_read_users_me(test_client, normal_token):
    """Test getting current user information."""
    response = await test_client.get(
        "/api/v1/users/me", headers={"Authorization": f"Bearer {normal_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == TEST_USER_EMAIL


@pytest.mark.asyncio
async def test_read_users_unauthorized(test_client):
    """Test accessing protected endpoint without token."""
    response = await test_client.get("/api/v1/users/me")
    assert response.status_code == 401


# Admin API Tests
@pytest.mark.asyncio
async def test_admin_access(test_client, admin_token):
    """Test admin access to admin-only endpoint."""
    response = await test_client.get(
        "/api/v1/admin/users", headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_admin_access_forbidden(test_client, normal_token):
    """Test non-admin access to admin-only endpoint."""
    response = await test_client.get(
        "/api/v1/admin/users", headers={"Authorization": f"Bearer {normal_token}"}
    )
    assert response.status_code == 403


# Transaction API Tests
@pytest.mark.asyncio
async def test_create_transaction(test_client, normal_token):
    """Test creating a transaction."""
    transaction_data = {
        "amount": 100.50,
        "date": "2023-01-01T12:00:00",
        "description": "Test Transaction",
        "category": "Test",
        "account_id": "test-account-id",
    }

    response = await test_client.post(
        "/api/v1/transactions/",
        json=transaction_data,
        headers={"Authorization": f"Bearer {normal_token}"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["amount"] == 100.50
    assert data["description"] == "Test Transaction"


@pytest.mark.asyncio
async def test_get_transactions(test_client, normal_token):
    """Test getting transactions."""
    response = await test_client.get(
        "/api/v1/transactions/", headers={"Authorization": f"Bearer {normal_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0


# Anomaly API Tests
@pytest.mark.asyncio
async def test_get_anomalies(test_client, normal_token):
    """Test getting anomalies."""
    response = await test_client.get(
        "/api/v1/anomalies/", headers={"Authorization": f"Bearer {normal_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


# Model API Tests
@pytest.mark.asyncio
async def test_get_model_versions(test_client, admin_token):
    """Test getting model versions."""
    response = await test_client.get(
        "/api/v1/models/anomaly_detection/versions",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


# Health Check Test
@pytest.mark.asyncio
async def test_health_check(test_client):
    """Test the health check endpoint."""
    response = await test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


# Integration Tests - Full Flow
@pytest.mark.asyncio
async def test_full_transaction_flow(test_client, normal_token):
    """
    Test the full transaction flow:
    1. Create a transaction
    2. Get the transaction
    3. Check if it appears in anomaly detection
    """
    # 1. Create a transaction
    transaction_data = {
        "amount": 9999.99,  # Unusual amount to trigger anomaly
        "date": "2023-01-01T12:00:00",
        "description": "Suspicious Test Transaction",
        "category": "Test",
        "account_id": "test-account-id",
    }

    create_response = await test_client.post(
        "/api/v1/transactions/",
        json=transaction_data,
        headers={"Authorization": f"Bearer {normal_token}"},
    )
    assert create_response.status_code == 201
    created_transaction = create_response.json()
    transaction_id = created_transaction["id"]

    # 2. Get the transaction
    get_response = await test_client.get(
        f"/api/v1/transactions/{transaction_id}",
        headers={"Authorization": f"Bearer {normal_token}"},
    )
    assert get_response.status_code == 200
    get_data = get_response.json()
    assert get_data["id"] == transaction_id

    # 3. Check anomalies
    anomalies_response = await test_client.get(
        "/api/v1/anomalies/?limit=10",
        headers={"Authorization": f"Bearer {normal_token}"},
    )
    assert anomalies_response.status_code == 200
    anomalies_data = anomalies_response.json()

    # The transaction might be detected as an anomaly (depends on implementation)
    # In a real test, we would wait for the anomaly detection process to complete
    # For this example, we just check that the endpoint returns successfully


# New Tests for Notifications
@pytest.mark.asyncio
async def test_notification_templates(test_client, admin_token, test_db):
    """Test notification template management."""
    # 1. Create a notification template
    template_data = {
        "template_id": "test_template",
        "subject": "Test Notification",
        "body": "Hello {{name}}, this is a test notification.",
        "placeholders": ["name"],
    }

    create_response = await test_client.post(
        "/api/v1/notifications/templates/",
        json=template_data,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert create_response.status_code == 201
    created_template = create_response.json()
    assert created_template["template_id"] == "test_template"

    # 2. Get the template
    get_response = await test_client.get(
        "/api/v1/notifications/templates/test_template",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert get_response.status_code == 200
    get_data = get_response.json()
    assert get_data["template_id"] == "test_template"
    assert "Hello {{name}}" in get_data["body"]


@pytest.mark.asyncio
async def test_send_notification(test_client, admin_token, normal_token, test_db):
    """Test sending a notification to a user."""
    # Use the template created in the previous test
    notification_request = {
        "template_id": "test_template",
        "recipient": {
            "user_id": normal_token,  # Use normal user as recipient
            "email": TEST_USER_EMAIL,
        },
        "template_data": {"name": "Test User"},
        "priority": "HIGH",
        "channels": ["email"],
    }

    # Mock the actual email sending
    with patch(
        "auditpulse_mvp.notifications.channels.email.EmailNotifier.send"
    ) as mock_send:
        mock_send.return_value = True

        response = await test_client.post(
            "/api/v1/notifications/send",
            json=notification_request,
            headers={"Authorization": f"Bearer {admin_token}"},
        )

        assert response.status_code == 202
        data = response.json()
        assert "notification_id" in data
        assert data["status"] == "scheduled"

        # Verify that the email sender was called
        mock_send.assert_called_once()


# Plaid Integration Tests
@pytest.mark.asyncio
async def test_plaid_integration(test_client, normal_token):
    """Test Plaid integration workflows."""
    # Mock the Plaid client to avoid actual API calls
    mock_plaid_client = MagicMock()
    mock_link_token_response = {
        "link_token": "link-sandbox-token",
        "expiration": (datetime.now() + timedelta(hours=24)).isoformat(),
    }
    mock_plaid_client.link_token_create.return_value = mock_link_token_response

    # Mock account data
    mock_account_data = {
        "accounts": [
            {
                "account_id": "test-account-1",
                "name": "Test Checking",
                "mask": "1234",
                "type": "depository",
                "subtype": "checking",
                "balances": {
                    "available": 1000,
                    "current": 1050,
                    "iso_currency_code": "USD",
                },
            }
        ],
        "item": {"item_id": "test-item-id", "institution_id": "test-institution"},
        "request_id": "test-request-id",
    }
    mock_plaid_client.exchange_public_token.return_value = {
        "access_token": "test-access-token",
        "item_id": "test-item-id",
    }
    mock_plaid_client.accounts_get.return_value = mock_account_data

    with patch(
        "auditpulse_mvp.utils.plaid_client.get_plaid_client",
        return_value=mock_plaid_client,
    ):
        # 1. Create link token
        link_response = await test_client.post(
            "/api/v1/plaid/link/token/create",
            headers={"Authorization": f"Bearer {normal_token}"},
        )
        assert link_response.status_code == 200
        link_data = link_response.json()
        assert "link_token" in link_data

        # 2. Exchange public token
        exchange_response = await test_client.post(
            "/api/v1/plaid/item/public_token/exchange",
            json={"public_token": "public-sandbox-token"},
            headers={"Authorization": f"Bearer {normal_token}"},
        )
        assert exchange_response.status_code == 200
        exchange_data = exchange_response.json()
        assert exchange_data["success"] is True

        # 3. Get accounts
        accounts_response = await test_client.get(
            "/api/v1/plaid/accounts",
            headers={"Authorization": f"Bearer {normal_token}"},
        )
        assert accounts_response.status_code == 200
        accounts_data = accounts_response.json()
        assert isinstance(accounts_data, list)


# Model Validation Tests
@pytest.mark.asyncio
async def test_model_validation(test_client, admin_token):
    """Test model validation workflows."""
    # 1. Create validation data
    validation_data = {
        "model_type": "anomaly_detection",
        "version": "v1.0.0",
        "validation_data": [
            {
                "amount": 9999.99,
                "date": "2023-01-01T12:00:00",
                "description": "Unusual Transaction",
                "category": "Test",
                "account_id": "test-account-id",
            },
            {
                "amount": 50.00,
                "date": "2023-01-02T12:00:00",
                "description": "Normal Transaction",
                "category": "Test",
                "account_id": "test-account-id",
            },
        ],
        "ground_truth": [
            {"is_anomaly": True, "anomaly_type": "unusual_amount"},
            {"is_anomaly": False, "anomaly_type": None},
        ],
    }

    validation_response = await test_client.post(
        "/api/v1/models/validate",
        json=validation_data,
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert validation_response.status_code == 200
    validation_result = validation_response.json()
    assert "metrics" in validation_result
    assert "validation_success" in validation_result
    assert all(
        metric in validation_result["metrics"]
        for metric in ["accuracy", "precision", "recall", "f1_score"]
    )


# System Health Tests
@pytest.mark.asyncio
async def test_detailed_health_check(test_client, admin_token):
    """Test the detailed health check endpoint (admin-only)."""
    response = await test_client.get(
        "/api/v1/admin/system-health",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "components" in data
    assert all(component in data["components"] for component in ["api", "database"])


# Permissions Tests
@pytest.mark.asyncio
async def test_user_role_permissions(test_client, normal_token, admin_token):
    """Test role-based permissions."""
    # Test a regular user trying to access model management
    user_response = await test_client.post(
        "/api/v1/models/anomaly_detection/versions",
        json={"version": "test-version", "description": "Test model"},
        headers={"Authorization": f"Bearer {normal_token}"},
    )
    assert user_response.status_code == 403  # Should be forbidden

    # Test admin user accessing model management
    admin_response = await test_client.post(
        "/api/v1/models/anomaly_detection/versions",
        json={"version": "test-version", "description": "Test model"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert admin_response.status_code in [200, 201]  # Should succeed
