"""Tests for configuration API endpoints.

This module contains tests for the configuration API endpoints.
"""

import uuid
from typing import Dict, Any, Tuple

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.models import User, Tenant, TenantConfiguration
from auditpulse_mvp.api.api_v1.endpoints.config import (
    SensitivityLevel,
    get_preset_configuration,
)


@pytest.fixture
async def test_admin_user(db: AsyncSession, test_tenant: Tenant) -> User:
    """Create a test admin user.

    Args:
        db: Database session
        test_tenant: Test tenant

    Returns:
        User: Admin user for testing
    """
    admin_user = User(
        id=uuid.uuid4(),
        tenant_id=test_tenant.id,
        email="admin@example.com",
        hashed_password="hashed_password",
        full_name="Admin User",
        role="admin",
    )
    db.add(admin_user)
    await db.commit()
    await db.refresh(admin_user)
    return admin_user


@pytest.fixture
async def test_admin_token(test_admin_user: User) -> str:
    """Create a token for the admin user.

    Args:
        test_admin_user: Admin user

    Returns:
        str: JWT token for the admin user
    """
    # In a real test, this would generate a valid JWT
    # For simplicity, we'll use a mock token
    return "admin_token"


@pytest.fixture
async def test_viewer_user(db: AsyncSession, test_tenant: Tenant) -> User:
    """Create a test viewer user.

    Args:
        db: Database session
        test_tenant: Test tenant

    Returns:
        User: Viewer user for testing
    """
    viewer_user = User(
        id=uuid.uuid4(),
        tenant_id=test_tenant.id,
        email="viewer@example.com",
        hashed_password="hashed_password",
        full_name="Viewer User",
        role="viewer",
    )
    db.add(viewer_user)
    await db.commit()
    await db.refresh(viewer_user)
    return viewer_user


@pytest.fixture
async def test_viewer_token(test_viewer_user: User) -> str:
    """Create a token for the viewer user.

    Args:
        test_viewer_user: Viewer user

    Returns:
        str: JWT token for the viewer user
    """
    # In a real test, this would generate a valid JWT
    # For simplicity, we'll use a mock token
    return "viewer_token"


async def test_get_sensitivity_config(
    async_client: AsyncClient,
    test_token: str,
    test_tenant: Tenant,
    db: AsyncSession,
):
    """Test getting sensitivity configuration.

    Args:
        async_client: Async test client
        test_token: Authentication token
        test_tenant: Test tenant
        db: Database session
    """
    # Make request to get config
    response = await async_client.get(
        "/api/v1/config/sensitivity",
        headers={
            "Authorization": f"Bearer {test_token}",
            "X-Tenant-ID": str(test_tenant.id),
        },
    )

    # Check response
    assert response.status_code == 200
    data = response.json()

    # Validate response structure
    assert "sensitivity_level" in data
    assert "risk_engine" in data
    assert "rules" in data

    # Default should be medium sensitivity
    assert data["sensitivity_level"] == SensitivityLevel.MEDIUM.value


async def test_update_sensitivity_config(
    async_client: AsyncClient,
    test_admin_token: str,
    test_viewer_token: str,
    test_tenant: Tenant,
    db: AsyncSession,
):
    """Test updating sensitivity configuration.

    Args:
        async_client: Async test client
        test_admin_token: Admin authentication token
        test_viewer_token: Viewer authentication token
        test_tenant: Test tenant
        db: Database session
    """
    # Test that viewer cannot update config
    viewer_response = await async_client.put(
        "/api/v1/config/sensitivity",
        json={
            "sensitivity_level": "high",
            "risk_engine": {
                "ml_threshold": 0.6,
                "rules_score_weight": 0.5,
                "ml_score_weight": 0.5,
                "min_transaction_amount": 50.0,
            },
        },
        headers={
            "Authorization": f"Bearer {test_viewer_token}",
            "X-Tenant-ID": str(test_tenant.id),
        },
    )

    # Should get 403 Forbidden
    assert viewer_response.status_code == 403

    # Test that admin can update config
    admin_response = await async_client.put(
        "/api/v1/config/sensitivity",
        json={
            "sensitivity_level": "high",
            "risk_engine": {
                "ml_threshold": 0.6,
                "rules_score_weight": 0.5,
                "ml_score_weight": 0.5,
                "min_transaction_amount": 50.0,
            },
        },
        headers={
            "Authorization": f"Bearer {test_admin_token}",
            "X-Tenant-ID": str(test_tenant.id),
        },
    )

    # Should be successful
    assert admin_response.status_code == 200
    data = admin_response.json()

    # Validate response
    assert data["sensitivity_level"] == "high"
    assert data["risk_engine"]["ml_threshold"] == 0.6

    # Verify database was updated
    config_query = await db.execute(
        """
        SELECT * FROM tenant_configurations
        WHERE tenant_id = :tenant_id AND key = 'sensitivity_config'
        """,
        {"tenant_id": str(test_tenant.id)},
    )
    config = config_query.first()
    assert config is not None

    # Test updating to custom configuration
    custom_response = await async_client.put(
        "/api/v1/config/sensitivity",
        json={
            "sensitivity_level": "custom",
            "risk_engine": {
                "ml_threshold": 0.75,
                "rules_score_weight": 0.7,
                "ml_score_weight": 0.3,
                "min_transaction_amount": 200.0,
            },
            "rules": {
                "large_transaction": {
                    "enabled": True,
                    "threshold": 15000.0,
                    "parameters": {"scale_factor": 1.1},
                },
                "unusual_counterparty": {
                    "enabled": True,
                    "parameters": {"min_frequency": 2},
                },
                "weekend_transaction": {"enabled": False},
                "irregular_amount": {
                    "enabled": True,
                    "threshold": 2.5,
                    "parameters": {"std_dev_threshold": 2.5},
                },
                "round_number_transaction": {
                    "enabled": True,
                    "parameters": {"score_multiplier": 1.3},
                },
            },
            "custom_settings": {"notification_threshold": 75},
        },
        headers={
            "Authorization": f"Bearer {test_admin_token}",
            "X-Tenant-ID": str(test_tenant.id),
        },
    )

    # Should be successful
    assert custom_response.status_code == 200
    custom_data = custom_response.json()

    # Validate custom response
    assert custom_data["sensitivity_level"] == "custom"
    assert custom_data["risk_engine"]["ml_threshold"] == 0.75
    assert custom_data["rules"]["weekend_transaction"]["enabled"] == False
    assert custom_data["custom_settings"]["notification_threshold"] == 75


async def test_preset_configurations(
    async_client: AsyncClient,
    test_admin_token: str,
    test_tenant: Tenant,
):
    """Test that preset configurations are correctly applied.

    Args:
        async_client: Async test client
        test_admin_token: Admin authentication token
        test_tenant: Test tenant
    """
    # Test low sensitivity preset
    low_response = await async_client.put(
        "/api/v1/config/sensitivity",
        json={"sensitivity_level": "low"},
        headers={
            "Authorization": f"Bearer {test_admin_token}",
            "X-Tenant-ID": str(test_tenant.id),
        },
    )

    assert low_response.status_code == 200
    low_data = low_response.json()
    assert low_data["sensitivity_level"] == "low"
    assert low_data["risk_engine"]["ml_threshold"] == 0.85

    # Test medium sensitivity preset
    medium_response = await async_client.put(
        "/api/v1/config/sensitivity",
        json={"sensitivity_level": "medium"},
        headers={
            "Authorization": f"Bearer {test_admin_token}",
            "X-Tenant-ID": str(test_tenant.id),
        },
    )

    assert medium_response.status_code == 200
    medium_data = medium_response.json()
    assert medium_data["sensitivity_level"] == "medium"
    assert medium_data["risk_engine"]["ml_threshold"] == 0.7

    # Test high sensitivity preset
    high_response = await async_client.put(
        "/api/v1/config/sensitivity",
        json={"sensitivity_level": "high"},
        headers={
            "Authorization": f"Bearer {test_admin_token}",
            "X-Tenant-ID": str(test_tenant.id),
        },
    )

    assert high_response.status_code == 200
    high_data = high_response.json()
    assert high_data["sensitivity_level"] == "high"
    assert high_data["risk_engine"]["ml_threshold"] == 0.6
    assert high_data["rules"]["large_transaction"]["threshold"] == 5000.0
