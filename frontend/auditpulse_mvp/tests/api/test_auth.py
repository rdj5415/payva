"""Tests for authentication API endpoints.

This module contains tests for the authentication API endpoints.
"""

import uuid
from typing import Dict, Any, List

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.models import User, Tenant
from auditpulse_mvp.api.api_v1.endpoints.auth import UserRole


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
async def test_auditor_user(db: AsyncSession, test_tenant: Tenant) -> User:
    """Create a test auditor user.

    Args:
        db: Database session
        test_tenant: Test tenant

    Returns:
        User: Auditor user for testing
    """
    auditor_user = User(
        id=uuid.uuid4(),
        tenant_id=test_tenant.id,
        email="auditor@example.com",
        hashed_password="hashed_password",
        full_name="Auditor User",
        role="auditor",
    )
    db.add(auditor_user)
    await db.commit()
    await db.refresh(auditor_user)
    return auditor_user


@pytest.fixture
async def test_auditor_token(test_auditor_user: User) -> str:
    """Create a token for the auditor user.

    Args:
        test_auditor_user: Auditor user

    Returns:
        str: JWT token for the auditor user
    """
    # In a real test, this would generate a valid JWT
    # For simplicity, we'll use a mock token
    return "auditor_token"


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


async def test_auth_settings(
    async_client: AsyncClient,
):
    """Test getting authentication settings.

    Args:
        async_client: Async test client
    """
    # Make request to get auth settings
    response = await async_client.get("/api/v1/auth/settings")

    # Check response
    assert response.status_code == 200
    data = response.json()

    # Validate response structure
    assert "auth0_domain" in data
    assert "auth0_client_id" in data
    assert "auth0_audience" in data
    assert "login_providers" in data

    # Verify login providers
    assert isinstance(data["login_providers"], list)
    assert len(data["login_providers"]) > 0


async def test_get_user_profile(
    async_client: AsyncClient,
    test_admin_token: str,
    test_admin_user: User,
    test_tenant: Tenant,
):
    """Test getting user profile.

    Args:
        async_client: Async test client
        test_admin_token: Admin user token
        test_admin_user: Admin user
        test_tenant: Test tenant
    """
    # Make request to get user profile
    response = await async_client.get(
        "/api/v1/auth/me",
        headers={
            "Authorization": f"Bearer {test_admin_token}",
            "X-Tenant-ID": str(test_tenant.id),
        },
    )

    # Check response
    assert response.status_code == 200
    data = response.json()

    # Validate response structure
    assert "id" in data
    assert "email" in data
    assert "full_name" in data
    assert "role" in data
    assert "tenant_id" in data
    assert "permissions" in data

    # Validate permissions for admin
    assert isinstance(data["permissions"], list)
    assert "read:anomalies" in data["permissions"]
    assert "write:anomalies" in data["permissions"]
    assert "read:transactions" in data["permissions"]
    assert "write:users" in data["permissions"]
    assert "write:settings" in data["permissions"]


async def test_update_user_profile(
    async_client: AsyncClient,
    test_admin_token: str,
    test_admin_user: User,
    test_tenant: Tenant,
    db: AsyncSession,
):
    """Test updating user profile.

    Args:
        async_client: Async test client
        test_admin_token: Admin user token
        test_admin_user: Admin user
        test_tenant: Test tenant
        db: Database session
    """
    # Make request to update user profile
    response = await async_client.patch(
        "/api/v1/auth/me",
        json={
            "full_name": "Updated Admin Name",
            "email_notifications": False,
            "slack_notifications": True,
            "slack_user_id": "U1234567890",
        },
        headers={
            "Authorization": f"Bearer {test_admin_token}",
            "X-Tenant-ID": str(test_tenant.id),
        },
    )

    # Check response
    assert response.status_code == 200
    data = response.json()

    # Validate updated fields
    assert data["full_name"] == "Updated Admin Name"
    assert data["email_notifications"] is False
    assert data["slack_notifications"] is True
    assert data["slack_user_id"] == "U1234567890"

    # Refresh user from database to verify changes were persisted
    await db.refresh(test_admin_user)
    assert test_admin_user.full_name == "Updated Admin Name"
    assert test_admin_user.email_notifications is False
    assert test_admin_user.slack_notifications is True
    assert test_admin_user.slack_user_id == "U1234567890"


async def test_role_permissions(
    async_client: AsyncClient,
    test_admin_token: str,
    test_auditor_token: str,
    test_viewer_token: str,
    test_tenant: Tenant,
):
    """Test role-based permissions.

    Args:
        async_client: Async test client
        test_admin_token: Admin user token
        test_auditor_token: Auditor user token
        test_viewer_token: Viewer user token
        test_tenant: Test tenant
    """
    # Test admin permissions
    admin_response = await async_client.get(
        "/api/v1/auth/me",
        headers={
            "Authorization": f"Bearer {test_admin_token}",
            "X-Tenant-ID": str(test_tenant.id),
        },
    )

    admin_data = admin_response.json()
    admin_permissions = set(admin_data["permissions"])

    # Test auditor permissions
    auditor_response = await async_client.get(
        "/api/v1/auth/me",
        headers={
            "Authorization": f"Bearer {test_auditor_token}",
            "X-Tenant-ID": str(test_tenant.id),
        },
    )

    auditor_data = auditor_response.json()
    auditor_permissions = set(auditor_data["permissions"])

    # Test viewer permissions
    viewer_response = await async_client.get(
        "/api/v1/auth/me",
        headers={
            "Authorization": f"Bearer {test_viewer_token}",
            "X-Tenant-ID": str(test_tenant.id),
        },
    )

    viewer_data = viewer_response.json()
    viewer_permissions = set(viewer_data["permissions"])

    # Verify permission hierarchy
    assert viewer_permissions.issubset(auditor_permissions)
    assert auditor_permissions.issubset(admin_permissions)

    # Verify specific permissions
    assert "read:anomalies" in viewer_permissions
    assert "write:anomalies" in auditor_permissions
    assert "write:users" in admin_permissions
    assert "write:settings" in admin_permissions

    assert "write:anomalies" not in viewer_permissions
    assert "write:settings" not in auditor_permissions


async def test_auth0_callback(
    async_client: AsyncClient,
):
    """Test Auth0 callback handling.

    Args:
        async_client: Async test client
    """
    # Make request to Auth0 callback endpoint
    response = await async_client.post(
        "/api/v1/auth/callback?code=test_code&state=test_state"
    )

    # Check response
    assert response.status_code == 200
    data = response.json()

    # Validate response structure
    assert "success" in data
    assert "message" in data
    assert "next" in data

    # Validate response data
    assert data["success"] is True
    assert "Auth0 callback processed" in data["message"]
    assert data["next"] == "/dashboard"
