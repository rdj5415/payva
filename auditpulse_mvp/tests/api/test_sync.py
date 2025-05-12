"""Tests for synchronization API endpoints.

This module contains tests for the data synchronization API endpoints.
"""
import uuid
from typing import Dict, Any, Optional

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.models import User, Tenant, TenantConfiguration


@pytest.fixture
async def mock_quickbooks_config(db: AsyncSession, test_tenant: Tenant) -> TenantConfiguration:
    """Create a mock QuickBooks configuration for testing.
    
    Args:
        db: Database session
        test_tenant: Test tenant
        
    Returns:
        TenantConfiguration: QuickBooks configuration
    """
    config = TenantConfiguration(
        tenant_id=test_tenant.id,
        key="quickbooks_config",
        value={
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "realm_id": "test_realm_id",
            "refresh_token": "test_refresh_token",
            "company_id": "test_company_id",
            "environment": "sandbox",
            "last_sync": "2023-01-01T00:00:00Z",
        }
    )
    db.add(config)
    await db.commit()
    await db.refresh(config)
    return config


@pytest.fixture
async def mock_plaid_config(db: AsyncSession, test_tenant: Tenant) -> TenantConfiguration:
    """Create a mock Plaid configuration for testing.
    
    Args:
        db: Database session
        test_tenant: Test tenant
        
    Returns:
        TenantConfiguration: Plaid configuration
    """
    config = TenantConfiguration(
        tenant_id=test_tenant.id,
        key="plaid_config",
        value={
            "client_id": "test_client_id",
            "access_tokens": {
                "account1": "access_token_1",
                "account2": "access_token_2"
            },
            "item_ids": {
                "account1": "item_id_1",
                "account2": "item_id_2"
            },
            "environment": "sandbox",
            "last_sync": "2023-01-01T00:00:00Z",
        }
    )
    db.add(config)
    await db.commit()
    await db.refresh(config)
    return config


async def test_quickbooks_sync(
    async_client: AsyncClient,
    test_token: str,
    test_tenant: Tenant,
    mock_quickbooks_config: TenantConfiguration,
):
    """Test QuickBooks synchronization endpoint.
    
    Args:
        async_client: Async test client
        test_token: Authentication token
        test_tenant: Test tenant
        mock_quickbooks_config: Mock QuickBooks configuration
    """
    # Make request to sync QuickBooks
    response = await async_client.post(
        "/api/v1/sync/quickbooks",
        json={
            "force_full_sync": False,
            "company_id": "test_company_id",
            "start_date": "2023-01-01",
            "end_date": "2023-03-31"
        },
        headers={
            "Authorization": f"Bearer {test_token}",
            "X-Tenant-ID": str(test_tenant.id),
        }
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Validate response structure
    assert "success" in data
    assert "message" in data
    assert "status" in data
    assert "details" in data
    
    # Validate response data
    assert data["success"] is True
    assert "QuickBooks sync started" in data["message"]
    assert data["status"] == "pending"
    assert "company_id" in data["details"]
    assert data["details"]["company_id"] == "test_company_id"


async def test_plaid_sync(
    async_client: AsyncClient,
    test_token: str,
    test_tenant: Tenant,
    mock_plaid_config: TenantConfiguration,
):
    """Test Plaid synchronization endpoint.
    
    Args:
        async_client: Async test client
        test_token: Authentication token
        test_tenant: Test tenant
        mock_plaid_config: Mock Plaid configuration
    """
    # Make request to sync Plaid
    response = await async_client.post(
        "/api/v1/sync/plaid",
        json={
            "force_full_sync": False,
            "account_ids": ["account1", "account2"],
            "start_date": "2023-01-01",
            "end_date": "2023-03-31"
        },
        headers={
            "Authorization": f"Bearer {test_token}",
            "X-Tenant-ID": str(test_tenant.id),
        }
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Validate response structure
    assert "success" in data
    assert "message" in data
    assert "status" in data
    assert "details" in data
    
    # Validate response data
    assert data["success"] is True
    assert "Plaid sync started" in data["message"]
    assert data["status"] == "pending"
    assert "accounts" in data["details"]
    assert data["details"]["accounts"] == ["account1", "account2"]


async def test_quickbooks_sync_no_config(
    async_client: AsyncClient,
    test_token: str,
    test_tenant: Tenant,
):
    """Test QuickBooks sync without configuration.
    
    Args:
        async_client: Async test client
        test_token: Authentication token
        test_tenant: Test tenant
    """
    # Make request to sync QuickBooks without config
    response = await async_client.post(
        "/api/v1/sync/quickbooks",
        json={
            "force_full_sync": False
        },
        headers={
            "Authorization": f"Bearer {test_token}",
            "X-Tenant-ID": str(test_tenant.id),
        }
    )
    
    # In development mode, the API should still accept the request
    # but log a warning about missing configuration
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


async def test_plaid_sync_no_config(
    async_client: AsyncClient,
    test_token: str,
    test_tenant: Tenant,
):
    """Test Plaid sync without configuration.
    
    Args:
        async_client: Async test client
        test_token: Authentication token
        test_tenant: Test tenant
    """
    # Make request to sync Plaid without config
    response = await async_client.post(
        "/api/v1/sync/plaid",
        json={
            "force_full_sync": False
        },
        headers={
            "Authorization": f"Bearer {test_token}",
            "X-Tenant-ID": str(test_tenant.id),
        }
    )
    
    # In development mode, the API should still accept the request
    # but log a warning about missing configuration
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True 