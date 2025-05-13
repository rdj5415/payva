"""Tests for the QuickBooks ingestion module.

This module contains tests for the QuickBooks OAuth2, data fetching,
normalization, and webhook handling functionality.
"""

import datetime
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.models import DataSource, Tenant, Transaction
from auditpulse_mvp.ingestion.quickbooks import (
    QuickBooksClient,
    QuickBooksService,
    handle_quickbooks_webhook,
    update_tenant_quickbooks_settings,
)


@pytest.fixture
def mock_tenant_id() -> uuid.UUID:
    """Return a mock tenant ID."""
    return uuid.UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def mock_qb_settings() -> dict:
    """Return mock QuickBooks settings."""
    return {
        "access_token": "mock_access_token",
        "refresh_token": "mock_refresh_token",
        "realm_id": "12345",
    }


@pytest.fixture
async def mock_tenant(
    db_session: AsyncSession, mock_tenant_id: uuid.UUID, mock_qb_settings: dict
) -> Tenant:
    """Create a mock tenant with QuickBooks settings."""
    tenant = Tenant(
        id=mock_tenant_id,
        name="Test Tenant",
        slug="test-tenant-qb",
        quickbooks_settings=mock_qb_settings,
    )
    db_session.add(tenant)
    await db_session.flush()
    return tenant


@pytest.fixture
def mock_qb_transaction() -> dict:
    """Return a mock QuickBooks transaction."""
    return {
        "Id": "123",
        "PaymentType": "CreditCard",
        "AccountRef": {"value": "456", "name": "Credit Card"},
        "EntityRef": {"value": "789", "name": "Vendor XYZ"},
        "TotalAmt": 100.50,
        "TxnDate": "2023-01-15",
        "PrivateNote": "Test transaction",
        "Line": [
            {
                "Description": "Item 1",
                "AccountBasedExpenseLineDetail": {
                    "AccountRef": {"value": "101", "name": "Office Supplies"}
                },
            }
        ],
    }


@pytest.mark.asyncio
async def test_qb_client_init():
    """Test QuickBooks client initialization."""
    tenant_id = uuid.uuid4()

    with patch("auditpulse_mvp.ingestion.quickbooks.settings") as mock_settings:
        mock_settings.QUICKBOOKS_CLIENT_ID = "test_client_id"
        mock_settings.QUICKBOOKS_CLIENT_SECRET.get_secret_value.return_value = (
            "test_secret"
        )
        mock_settings.QUICKBOOKS_REDIRECT_URI = "https://example.com/callback"
        mock_settings.QUICKBOOKS_ENVIRONMENT = "sandbox"

        client = QuickBooksClient(tenant_id=tenant_id)

        assert client.tenant_id == tenant_id
        assert client.client_id == "test_client_id"
        assert client.client_secret == "test_secret"
        assert client.redirect_uri == "https://example.com/callback"
        assert client.environment == "sandbox"


@pytest.mark.asyncio
async def test_qb_client_get_authorization_url():
    """Test getting the QuickBooks authorization URL."""
    tenant_id = uuid.uuid4()

    with patch("auditpulse_mvp.ingestion.quickbooks.settings") as mock_settings, patch(
        "auditpulse_mvp.ingestion.quickbooks.AuthClient"
    ) as MockAuthClient:
        mock_settings.QUICKBOOKS_CLIENT_ID = "test_client_id"
        mock_settings.QUICKBOOKS_CLIENT_SECRET.get_secret_value.return_value = (
            "test_secret"
        )
        mock_settings.QUICKBOOKS_REDIRECT_URI = "https://example.com/callback"

        mock_auth_client = MagicMock()
        mock_auth_client.get_authorization_url.return_value = (
            "https://oauth.platform.intuit.com/oauth2/v1/authorize?..."
        )
        MockAuthClient.return_value = mock_auth_client

        client = QuickBooksClient(tenant_id=tenant_id)
        auth_url = client.get_authorization_url("test_state")

        assert "https://oauth.platform.intuit.com" in auth_url
        mock_auth_client.get_authorization_url.assert_called_once()


@pytest.mark.asyncio
async def test_qb_client_exchange_code_for_token():
    """Test exchanging auth code for tokens."""
    tenant_id = uuid.uuid4()

    with patch("auditpulse_mvp.ingestion.quickbooks.settings") as mock_settings, patch(
        "auditpulse_mvp.ingestion.quickbooks.AuthClient"
    ) as MockAuthClient:
        mock_settings.QUICKBOOKS_CLIENT_ID = "test_client_id"
        mock_settings.QUICKBOOKS_CLIENT_SECRET.get_secret_value.return_value = (
            "test_secret"
        )
        mock_settings.QUICKBOOKS_REDIRECT_URI = "https://example.com/callback"

        mock_auth_client = MagicMock()
        mock_auth_client.access_token = "new_access_token"
        mock_auth_client.refresh_token = "new_refresh_token"
        MockAuthClient.return_value = mock_auth_client

        client = QuickBooksClient(tenant_id=tenant_id)
        tokens = await client.exchange_code_for_token("test_code", "12345")

        assert tokens["access_token"] == "new_access_token"
        assert tokens["refresh_token"] == "new_refresh_token"
        assert tokens["realm_id"] == "12345"
        mock_auth_client.get_bearer_token.assert_called_once_with("test_code", "12345")


@pytest.mark.asyncio
async def test_qb_client_fetch_transactions_success():
    """Test fetching transactions successfully."""
    tenant_id = uuid.uuid4()

    with patch("auditpulse_mvp.ingestion.quickbooks.settings") as mock_settings, patch(
        "httpx.AsyncClient"
    ) as MockAsyncClient:
        mock_settings.QUICKBOOKS_CLIENT_ID = "test_client_id"
        mock_settings.QUICKBOOKS_CLIENT_SECRET.get_secret_value.return_value = (
            "test_secret"
        )
        mock_settings.QUICKBOOKS_REDIRECT_URI = "https://example.com/callback"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "QueryResponse": {"Purchase": [{"Id": "123", "TotalAmt": 100.00}]}
        }

        mock_async_client = AsyncMock()
        mock_async_client.__aenter__.return_value.post.return_value = mock_response
        MockAsyncClient.return_value = mock_async_client

        client = QuickBooksClient(tenant_id=tenant_id)
        client.access_token = "test_access_token"
        client.realm_id = "12345"

        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 2, 1)

        transactions = await client.fetch_transactions(start_date, end_date)

        assert transactions == [{"Id": "123", "TotalAmt": 100.00}]
        mock_async_client.__aenter__.return_value.post.assert_called_once()
        assert (
            "2023-01-01"
            in mock_async_client.__aenter__.return_value.post.call_args[1]["data"]
        )
        assert (
            "2023-02-01"
            in mock_async_client.__aenter__.return_value.post.call_args[1]["data"]
        )


@pytest.mark.asyncio
async def test_qb_client_fetch_transactions_http_error():
    """Test fetching transactions with HTTP error."""
    tenant_id = uuid.uuid4()

    with patch("auditpulse_mvp.ingestion.quickbooks.settings") as mock_settings, patch(
        "httpx.AsyncClient"
    ) as MockAsyncClient, patch.object(
        QuickBooksClient, "refresh_access_token"
    ) as mock_refresh:
        mock_settings.QUICKBOOKS_CLIENT_ID = "test_client_id"
        mock_settings.QUICKBOOKS_CLIENT_SECRET.get_secret_value.return_value = (
            "test_secret"
        )
        mock_settings.QUICKBOOKS_REDIRECT_URI = "https://example.com/callback"

        # Mock HTTP 401 error
        mock_response = MagicMock()
        http_error = httpx.HTTPStatusError(
            "401 Unauthorized",
            request=MagicMock(),
            response=MagicMock(status_code=401),
        )
        mock_response.raise_for_status.side_effect = http_error

        mock_async_client = AsyncMock()
        mock_async_client.__aenter__.return_value.post.return_value = mock_response
        MockAsyncClient.return_value = mock_async_client

        mock_refresh.return_value = {"access_token": "new_token"}

        # Setup retry on second attempt
        mock_async_client.__aenter__.return_value.post.side_effect = [
            mock_response,  # First call raises 401
            MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(
                    return_value={"QueryResponse": {"Purchase": [{"Id": "123"}]}}
                ),
            ),  # Second call succeeds
        ]

        client = QuickBooksClient(tenant_id=tenant_id)
        client.access_token = "test_access_token"
        client.realm_id = "12345"
        client.refresh_token = "test_refresh_token"

        # Should retry and succeed
        with patch("auditpulse_mvp.ingestion.quickbooks.retry", lambda f, **kwargs: f):
            with pytest.raises(HTTPException):
                await client.fetch_transactions()


@pytest.mark.asyncio
async def test_qb_client_refresh_access_token():
    """Test refreshing access token."""
    tenant_id = uuid.uuid4()

    with patch("auditpulse_mvp.ingestion.quickbooks.settings") as mock_settings, patch(
        "auditpulse_mvp.ingestion.quickbooks.AuthClient"
    ) as MockAuthClient:
        mock_settings.QUICKBOOKS_CLIENT_ID = "test_client_id"
        mock_settings.QUICKBOOKS_CLIENT_SECRET.get_secret_value.return_value = (
            "test_secret"
        )
        mock_settings.QUICKBOOKS_REDIRECT_URI = "https://example.com/callback"

        mock_auth_client = MagicMock()
        mock_auth_client.access_token = "new_access_token"
        mock_auth_client.refresh_token = "new_refresh_token"
        MockAuthClient.return_value = mock_auth_client

        client = QuickBooksClient(tenant_id=tenant_id)
        client.refresh_token = "old_refresh_token"
        client.realm_id = "12345"

        tokens = await client.refresh_access_token()

        assert tokens["access_token"] == "new_access_token"
        assert tokens["refresh_token"] == "new_refresh_token"
        assert tokens["realm_id"] == "12345"
        mock_auth_client.refresh.assert_called_once_with(
            refresh_token="old_refresh_token"
        )


@pytest.mark.asyncio
async def test_qb_service_get_client_for_tenant(
    db_session: AsyncSession, mock_tenant: Tenant, mock_tenant_id: uuid.UUID
):
    """Test getting a QuickBooks client for a tenant."""
    with patch("auditpulse_mvp.ingestion.quickbooks.settings") as mock_settings:
        mock_settings.QUICKBOOKS_CLIENT_ID = "test_client_id"
        mock_settings.QUICKBOOKS_CLIENT_SECRET.get_secret_value.return_value = (
            "test_secret"
        )
        mock_settings.QUICKBOOKS_REDIRECT_URI = "https://example.com/callback"

        service = QuickBooksService(db_session)
        client = await service.get_client_for_tenant(mock_tenant_id)

        assert client.tenant_id == mock_tenant_id
        assert client.access_token == "mock_access_token"
        assert client.refresh_token == "mock_refresh_token"
        assert client.realm_id == "12345"


@pytest.mark.asyncio
async def test_qb_service_get_client_for_nonexistent_tenant(db_session: AsyncSession):
    """Test getting a client for a nonexistent tenant."""
    nonexistent_id = uuid.uuid4()

    service = QuickBooksService(db_session)

    with pytest.raises(HTTPException) as excinfo:
        await service.get_client_for_tenant(nonexistent_id)

    assert excinfo.value.status_code == 404
    assert f"Tenant {nonexistent_id} not found" in str(excinfo.value.detail)


@pytest.mark.asyncio
async def test_qb_service_get_client_for_tenant_without_qb(
    db_session: AsyncSession, mock_tenant_id: uuid.UUID
):
    """Test getting a client for a tenant without QuickBooks configured."""
    # Create tenant without QuickBooks settings
    tenant = Tenant(
        id=mock_tenant_id,
        name="Test Tenant No QB",
        slug="test-tenant-no-qb",
    )
    db_session.add(tenant)
    await db_session.flush()

    service = QuickBooksService(db_session)

    with pytest.raises(HTTPException) as excinfo:
        await service.get_client_for_tenant(mock_tenant_id)

    assert excinfo.value.status_code == 400
    assert f"QuickBooks not configured for tenant {mock_tenant_id}" in str(
        excinfo.value.detail
    )


@pytest.mark.asyncio
async def test_qb_service_sync_transactions(
    db_session: AsyncSession,
    mock_tenant: Tenant,
    mock_tenant_id: uuid.UUID,
    mock_qb_transaction: dict,
):
    """Test syncing transactions from QuickBooks."""
    with patch.object(QuickBooksClient, "fetch_transactions") as mock_fetch:
        mock_fetch.return_value = [mock_qb_transaction]

        service = QuickBooksService(db_session)

        # Patch get_client_for_tenant to return a mocked client
        with patch.object(
            QuickBooksService, "get_client_for_tenant"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client.fetch_transactions = mock_fetch
            mock_get_client.return_value = mock_client

            # Call sync_transactions
            fetched, created, updated = await service.sync_transactions(
                mock_tenant_id,
                start_date=datetime.date(2023, 1, 1),
                end_date=datetime.date(2023, 2, 1),
            )

            # Check results
            assert fetched == 1
            assert created == 1
            assert updated == 0

            # Verify transaction was created in DB
            result = await db_session.execute(
                select(Transaction).where(
                    Transaction.tenant_id == mock_tenant_id,
                    Transaction.source == DataSource.QUICKBOOKS,
                )
            )
            transaction = result.scalar_one()

            assert transaction.transaction_id == "123"
            assert transaction.amount == 100.50
            assert transaction.description == "Test transaction"
            assert transaction.category == "Office Supplies"
            assert transaction.merchant_name == "Vendor XYZ"


@pytest.mark.asyncio
async def test_qb_service_sync_transactions_update_existing(
    db_session: AsyncSession,
    mock_tenant: Tenant,
    mock_tenant_id: uuid.UUID,
    mock_qb_transaction: dict,
):
    """Test syncing transactions that already exist."""
    # Create an existing transaction
    existing_txn = Transaction(
        tenant_id=mock_tenant_id,
        transaction_id="123",
        source=DataSource.QUICKBOOKS,
        source_account_id="old-account",
        amount=50.25,
        currency="USD",
        description="Old description",
        category="Old category",
        merchant_name="Old vendor",
        transaction_date=datetime.datetime(2023, 1, 1),
    )
    db_session.add(existing_txn)
    await db_session.flush()

    with patch.object(QuickBooksClient, "fetch_transactions") as mock_fetch:
        # Return the same transaction with updated values
        mock_fetch.return_value = [mock_qb_transaction]

        service = QuickBooksService(db_session)

        # Patch get_client_for_tenant to return a mocked client
        with patch.object(
            QuickBooksService, "get_client_for_tenant"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client.fetch_transactions = mock_fetch
            mock_get_client.return_value = mock_client

            # Call sync_transactions
            fetched, created, updated = await service.sync_transactions(mock_tenant_id)

            # Check results
            assert fetched == 1
            assert created == 0
            assert updated == 1

            # Verify transaction was updated
            result = await db_session.execute(
                select(Transaction).where(
                    Transaction.tenant_id == mock_tenant_id,
                    Transaction.source == DataSource.QUICKBOOKS,
                    Transaction.transaction_id == "123",
                )
            )
            transaction = result.scalar_one()

            assert transaction.amount == 100.50
            assert transaction.description == "Test transaction"
            assert transaction.category == "Office Supplies"
            assert transaction.merchant_name == "Vendor XYZ"


@pytest.mark.asyncio
async def test_qb_normalize_transaction(mock_qb_transaction: dict):
    """Test normalizing a QuickBooks transaction."""
    service = QuickBooksService()
    normalized = service._normalize_transaction(mock_qb_transaction)

    assert normalized["transaction_id"] == "123"
    assert normalized["source_account_id"] == "456"
    assert normalized["amount"] == 100.50
    assert normalized["currency"] == "USD"
    assert normalized["description"] == "Test transaction"
    assert normalized["category"] == "Office Supplies"
    assert normalized["merchant_name"] == "Vendor XYZ"
    assert isinstance(normalized["transaction_date"], datetime.datetime)
    assert normalized["transaction_date"].strftime("%Y-%m-%d") == "2023-01-15"
    assert normalized["raw_data"] == mock_qb_transaction


@pytest.mark.asyncio
async def test_update_tenant_quickbooks_settings(
    db_session: AsyncSession, mock_tenant: Tenant, mock_tenant_id: uuid.UUID
):
    """Test updating QuickBooks settings for a tenant."""
    new_settings = {
        "access_token": "updated_token",
        "refresh_token": "updated_refresh",
        "realm_id": "54321",
    }

    success = await update_tenant_quickbooks_settings(
        db_session, mock_tenant_id, new_settings
    )

    assert success is True

    # Verify settings were updated
    result = await db_session.execute(select(Tenant).where(Tenant.id == mock_tenant_id))
    updated_tenant = result.scalar_one()

    assert updated_tenant.quickbooks_settings["access_token"] == "updated_token"
    assert updated_tenant.quickbooks_settings["refresh_token"] == "updated_refresh"
    assert updated_tenant.quickbooks_settings["realm_id"] == "54321"


@pytest.mark.asyncio
async def test_update_tenant_quickbooks_settings_nonexistent(db_session: AsyncSession):
    """Test updating QuickBooks settings for a nonexistent tenant."""
    new_settings = {"access_token": "new_token"}
    nonexistent_id = uuid.uuid4()

    success = await update_tenant_quickbooks_settings(
        db_session, nonexistent_id, new_settings
    )

    assert success is False


@pytest.mark.asyncio
async def test_handle_quickbooks_webhook(
    db_session: AsyncSession, mock_tenant: Tenant, mock_tenant_id: uuid.UUID
):
    """Test handling a QuickBooks webhook."""
    webhook_payload = {
        "eventNotifications": [
            {
                "realmId": "12345",
                "dataChangeEvent": {
                    "entities": [
                        {
                            "name": "Purchase",
                            "id": "123",
                            "operation": "Create",
                            "lastUpdated": "2023-03-01T12:00:00Z",
                        }
                    ]
                },
            }
        ]
    }

    with patch.object(QuickBooksService, "sync_transactions") as mock_sync:
        mock_sync.return_value = (1, 1, 0)

        response = await handle_quickbooks_webhook(db_session, webhook_payload)

        assert response["status"] == "success"
        assert "Processed 1 QuickBooks events" in response["message"]
        mock_sync.assert_called_once()
        assert mock_sync.call_args[0][0] == mock_tenant_id


@pytest.mark.asyncio
async def test_handle_quickbooks_webhook_missing_realm_id(db_session: AsyncSession):
    """Test handling a webhook with missing realm ID."""
    webhook_payload = {"eventNotifications": [{}]}

    response = await handle_quickbooks_webhook(db_session, webhook_payload)

    assert response["status"] == "error"
    assert "Missing realm ID" in response["message"]


@pytest.mark.asyncio
async def test_handle_quickbooks_webhook_tenant_not_found(db_session: AsyncSession):
    """Test handling a webhook for an unknown tenant."""
    webhook_payload = {
        "eventNotifications": [
            {
                "realmId": "unknown-realm",
                "dataChangeEvent": {
                    "entities": [
                        {
                            "name": "Purchase",
                            "id": "123",
                            "operation": "Create",
                        }
                    ]
                },
            }
        ]
    }

    response = await handle_quickbooks_webhook(db_session, webhook_payload)

    assert response["status"] == "error"
    assert "No tenant found for realm ID" in response["message"]


@pytest.mark.asyncio
async def test_handle_quickbooks_webhook_no_events(
    db_session: AsyncSession, mock_tenant: Tenant
):
    """Test handling a webhook with no events."""
    webhook_payload = {
        "eventNotifications": [
            {
                "realmId": "12345",
                "dataChangeEvent": {"entities": []},
            }
        ]
    }

    response = await handle_quickbooks_webhook(db_session, webhook_payload)

    assert response["status"] == "success"
    assert "No events to process" in response["message"]
