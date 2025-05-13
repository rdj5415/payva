"""Tests for the Plaid ingestion module.

This module contains tests for the Plaid Link, data fetching,
normalization, and webhook handling functionality.
"""

import datetime
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import plaid
import pytest
from fastapi import HTTPException
from plaid.model.link_token_create_response import LinkTokenCreateResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.models import DataSource, Tenant, Transaction
from auditpulse_mvp.ingestion.plaid import (
    PlaidClient,
    PlaidService,
    handle_plaid_webhook,
    update_tenant_plaid_settings,
)


@pytest.fixture
def mock_tenant_id() -> uuid.UUID:
    """Return a mock tenant ID."""
    return uuid.UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def mock_plaid_settings() -> dict:
    """Return mock Plaid settings."""
    return {
        "access_token": "access-sandbox-123456",
        "item_id": "item-sandbox-123456",
        "institution_id": "ins_1",
        "institution_name": "Test Bank",
    }


@pytest.fixture
async def mock_tenant(
    db_session: AsyncSession, mock_tenant_id: uuid.UUID, mock_plaid_settings: dict
) -> Tenant:
    """Create a mock tenant with Plaid settings."""
    tenant = Tenant(
        id=mock_tenant_id,
        name="Test Tenant",
        slug="test-tenant-plaid",
        plaid_settings=mock_plaid_settings,
    )
    db_session.add(tenant)
    await db_session.flush()
    return tenant


@pytest.fixture
def mock_plaid_transaction() -> dict:
    """Return a mock Plaid transaction."""
    return {
        "transaction_id": "txn_123",
        "account_id": "acc_123",
        "amount": 100.50,  # Plaid uses positive for debit, negative for credit
        "date": "2023-01-15",
        "authorized_date": "2023-01-14",
        "name": "Coffee Shop",
        "merchant_name": "Starbucks",
        "category": ["Food and Drink", "Restaurants", "Coffee Shop"],
        "payment_channel": "in store",
        "pending": False,
    }


@pytest.fixture
def mock_plaid_account() -> dict:
    """Return a mock Plaid account."""
    return {
        "account_id": "acc_123",
        "name": "Checking Account",
        "mask": "1234",
        "type": "depository",
        "subtype": "checking",
    }


@pytest.mark.asyncio
async def test_plaid_client_init():
    """Test Plaid client initialization."""
    tenant_id = uuid.uuid4()

    with patch("auditpulse_mvp.ingestion.plaid.settings") as mock_settings, patch(
        "auditpulse_mvp.ingestion.plaid.plaid.ApiClient"
    ) as MockApiClient, patch(
        "auditpulse_mvp.ingestion.plaid.plaid_api.PlaidApi"
    ) as MockPlaidApi:
        mock_settings.PLAID_CLIENT_ID = "test-client-id"
        mock_settings.PLAID_SECRET.get_secret_value.return_value = "test-secret"
        mock_settings.PLAID_ENVIRONMENT = "sandbox"

        MockPlaidApi.return_value = MagicMock()

        client = PlaidClient(tenant_id=tenant_id)

        assert client.tenant_id == tenant_id
        assert client.client_id == "test-client-id"
        assert client.client_secret == "test-secret"
        assert client.access_token is None
        assert client.item_id is None
        assert MockApiClient.call_count == 1
        assert MockPlaidApi.call_count == 1


@pytest.mark.asyncio
async def test_plaid_client_get_plaid_host():
    """Test getting the Plaid API host URL."""
    tenant_id = uuid.uuid4()

    with patch("auditpulse_mvp.ingestion.plaid.settings") as mock_settings, patch(
        "auditpulse_mvp.ingestion.plaid.plaid.ApiClient"
    ), patch("auditpulse_mvp.ingestion.plaid.plaid_api.PlaidApi"):
        mock_settings.PLAID_CLIENT_ID = "test-client-id"
        mock_settings.PLAID_SECRET.get_secret_value.return_value = "test-secret"

        client = PlaidClient(tenant_id=tenant_id)

        # Test production environment
        assert client._get_plaid_host("production") == plaid.Environment.Production

        # Test development environment
        assert client._get_plaid_host("development") == plaid.Environment.Development

        # Test sandbox environment
        assert client._get_plaid_host("sandbox") == plaid.Environment.Sandbox

        # Test default (should be sandbox)
        assert client._get_plaid_host("unknown") == plaid.Environment.Sandbox


@pytest.mark.asyncio
async def test_plaid_client_create_link_token_success():
    """Test creating a Link token successfully."""
    tenant_id = uuid.uuid4()

    with patch("auditpulse_mvp.ingestion.plaid.settings") as mock_settings, patch(
        "auditpulse_mvp.ingestion.plaid.plaid.ApiClient"
    ), patch(
        "auditpulse_mvp.ingestion.plaid.plaid_api.PlaidApi"
    ) as MockPlaidApi, patch(
        "auditpulse_mvp.ingestion.plaid.LinkTokenCreateRequest"
    ) as MockLinkTokenRequest:
        mock_settings.PLAID_CLIENT_ID = "test-client-id"
        mock_settings.PLAID_SECRET.get_secret_value.return_value = "test-secret"

        mock_plaid_client = MagicMock()
        mock_response = {
            "link_token": "link-sandbox-123456",
            "expiration": "2023-01-31T12:00:00Z",
        }
        mock_plaid_client.link_token_create.return_value = mock_response
        MockPlaidApi.return_value = mock_plaid_client

        client = PlaidClient(tenant_id=tenant_id)
        response = await client.create_link_token("user-123", "John Doe")

        assert response["link_token"] == "link-sandbox-123456"
        assert response["expiration"] == "2023-01-31T12:00:00Z"
        mock_plaid_client.link_token_create.assert_called_once()
        MockLinkTokenRequest.assert_called_once()


@pytest.mark.asyncio
async def test_plaid_client_create_link_token_api_error():
    """Test creating a Link token with API error."""
    tenant_id = uuid.uuid4()

    with patch("auditpulse_mvp.ingestion.plaid.settings") as mock_settings, patch(
        "auditpulse_mvp.ingestion.plaid.plaid.ApiClient"
    ), patch(
        "auditpulse_mvp.ingestion.plaid.plaid_api.PlaidApi"
    ) as MockPlaidApi, patch(
        "auditpulse_mvp.ingestion.plaid.retry", lambda f, **kwargs: f
    ):
        mock_settings.PLAID_CLIENT_ID = "test-client-id"
        mock_settings.PLAID_SECRET.get_secret_value.return_value = "test-secret"

        mock_plaid_client = MagicMock()
        mock_plaid_client.link_token_create.side_effect = plaid.ApiException(
            status=400, reason="Bad Request"
        )
        MockPlaidApi.return_value = mock_plaid_client

        client = PlaidClient(tenant_id=tenant_id)

        with pytest.raises(HTTPException) as excinfo:
            await client.create_link_token("user-123", "John Doe")

        assert excinfo.value.status_code == 502
        assert "Error creating Plaid Link token" in str(excinfo.value.detail)


@pytest.mark.asyncio
async def test_plaid_client_exchange_public_token_success():
    """Test exchanging a public token successfully."""
    tenant_id = uuid.uuid4()

    with patch("auditpulse_mvp.ingestion.plaid.settings") as mock_settings, patch(
        "auditpulse_mvp.ingestion.plaid.plaid.ApiClient"
    ), patch(
        "auditpulse_mvp.ingestion.plaid.plaid_api.PlaidApi"
    ) as MockPlaidApi, patch(
        "auditpulse_mvp.ingestion.plaid.plaid.model.item_public_token_exchange_request.ItemPublicTokenExchangeRequest"
    ) as MockExchangeRequest, patch(
        "auditpulse_mvp.ingestion.plaid.ItemGetRequest"
    ) as MockItemGetRequest, patch(
        "auditpulse_mvp.ingestion.plaid.InstitutionsGetByIdRequest"
    ) as MockInstitutionRequest:
        mock_settings.PLAID_CLIENT_ID = "test-client-id"
        mock_settings.PLAID_SECRET.get_secret_value.return_value = "test-secret"

        mock_plaid_client = MagicMock()
        # Mock exchange response
        mock_exchange_response = {
            "access_token": "access-sandbox-123456",
            "item_id": "item-sandbox-123456",
        }
        mock_plaid_client.item_public_token_exchange.return_value = (
            mock_exchange_response
        )

        # Mock item response
        mock_item_response = {
            "item": {
                "institution_id": "ins_1",
            }
        }
        mock_plaid_client.item_get.return_value = mock_item_response

        # Mock institution response
        mock_institution_response = {
            "institution": {
                "name": "Test Bank",
            }
        }
        mock_plaid_client.institutions_get_by_id.return_value = (
            mock_institution_response
        )

        MockPlaidApi.return_value = mock_plaid_client

        client = PlaidClient(tenant_id=tenant_id)
        response = await client.exchange_public_token("public-sandbox-123456")

        assert response["access_token"] == "access-sandbox-123456"
        assert response["item_id"] == "item-sandbox-123456"
        assert response["institution_id"] == "ins_1"
        assert response["institution_name"] == "Test Bank"
        assert client.access_token == "access-sandbox-123456"
        assert client.item_id == "item-sandbox-123456"

        mock_plaid_client.item_public_token_exchange.assert_called_once()
        mock_plaid_client.item_get.assert_called_once()
        mock_plaid_client.institutions_get_by_id.assert_called_once()
        MockExchangeRequest.assert_called_once()
        MockItemGetRequest.assert_called_once()
        MockInstitutionRequest.assert_called_once()


@pytest.mark.asyncio
async def test_plaid_client_exchange_public_token_api_error():
    """Test exchanging a public token with API error."""
    tenant_id = uuid.uuid4()

    with patch("auditpulse_mvp.ingestion.plaid.settings") as mock_settings, patch(
        "auditpulse_mvp.ingestion.plaid.plaid.ApiClient"
    ), patch(
        "auditpulse_mvp.ingestion.plaid.plaid_api.PlaidApi"
    ) as MockPlaidApi, patch(
        "auditpulse_mvp.ingestion.plaid.retry", lambda f, **kwargs: f
    ):
        mock_settings.PLAID_CLIENT_ID = "test-client-id"
        mock_settings.PLAID_SECRET.get_secret_value.return_value = "test-secret"

        mock_plaid_client = MagicMock()
        mock_plaid_client.item_public_token_exchange.side_effect = plaid.ApiException(
            status=400, reason="Bad Request"
        )
        MockPlaidApi.return_value = mock_plaid_client

        client = PlaidClient(tenant_id=tenant_id)

        with pytest.raises(HTTPException) as excinfo:
            await client.exchange_public_token("public-sandbox-123456")

        assert excinfo.value.status_code == 502
        assert "Error exchanging Plaid public token" in str(excinfo.value.detail)


@pytest.mark.asyncio
async def test_plaid_client_fetch_transactions_success(
    mock_plaid_transaction, mock_plaid_account
):
    """Test fetching transactions successfully."""
    tenant_id = uuid.uuid4()

    with patch("auditpulse_mvp.ingestion.plaid.settings") as mock_settings, patch(
        "auditpulse_mvp.ingestion.plaid.plaid.ApiClient"
    ), patch(
        "auditpulse_mvp.ingestion.plaid.plaid_api.PlaidApi"
    ) as MockPlaidApi, patch(
        "auditpulse_mvp.ingestion.plaid.TransactionsGetRequest"
    ) as MockGetRequest, patch(
        "auditpulse_mvp.ingestion.plaid.TransactionsGetRequestOptions"
    ) as MockOptions:
        mock_settings.PLAID_CLIENT_ID = "test-client-id"
        mock_settings.PLAID_SECRET.get_secret_value.return_value = "test-secret"

        mock_plaid_client = MagicMock()
        # Mock transactions response
        mock_txn_response = {
            "transactions": [mock_plaid_transaction],
            "accounts": [mock_plaid_account],
            "item": {"item_id": "item-sandbox-123456"},
            "total_transactions": 1,
        }
        mock_plaid_client.transactions_get.return_value = mock_txn_response
        MockPlaidApi.return_value = mock_plaid_client

        client = PlaidClient(tenant_id=tenant_id)
        client.access_token = "access-sandbox-123456"

        response = await client.fetch_transactions(
            start_date=datetime.date(2023, 1, 1),
            end_date=datetime.date(2023, 2, 1),
        )

        assert response["transactions"] == [mock_plaid_transaction]
        assert response["accounts"] == [mock_plaid_account]
        assert response["total_transactions"] == 1

        mock_plaid_client.transactions_get.assert_called_once()
        MockGetRequest.assert_called_once()


@pytest.mark.asyncio
async def test_plaid_client_fetch_transactions_pagination(
    mock_plaid_transaction, mock_plaid_account
):
    """Test fetching transactions with pagination."""
    tenant_id = uuid.uuid4()

    with patch("auditpulse_mvp.ingestion.plaid.settings") as mock_settings, patch(
        "auditpulse_mvp.ingestion.plaid.plaid.ApiClient"
    ), patch(
        "auditpulse_mvp.ingestion.plaid.plaid_api.PlaidApi"
    ) as MockPlaidApi, patch(
        "auditpulse_mvp.ingestion.plaid.TransactionsGetRequest"
    ) as MockGetRequest, patch(
        "auditpulse_mvp.ingestion.plaid.TransactionsGetRequestOptions"
    ) as MockOptions:
        mock_settings.PLAID_CLIENT_ID = "test-client-id"
        mock_settings.PLAID_SECRET.get_secret_value.return_value = "test-secret"

        mock_plaid_client = MagicMock()
        # First page response
        first_txn = {**mock_plaid_transaction, "transaction_id": "txn_1"}
        mock_first_response = {
            "transactions": [first_txn],
            "accounts": [mock_plaid_account],
            "item": {"item_id": "item-sandbox-123456"},
            "total_transactions": 2,  # Indicate there are more transactions
        }

        # Second page response
        second_txn = {**mock_plaid_transaction, "transaction_id": "txn_2"}
        mock_second_response = {
            "transactions": [second_txn],
            "accounts": [mock_plaid_account],
            "item": {"item_id": "item-sandbox-123456"},
            "total_transactions": 2,
        }

        # Mock multiple call responses
        mock_plaid_client.transactions_get.side_effect = [
            mock_first_response,
            mock_second_response,
        ]
        MockPlaidApi.return_value = mock_plaid_client

        client = PlaidClient(tenant_id=tenant_id)
        client.access_token = "access-sandbox-123456"

        response = await client.fetch_transactions()

        assert len(response["transactions"]) == 2
        assert response["transactions"][0]["transaction_id"] == "txn_1"
        assert response["transactions"][1]["transaction_id"] == "txn_2"
        assert response["total_transactions"] == 2

        assert mock_plaid_client.transactions_get.call_count == 2
        assert MockGetRequest.call_count == 2
        # First call is default, second call has offset
        assert MockOptions.call_count == 1  # Only for pagination


@pytest.mark.asyncio
async def test_plaid_client_fetch_transactions_api_error():
    """Test fetching transactions with API error."""
    tenant_id = uuid.uuid4()

    with patch("auditpulse_mvp.ingestion.plaid.settings") as mock_settings, patch(
        "auditpulse_mvp.ingestion.plaid.plaid.ApiClient"
    ), patch(
        "auditpulse_mvp.ingestion.plaid.plaid_api.PlaidApi"
    ) as MockPlaidApi, patch(
        "auditpulse_mvp.ingestion.plaid.retry", lambda f, **kwargs: f
    ):
        mock_settings.PLAID_CLIENT_ID = "test-client-id"
        mock_settings.PLAID_SECRET.get_secret_value.return_value = "test-secret"

        mock_plaid_client = MagicMock()
        mock_plaid_client.transactions_get.side_effect = plaid.ApiException(
            status=400, reason="Bad Request"
        )
        MockPlaidApi.return_value = mock_plaid_client

        client = PlaidClient(tenant_id=tenant_id)
        client.access_token = "access-sandbox-123456"

        with pytest.raises(HTTPException) as excinfo:
            await client.fetch_transactions()

        assert excinfo.value.status_code == 502
        assert "Error fetching Plaid transactions" in str(excinfo.value.detail)


@pytest.mark.asyncio
async def test_plaid_client_refresh_transactions_success():
    """Test refreshing transactions successfully."""
    tenant_id = uuid.uuid4()

    with patch("auditpulse_mvp.ingestion.plaid.settings") as mock_settings, patch(
        "auditpulse_mvp.ingestion.plaid.plaid.ApiClient"
    ), patch(
        "auditpulse_mvp.ingestion.plaid.plaid_api.PlaidApi"
    ) as MockPlaidApi, patch(
        "auditpulse_mvp.ingestion.plaid.TransactionsRefreshRequest"
    ) as MockRefreshRequest:
        mock_settings.PLAID_CLIENT_ID = "test-client-id"
        mock_settings.PLAID_SECRET.get_secret_value.return_value = "test-secret"

        mock_plaid_client = MagicMock()
        # Refresh doesn't return anything meaningful
        mock_plaid_client.transactions_refresh.return_value = None
        MockPlaidApi.return_value = mock_plaid_client

        client = PlaidClient(tenant_id=tenant_id)
        client.access_token = "access-sandbox-123456"

        response = await client.refresh_transactions()

        assert response["status"] == "refresh_requested"
        assert "Plaid will send a webhook" in response["message"]

        mock_plaid_client.transactions_refresh.assert_called_once()
        MockRefreshRequest.assert_called_once()


@pytest.mark.asyncio
async def test_plaid_client_fire_webhook_sandbox_success():
    """Test firing a sandbox webhook successfully."""
    tenant_id = uuid.uuid4()

    with patch("auditpulse_mvp.ingestion.plaid.settings") as mock_settings, patch(
        "auditpulse_mvp.ingestion.plaid.plaid.ApiClient"
    ), patch(
        "auditpulse_mvp.ingestion.plaid.plaid_api.PlaidApi"
    ) as MockPlaidApi, patch(
        "auditpulse_mvp.ingestion.plaid.SandboxItemFireWebhookRequest"
    ) as MockWebhookRequest:
        mock_settings.PLAID_CLIENT_ID = "test-client-id"
        mock_settings.PLAID_SECRET.get_secret_value.return_value = "test-secret"
        mock_settings.PLAID_ENVIRONMENT = "sandbox"

        mock_plaid_client = MagicMock()
        mock_plaid_client.sandbox_item_fire_webhook.return_value = {
            "webhook_fired": True,
        }
        MockPlaidApi.return_value = mock_plaid_client

        client = PlaidClient(tenant_id=tenant_id)
        client.access_token = "access-sandbox-123456"

        response = await client.fire_webhook_sandbox("DEFAULT_UPDATE")

        assert response["status"] == "webhook_fired"
        assert response["webhook_fired"] is True

        mock_plaid_client.sandbox_item_fire_webhook.assert_called_once()
        MockWebhookRequest.assert_called_once()


@pytest.mark.asyncio
async def test_plaid_client_fire_webhook_non_sandbox():
    """Test firing a sandbox webhook in non-sandbox environment."""
    tenant_id = uuid.uuid4()

    with patch("auditpulse_mvp.ingestion.plaid.settings") as mock_settings, patch(
        "auditpulse_mvp.ingestion.plaid.plaid.ApiClient"
    ), patch("auditpulse_mvp.ingestion.plaid.plaid_api.PlaidApi"):
        mock_settings.PLAID_CLIENT_ID = "test-client-id"
        mock_settings.PLAID_SECRET.get_secret_value.return_value = "test-secret"
        mock_settings.PLAID_ENVIRONMENT = "development"

        client = PlaidClient(tenant_id=tenant_id)
        client.access_token = "access-sandbox-123456"

        with pytest.raises(HTTPException) as excinfo:
            await client.fire_webhook_sandbox("DEFAULT_UPDATE")

        assert excinfo.value.status_code == 400
        assert "Sandbox webhooks can only be fired in the sandbox environment" in str(
            excinfo.value.detail
        )


@pytest.mark.asyncio
async def test_plaid_service_get_client_for_tenant(
    db_session: AsyncSession, mock_tenant: Tenant, mock_tenant_id: uuid.UUID
):
    """Test getting a Plaid client for a tenant."""
    with patch("auditpulse_mvp.ingestion.plaid.settings") as mock_settings, patch(
        "auditpulse_mvp.ingestion.plaid.plaid.ApiClient"
    ), patch("auditpulse_mvp.ingestion.plaid.plaid_api.PlaidApi"):
        mock_settings.PLAID_CLIENT_ID = "test-client-id"
        mock_settings.PLAID_SECRET.get_secret_value.return_value = "test-secret"

        service = PlaidService(db_session)
        client = await service.get_client_for_tenant(mock_tenant_id)

        assert client.tenant_id == mock_tenant_id
        assert client.access_token == "access-sandbox-123456"
        assert client.item_id == "item-sandbox-123456"


@pytest.mark.asyncio
async def test_plaid_service_get_client_for_nonexistent_tenant(
    db_session: AsyncSession,
):
    """Test getting a client for a nonexistent tenant."""
    nonexistent_id = uuid.uuid4()

    with patch("auditpulse_mvp.ingestion.plaid.settings") as mock_settings, patch(
        "auditpulse_mvp.ingestion.plaid.plaid.ApiClient"
    ), patch("auditpulse_mvp.ingestion.plaid.plaid_api.PlaidApi"):
        mock_settings.PLAID_CLIENT_ID = "test-client-id"
        mock_settings.PLAID_SECRET.get_secret_value.return_value = "test-secret"

        service = PlaidService(db_session)

        with pytest.raises(HTTPException) as excinfo:
            await service.get_client_for_tenant(nonexistent_id)

        assert excinfo.value.status_code == 404
        assert f"Tenant {nonexistent_id} not found" in str(excinfo.value.detail)


@pytest.mark.asyncio
async def test_plaid_service_get_client_for_tenant_without_plaid(
    db_session: AsyncSession, mock_tenant_id: uuid.UUID
):
    """Test getting a client for a tenant without Plaid configured."""
    # Create tenant without Plaid settings
    tenant = Tenant(
        id=mock_tenant_id,
        name="Test Tenant No Plaid",
        slug="test-tenant-no-plaid",
    )
    db_session.add(tenant)
    await db_session.flush()

    with patch("auditpulse_mvp.ingestion.plaid.settings") as mock_settings, patch(
        "auditpulse_mvp.ingestion.plaid.plaid.ApiClient"
    ), patch("auditpulse_mvp.ingestion.plaid.plaid_api.PlaidApi"):
        mock_settings.PLAID_CLIENT_ID = "test-client-id"
        mock_settings.PLAID_SECRET.get_secret_value.return_value = "test-secret"

        service = PlaidService(db_session)

        with pytest.raises(HTTPException) as excinfo:
            await service.get_client_for_tenant(mock_tenant_id)

        assert excinfo.value.status_code == 400
        assert f"Plaid not configured for tenant {mock_tenant_id}" in str(
            excinfo.value.detail
        )


@pytest.mark.asyncio
async def test_plaid_service_sync_transactions(
    db_session: AsyncSession,
    mock_tenant: Tenant,
    mock_tenant_id: uuid.UUID,
    mock_plaid_transaction: dict,
    mock_plaid_account: dict,
):
    """Test syncing transactions from Plaid."""
    with patch("auditpulse_mvp.ingestion.plaid.PlaidClient") as MockPlaidClient:
        # Mock the Plaid client
        mock_client = AsyncMock()
        mock_client.fetch_transactions.return_value = {
            "transactions": [mock_plaid_transaction],
            "accounts": [mock_plaid_account],
            "total_transactions": 1,
        }
        MockPlaidClient.return_value = mock_client

        service = PlaidService(db_session)

        # Patch get_client_for_tenant to return our mocked client
        with patch.object(PlaidService, "get_client_for_tenant") as mock_get_client:
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
                    Transaction.source == DataSource.PLAID,
                )
            )
            transaction = result.scalar_one()

            assert transaction.transaction_id == "txn_123"
            assert transaction.amount == 100.50  # Sign is flipped from Plaid's model
            assert "Coffee Shop" in transaction.description
            assert transaction.category == "Coffee Shop"
            assert transaction.merchant_name == "Starbucks"


@pytest.mark.asyncio
async def test_plaid_service_sync_transactions_update_existing(
    db_session: AsyncSession,
    mock_tenant: Tenant,
    mock_tenant_id: uuid.UUID,
    mock_plaid_transaction: dict,
    mock_plaid_account: dict,
):
    """Test syncing transactions that already exist."""
    # Create an existing transaction
    existing_txn = Transaction(
        tenant_id=mock_tenant_id,
        transaction_id="txn_123",
        source=DataSource.PLAID,
        source_account_id="old-account",
        amount=50.25,
        currency="USD",
        description="Old description",
        category="Old category",
        merchant_name="Old merchant",
        transaction_date=datetime.datetime(2023, 1, 1),
    )
    db_session.add(existing_txn)
    await db_session.flush()

    with patch("auditpulse_mvp.ingestion.plaid.PlaidClient") as MockPlaidClient:
        # Mock the Plaid client
        mock_client = AsyncMock()
        mock_client.fetch_transactions.return_value = {
            "transactions": [mock_plaid_transaction],
            "accounts": [mock_plaid_account],
            "total_transactions": 1,
        }
        MockPlaidClient.return_value = mock_client

        service = PlaidService(db_session)

        # Patch get_client_for_tenant to return our mocked client
        with patch.object(PlaidService, "get_client_for_tenant") as mock_get_client:
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
                    Transaction.source == DataSource.PLAID,
                    Transaction.transaction_id == "txn_123",
                )
            )
            transaction = result.scalar_one()

            assert transaction.amount == 100.50
            assert "Coffee Shop" in transaction.description
            assert transaction.category == "Coffee Shop"
            assert transaction.merchant_name == "Starbucks"


@pytest.mark.asyncio
async def test_plaid_normalize_transaction(
    mock_plaid_transaction: dict, mock_plaid_account: dict
):
    """Test normalizing a Plaid transaction."""
    accounts_map = {mock_plaid_account["account_id"]: mock_plaid_account}

    service = PlaidService()
    normalized = service._normalize_transaction(mock_plaid_transaction, accounts_map)

    assert normalized["transaction_id"] == "txn_123"
    assert normalized["source_account_id"] == "acc_123 (Checking Account)"
    assert (
        normalized["amount"] == 100.50
    )  # Should be flipped from Plaid's sign convention
    assert normalized["currency"] == "USD"
    assert "Coffee Shop" in normalized["description"]
    assert "in store" in normalized["description"]
    assert normalized["category"] == "Coffee Shop"
    assert normalized["merchant_name"] == "Starbucks"
    assert isinstance(normalized["transaction_date"], datetime.datetime)
    assert normalized["transaction_date"].strftime("%Y-%m-%d") == "2023-01-15"
    assert isinstance(normalized["posting_date"], datetime.datetime)
    assert normalized["posting_date"].strftime("%Y-%m-%d") == "2023-01-14"
    assert normalized["raw_data"] == mock_plaid_transaction


@pytest.mark.asyncio
async def test_update_tenant_plaid_settings(
    db_session: AsyncSession, mock_tenant: Tenant, mock_tenant_id: uuid.UUID
):
    """Test updating Plaid settings for a tenant."""
    new_settings = {
        "access_token": "updated-access-token",
        "item_id": "updated-item-id",
        "institution_id": "ins_2",
        "institution_name": "Updated Bank",
    }

    success = await update_tenant_plaid_settings(
        db_session, mock_tenant_id, new_settings
    )

    assert success is True

    # Verify settings were updated
    result = await db_session.execute(select(Tenant).where(Tenant.id == mock_tenant_id))
    updated_tenant = result.scalar_one()

    assert updated_tenant.plaid_settings["access_token"] == "updated-access-token"
    assert updated_tenant.plaid_settings["item_id"] == "updated-item-id"
    assert updated_tenant.plaid_settings["institution_name"] == "Updated Bank"


@pytest.mark.asyncio
async def test_update_tenant_plaid_settings_nonexistent(db_session: AsyncSession):
    """Test updating Plaid settings for a nonexistent tenant."""
    new_settings = {"access_token": "new-token"}
    nonexistent_id = uuid.uuid4()

    success = await update_tenant_plaid_settings(
        db_session, nonexistent_id, new_settings
    )

    assert success is False


@pytest.mark.asyncio
async def test_handle_plaid_webhook_initial_update(
    db_session: AsyncSession, mock_tenant: Tenant, mock_tenant_id: uuid.UUID
):
    """Test handling a Plaid INITIAL_UPDATE webhook."""
    webhook_data = {
        "webhook_type": "TRANSACTIONS",
        "webhook_code": "INITIAL_UPDATE",
        "item_id": "item-sandbox-123456",
        "new_transactions": 10,
    }

    response = await handle_plaid_webhook(db_session, webhook_data)

    assert response["status"] == "acknowledged"
    assert "Initial update received" in response["message"]


@pytest.mark.asyncio
async def test_handle_plaid_webhook_default_update(
    db_session: AsyncSession, mock_tenant: Tenant, mock_tenant_id: uuid.UUID
):
    """Test handling a Plaid DEFAULT_UPDATE webhook."""
    webhook_data = {
        "webhook_type": "TRANSACTIONS",
        "webhook_code": "DEFAULT_UPDATE",
        "item_id": "item-sandbox-123456",
        "new_transactions": 3,
    }

    with patch.object(PlaidService, "sync_transactions") as mock_sync:
        mock_sync.return_value = (3, 3, 0)

        response = await handle_plaid_webhook(db_session, webhook_data)

        assert response["status"] == "success"
        assert "Transactions synced" in response["message"]
        mock_sync.assert_called_once()
        assert mock_sync.call_args[0][0] == mock_tenant_id
        # Verify we're limiting to recent transactions
        assert (
            mock_sync.call_args[1]["end_date"] - mock_sync.call_args[1]["start_date"]
        ).days <= 30


@pytest.mark.asyncio
async def test_handle_plaid_webhook_missing_item_id(db_session: AsyncSession):
    """Test handling a webhook with missing item ID."""
    webhook_data = {
        "webhook_type": "TRANSACTIONS",
        "webhook_code": "DEFAULT_UPDATE",
    }

    response = await handle_plaid_webhook(db_session, webhook_data)

    assert response["status"] == "error"
    assert "Missing item ID" in response["message"]


@pytest.mark.asyncio
async def test_handle_plaid_webhook_tenant_not_found(db_session: AsyncSession):
    """Test handling a webhook for an unknown tenant."""
    webhook_data = {
        "webhook_type": "TRANSACTIONS",
        "webhook_code": "DEFAULT_UPDATE",
        "item_id": "unknown-item-id",
        "new_transactions": 3,
    }

    response = await handle_plaid_webhook(db_session, webhook_data)

    assert response["status"] == "error"
    assert "No tenant found for item ID" in response["message"]
