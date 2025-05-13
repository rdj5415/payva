"""Tests for Plaid client."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from auditpulse_mvp.integrations.plaid.client import PlaidClient
from auditpulse_mvp.utils.settings import Settings

@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock(spec=Settings)
    settings.PLAID_CLIENT_ID = "test-client-id"
    settings.PLAID_SECRET = "test-secret"
    settings.PLAID_ENVIRONMENT = "sandbox"
    settings.APP_NAME = "AuditPulse Test"
    return settings

@pytest.fixture
def plaid_client(mock_settings):
    """Create Plaid client with mocked API."""
    with patch("plaid.ApiClient") as mock_api_client, \
         patch("plaid.api.plaid_api.PlaidApi") as mock_plaid_api:
        # Mock API client
        api_client = MagicMock()
        mock_api_client.return_value = api_client
        
        # Mock Plaid API
        plaid_api = MagicMock()
        mock_plaid_api.return_value = plaid_api
        
        # Create client
        client = PlaidClient(settings=mock_settings)
        client.client = plaid_api
        
        yield client

@pytest.mark.asyncio
async def test_create_link_token(plaid_client):
    """Test creating a link token."""
    # Mock response
    response = MagicMock()
    response.to_dict.return_value = {
        "link_token": "test-link-token",
        "expiration": "2023-01-01T00:00:00Z",
        "request_id": "test-request-id",
    }
    plaid_client.client.link_token_create.return_value = response
    
    # Create link token
    result = await plaid_client.create_link_token(user_id="test-user")
    
    # Verify result
    assert result["link_token"] == "test-link-token"
    assert result["expiration"] == "2023-01-01T00:00:00Z"
    assert result["request_id"] == "test-request-id"
    
    # Verify API call
    plaid_client.client.link_token_create.assert_called_once()

@pytest.mark.asyncio
async def test_exchange_public_token(plaid_client):
    """Test exchanging a public token."""
    # Mock response
    response = MagicMock()
    response.to_dict.return_value = {
        "access_token": "test-access-token",
        "item_id": "test-item-id",
    }
    plaid_client.client.item_public_token_exchange.return_value = response
    
    # Exchange public token
    result = await plaid_client.exchange_public_token(public_token="test-public-token")
    
    # Verify result
    assert result["access_token"] == "test-access-token"
    assert result["item_id"] == "test-item-id"
    
    # Verify API call
    plaid_client.client.item_public_token_exchange.assert_called_once_with(
        {"public_token": "test-public-token"}
    )

@pytest.mark.asyncio
async def test_get_accounts(plaid_client):
    """Test getting accounts."""
    # Mock response
    response = MagicMock()
    response.to_dict.return_value = {
        "accounts": [
            {
                "account_id": "test-account-id",
                "name": "Test Account",
                "type": "depository",
                "balances": {"current": 1000},
            }
        ],
        "item": {"item_id": "test-item-id"},
    }
    plaid_client.client.accounts_get.return_value = response
    
    # Get accounts
    result = await plaid_client.get_accounts(access_token="test-access-token")
    
    # Verify result
    assert len(result["accounts"]) == 1
    assert result["accounts"][0]["account_id"] == "test-account-id"
    assert result["item"]["item_id"] == "test-item-id"
    
    # Verify API call
    plaid_client.client.accounts_get.assert_called_once()

@pytest.mark.asyncio
async def test_get_transactions(plaid_client):
    """Test getting transactions."""
    # Mock response
    response = MagicMock()
    response.to_dict.return_value = {
        "accounts": [
            {
                "account_id": "test-account-id",
                "name": "Test Account",
                "type": "depository",
                "balances": {"current": 1000},
            }
        ],
        "transactions": [
            {
                "transaction_id": "test-transaction-id",
                "account_id": "test-account-id",
                "amount": 100,
                "date": "2023-01-01",
                "name": "Test Transaction",
                "pending": False,
            }
        ],
        "total_transactions": 1,
    }
    plaid_client.client.transactions_get.return_value = response
    
    # Get transactions
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    result = await plaid_client.get_transactions(
        access_token="test-access-token",
        start_date=start_date,
        end_date=end_date,
    )
    
    # Verify result
    assert len(result["transactions"]) == 1
    assert result["transactions"][0]["transaction_id"] == "test-transaction-id"
    assert result["total_transactions"] == 1
    
    # Verify API call
    plaid_client.client.transactions_get.assert_called_once()

@pytest.mark.asyncio
async def test_get_all_transactions(plaid_client):
    """Test getting all transactions with pagination."""
    # Mock first response (has more transactions)
    first_response = MagicMock()
    first_response.to_dict.return_value = {
        "accounts": [
            {
                "account_id": "test-account-id",
                "name": "Test Account",
                "type": "depository",
                "balances": {"current": 1000},
            }
        ],
        "transactions": [
            {
                "transaction_id": "test-transaction-id-1",
                "account_id": "test-account-id",
                "amount": 100,
                "date": "2023-01-01",
                "name": "Test Transaction 1",
                "pending": False,
            }
        ],
        "total_transactions": 2,
    }
    
    # Mock second response (no more transactions)
    second_response = MagicMock()
    second_response.to_dict.return_value = {
        "accounts": [
            {
                "account_id": "test-account-id",
                "name": "Test Account",
                "type": "depository",
                "balances": {"current": 1000},
            }
        ],
        "transactions": [
            {
                "transaction_id": "test-transaction-id-2",
                "account_id": "test-account-id",
                "amount": 200,
                "date": "2023-01-02",
                "name": "Test Transaction 2",
                "pending": False,
            }
        ],
        "total_transactions": 2,
    }
    
    # Set up mock to return different responses
    plaid_client.client.transactions_get.side_effect = [
        first_response,
        second_response,
    ]
    
    # Get all transactions
    start_date = datetime.now() - timedelta(days=30)
    result = await plaid_client.get_all_transactions(
        access_token="test-access-token",
        start_date=start_date,
    )
    
    # Verify result
    assert len(result) == 2
    assert result[0]["transaction_id"] == "test-transaction-id-1"
    assert result[1]["transaction_id"] == "test-transaction-id-2"
    
    # Verify API calls
    assert plaid_client.client.transactions_get.call_count == 2 