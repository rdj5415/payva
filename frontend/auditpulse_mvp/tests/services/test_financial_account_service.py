"""Tests for financial account service."""

import pytest
from datetime import datetime, timedelta
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from auditpulse_mvp.database.models import (
    User,
    FinancialInstitution,
    FinancialAccount,
    FinancialTransaction,
)
from auditpulse_mvp.services.financial_account_service import FinancialAccountService


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def mock_user():
    """Create a mock user."""
    return User(
        id=uuid.uuid4(),
        email="test@example.com",
        full_name="Test User",
    )


@pytest.fixture
def mock_plaid_client():
    """Create a mock Plaid client."""
    client = AsyncMock()
    return client


@pytest.fixture
def mock_task_manager():
    """Create a mock task manager."""
    manager = AsyncMock()
    return manager


@pytest.fixture
def financial_account_service(
    mock_db_session, mock_user, mock_plaid_client, mock_task_manager
):
    """Create a financial account service."""
    return FinancialAccountService(
        db_session=mock_db_session,
        user=mock_user,
        plaid_client=mock_plaid_client,
        task_manager=mock_task_manager,
    )


@pytest.fixture
def mock_institution():
    """Create a mock financial institution."""
    return FinancialInstitution(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        name="Test Bank",
        plaid_access_token="test-access-token",
        plaid_item_id="test-item-id",
        plaid_institution_id="test-institution-id",
        is_active=True,
        created_at=datetime.now(),
        last_updated=datetime.now(),
    )


@pytest.mark.asyncio
async def test_store_plaid_access_token_new(financial_account_service, mock_db_session):
    """Test storing a new Plaid access token."""
    # Mock database query result (no existing institution)
    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db_session.execute.return_value = mock_result

    # Mock accounts response
    financial_account_service.plaid_client.get_accounts.return_value = {
        "accounts": [
            {
                "account_id": "test-account-id",
                "name": "Test Account",
                "type": "depository",
                "balances": {"current": 1000},
            }
        ]
    }

    # Store access token
    await financial_account_service.store_plaid_access_token(
        access_token="test-access-token",
        item_id="test-item-id",
        institution_name="Test Bank",
        institution_id="test-institution-id",
    )

    # Verify institution was created
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called()

    # Verify accounts were fetched
    financial_account_service.plaid_client.get_accounts.assert_called_once_with(
        access_token="test-access-token"
    )


@pytest.mark.asyncio
async def test_store_plaid_access_token_existing(
    financial_account_service, mock_db_session, mock_institution
):
    """Test updating an existing Plaid access token."""
    # Mock database query result (existing institution)
    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = mock_institution
    mock_db_session.execute.return_value = mock_result

    # Mock accounts response
    financial_account_service.plaid_client.get_accounts.return_value = {
        "accounts": [
            {
                "account_id": "test-account-id",
                "name": "Test Account",
                "type": "depository",
                "balances": {"current": 1000},
            }
        ]
    }

    # Store access token
    await financial_account_service.store_plaid_access_token(
        access_token="new-access-token",
        item_id="test-item-id",
        institution_name="Updated Bank Name",
        institution_id="test-institution-id",
    )

    # Verify institution was updated
    assert mock_institution.plaid_access_token == "new-access-token"
    assert mock_institution.name == "Updated Bank Name"
    mock_db_session.commit.assert_called()

    # Verify accounts were fetched
    financial_account_service.plaid_client.get_accounts.assert_called_once_with(
        access_token="new-access-token"
    )


@pytest.mark.asyncio
async def test_get_accounts(
    financial_account_service, mock_db_session, mock_institution
):
    """Test getting accounts."""
    # Mock database query results
    institution_result = AsyncMock()
    institution_result.scalar_one_or_none.return_value = mock_institution

    accounts_result = AsyncMock()
    accounts_result.scalars.return_value.all.return_value = [
        FinancialAccount(
            id=uuid.uuid4(),
            user_id=financial_account_service.user.id,
            institution_id=mock_institution.id,
            plaid_account_id="test-account-id",
            name="Test Account",
            type="depository",
            balances={"current": 1000},
            is_active=True,
            created_at=datetime.now(),
            last_updated=datetime.now(),
        )
    ]

    # Set up mock session to return different results
    mock_db_session.execute.side_effect = [institution_result, accounts_result]

    # Get accounts
    result = await financial_account_service.get_accounts(item_id="test-item-id")

    # Verify result
    assert len(result["accounts"]) == 1
    assert result["accounts"][0]["name"] == "Test Account"
    assert result["item_id"] == "test-item-id"
    assert result["institution_id"] == "test-institution-id"

    # Verify database queries
    assert mock_db_session.execute.call_count == 2


@pytest.mark.asyncio
async def test_get_transactions(
    financial_account_service, mock_db_session, mock_institution
):
    """Test getting transactions."""
    # Mock database query results
    institution_result = AsyncMock()
    institution_result.scalar_one_or_none.return_value = mock_institution

    account_ids_result = AsyncMock()
    account_ids_result.all.return_value = [("test-account-id",)]

    count_result = AsyncMock()
    count_result.scalar_one.return_value = 1

    transactions_result = AsyncMock()
    transactions_result.scalars.return_value.all.return_value = [
        FinancialTransaction(
            id=uuid.uuid4(),
            user_id=financial_account_service.user.id,
            account_id="test-account-id",
            transaction_id="test-transaction-id",
            amount=100,
            date=datetime.now(),
            name="Test Transaction",
            pending=False,
            category=["Food and Drink"],
            created_at=datetime.now(),
            last_updated=datetime.now(),
        )
    ]

    accounts_result = AsyncMock()
    accounts_result.scalars.return_value.all.return_value = [
        FinancialAccount(
            id=uuid.uuid4(),
            user_id=financial_account_service.user.id,
            institution_id=mock_institution.id,
            plaid_account_id="test-account-id",
            name="Test Account",
            type="depository",
            balances={"current": 1000},
            is_active=True,
            created_at=datetime.now(),
            last_updated=datetime.now(),
        )
    ]

    # Mock get_accounts method
    with patch.object(
        financial_account_service,
        "get_accounts",
        return_value={
            "accounts": [
                {
                    "id": "test-account-id",
                    "name": "Test Account",
                    "type": "depository",
                    "balances": {"current": 1000},
                    "institution_name": "Test Bank",
                }
            ],
            "item_id": "test-item-id",
            "institution_id": "test-institution-id",
        },
    ):
        # Set up mock session to return different results
        mock_db_session.execute.side_effect = [
            institution_result,
            account_ids_result,
            count_result,
            transactions_result,
        ]

        # Get transactions
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        result = await financial_account_service.get_transactions(
            start_date=start_date,
            end_date=end_date,
            item_id="test-item-id",
        )

        # Verify result
        assert len(result["transactions"]) == 1
        assert result["transactions"][0]["name"] == "Test Transaction"
        assert result["transactions"][0]["amount"] == 100
        assert len(result["accounts"]) == 1
        assert result["total_transactions"] == 1

        # Verify database queries
        assert mock_db_session.execute.call_count == 4


@pytest.mark.asyncio
async def test_queue_transaction_sync(
    financial_account_service, mock_db_session, mock_institution
):
    """Test queuing a transaction sync task."""
    # Mock database query result
    institutions_result = AsyncMock()
    institutions_result.scalars.return_value.all.return_value = [mock_institution]
    mock_db_session.execute.return_value = institutions_result

    # Mock task manager
    financial_account_service.task_manager.schedule_task.return_value = "test-task-id"

    # Queue transaction sync
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    result = await financial_account_service.queue_transaction_sync(
        start_date=start_date,
        end_date=end_date,
    )

    # Verify task was scheduled
    financial_account_service.task_manager.schedule_task.assert_called_once()
    assert result == "test-task-id"

    # Verify task parameters
    call_args = financial_account_service.task_manager.schedule_task.call_args[1]
    assert call_args["task_name"] == "sync_plaid_transactions"
    assert call_args["kwargs"]["user_id"] == str(financial_account_service.user.id)
    assert call_args["kwargs"]["institution_id"] == str(mock_institution.id)
    assert start_date.isoformat() in call_args["kwargs"]["start_date"]
    assert end_date.isoformat() in call_args["kwargs"]["end_date"]
