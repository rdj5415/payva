"""Plaid ingestion module.

This module handles Plaid Link integration, data fetching, and normalization.
"""
import datetime
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple, cast

import httpx
import plaid
from fastapi import Depends, HTTPException, status
from plaid.api import plaid_api
from plaid.model.country_code import CountryCode
from plaid.model.institutions_get_by_id_request import InstitutionsGetByIdRequest
from plaid.model.item_get_request import ItemGetRequest
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.model.products import Products
from plaid.model.sandbox_item_fire_webhook_request import SandboxItemFireWebhookRequest
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions
from plaid.model.transactions_refresh_request import TransactionsRefreshRequest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.database.models import DataSource, Tenant, Transaction
from auditpulse_mvp.utils.settings import settings


# Configure logging
logger = logging.getLogger(__name__)


class PlaidClient:
    """Client for interacting with the Plaid API."""

    def __init__(
        self,
        tenant_id: uuid.UUID,
        client_id: str = settings.PLAID_CLIENT_ID or "",
        client_secret: str = settings.PLAID_SECRET.get_secret_value() if settings.PLAID_SECRET else "",
        environment: str = settings.PLAID_ENVIRONMENT,
    ):
        """Initialize the Plaid client.

        Args:
            tenant_id: The tenant ID.
            client_id: The Plaid client ID.
            client_secret: The Plaid client secret.
            environment: The Plaid environment (sandbox, development, or production).
        """
        if not client_id or not client_secret:
            raise ValueError("Plaid credentials are not configured")

        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        
        # Configure the Plaid API client
        configuration = plaid.Configuration(
            host=self._get_plaid_host(environment),
            api_key={
                "clientId": client_id,
                "secret": client_secret,
                "plaidVersion": "2020-09-14",
            },
        )
        api_client = plaid.ApiClient(configuration)
        self.client = plaid_api.PlaidApi(api_client)
        self.access_token: Optional[str] = None
        self.item_id: Optional[str] = None

    def _get_plaid_host(self, environment: str) -> str:
        """Get the Plaid API host URL based on the environment.
        
        Args:
            environment: The Plaid environment.
            
        Returns:
            str: The Plaid API host URL.
        """
        if environment == "production":
            return plaid.Environment.Production
        elif environment == "development":
            return plaid.Environment.Development
        else:
            return plaid.Environment.Sandbox

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((plaid.ApiException, httpx.RequestError)),
        reraise=True,
    )
    async def create_link_token(self, user_id: str, user_name: str) -> Dict[str, Any]:
        """Create a Plaid Link token for client-side authentication.
        
        Args:
            user_id: The user ID.
            user_name: The user's name.
            
        Returns:
            Dict[str, Any]: The Link token response.
            
        Raises:
            HTTPException: If there's an error creating the Link token.
        """
        try:
            # Create a Link token for the user
            request = LinkTokenCreateRequest(
                user=LinkTokenCreateRequestUser(
                    client_user_id=user_id,
                    legal_name=user_name,
                ),
                client_name="AuditPulse AI",
                products=[Products("transactions")],
                country_codes=[CountryCode("US")],
                language="en",
                webhook="https://auditpulse.ai/api/webhooks/plaid",
            )
            response = self.client.link_token_create(request)
            return {
                "link_token": response["link_token"],
                "expiration": response["expiration"],
            }
        except plaid.ApiException as exc:
            logger.error(f"Error creating Plaid Link token: {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Error creating Plaid Link token: {exc}",
            )
        except Exception as exc:
            logger.error(f"Unexpected error creating Plaid Link token: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error creating Plaid Link token: {exc}",
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((plaid.ApiException, httpx.RequestError)),
        reraise=True,
    )
    async def exchange_public_token(self, public_token: str) -> Dict[str, Any]:
        """Exchange a public token for an access token.
        
        Args:
            public_token: The public token from Plaid Link.
            
        Returns:
            Dict[str, Any]: The access token response.
            
        Raises:
            HTTPException: If there's an error exchanging the token.
        """
        try:
            exchange_request = plaid.model.item_public_token_exchange_request.ItemPublicTokenExchangeRequest(
                public_token=public_token
            )
            exchange_response = self.client.item_public_token_exchange(exchange_request)
            
            self.access_token = exchange_response["access_token"]
            self.item_id = exchange_response["item_id"]
            
            # Get item details to retrieve institution information
            item_request = ItemGetRequest(access_token=self.access_token)
            item_response = self.client.item_get(item_request)
            
            # Get institution details
            institution_id = item_response["item"]["institution_id"]
            institution_request = InstitutionsGetByIdRequest(
                institution_id=institution_id,
                country_codes=[CountryCode("US")],
            )
            institution_response = self.client.institutions_get_by_id(institution_request)
            
            return {
                "access_token": self.access_token,
                "item_id": self.item_id,
                "institution_id": institution_id,
                "institution_name": institution_response["institution"]["name"],
            }
        except plaid.ApiException as exc:
            logger.error(f"Error exchanging Plaid public token: {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Error exchanging Plaid public token: {exc}",
            )
        except Exception as exc:
            logger.error(f"Unexpected error exchanging Plaid public token: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error exchanging Plaid public token: {exc}",
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((plaid.ApiException, httpx.RequestError)),
        reraise=True,
    )
    async def fetch_transactions(
        self, 
        start_date: Optional[datetime.date] = None, 
        end_date: Optional[datetime.date] = None,
        account_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Fetch transactions from Plaid.
        
        Args:
            start_date: The start date for fetching transactions.
            end_date: The end date for fetching transactions.
            account_ids: Optional list of account IDs to filter by.
            
        Returns:
            Dict[str, Any]: The transactions response.
            
        Raises:
            HTTPException: If there's an error fetching transactions.
        """
        if not self.access_token:
            raise ValueError("Access token is missing")

        # Default to last 30 days if no date range is provided
        if not start_date:
            start_date = datetime.date.today() - datetime.timedelta(days=30)
        if not end_date:
            end_date = datetime.date.today()

        try:
            # Build options
            options = None
            if account_ids:
                options = TransactionsGetRequestOptions(account_ids=account_ids)
                
            # Create request
            request = TransactionsGetRequest(
                access_token=self.access_token,
                start_date=start_date,
                end_date=end_date,
                options=options,
            )
            
            # First call - get transactions 
            response = self.client.transactions_get(request)
            
            transactions = response["transactions"]
            total_transactions = response["total_transactions"]
            
            # Pagination handling for large transaction sets
            while len(transactions) < total_transactions:
                # Use the last transaction ID for pagination
                options_with_offset = TransactionsGetRequestOptions(
                    offset=len(transactions),
                    account_ids=account_ids if account_ids else None,
                )
                
                paginated_request = TransactionsGetRequest(
                    access_token=self.access_token,
                    start_date=start_date,
                    end_date=end_date,
                    options=options_with_offset,
                )
                
                paginated_response = self.client.transactions_get(paginated_request)
                transactions.extend(paginated_response["transactions"])
                
                # Safety check to prevent infinite loops
                if len(paginated_response["transactions"]) == 0:
                    break
            
            # Build the response
            return {
                "transactions": transactions,
                "accounts": response["accounts"],
                "item": response["item"],
                "total_transactions": total_transactions,
            }
        except plaid.ApiException as exc:
            logger.error(f"Error fetching Plaid transactions: {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Error fetching Plaid transactions: {exc}",
            )
        except Exception as exc:
            logger.error(f"Unexpected error fetching Plaid transactions: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error fetching Plaid transactions: {exc}",
            )

    async def refresh_transactions(self) -> Dict[str, str]:
        """Request Plaid to refresh transactions for the item.
        
        This triggers Plaid to make a webhook notification when new transactions are available.
        
        Returns:
            Dict[str, str]: The refresh status.
            
        Raises:
            HTTPException: If there's an error refreshing transactions.
        """
        if not self.access_token:
            raise ValueError("Access token is missing")
            
        try:
            request = TransactionsRefreshRequest(access_token=self.access_token)
            self.client.transactions_refresh(request)
            
            return {"status": "refresh_requested", "message": "Plaid will send a webhook when complete"}
        except plaid.ApiException as exc:
            logger.error(f"Error refreshing Plaid transactions: {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Error refreshing Plaid transactions: {exc}",
            )
        except Exception as exc:
            logger.error(f"Unexpected error refreshing Plaid transactions: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error refreshing Plaid transactions: {exc}",
            )

    async def fire_webhook_sandbox(self, webhook_code: str) -> Dict[str, str]:
        """Fire a sandbox webhook for testing.
        
        This is only available in the sandbox environment.
        
        Args:
            webhook_code: The webhook code to fire.
            
        Returns:
            Dict[str, str]: The webhook status.
            
        Raises:
            HTTPException: If there's an error firing the webhook.
        """
        if not self.access_token:
            raise ValueError("Access token is missing")
            
        if settings.PLAID_ENVIRONMENT != "sandbox":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sandbox webhooks can only be fired in the sandbox environment",
            )
            
        try:
            request = SandboxItemFireWebhookRequest(
                access_token=self.access_token,
                webhook_code=webhook_code,
            )
            
            response = self.client.sandbox_item_fire_webhook(request)
            
            return {
                "status": "webhook_fired",
                "webhook_fired": response["webhook_fired"],
            }
        except plaid.ApiException as exc:
            logger.error(f"Error firing sandbox webhook: {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Error firing sandbox webhook: {exc}",
            )
        except Exception as exc:
            logger.error(f"Unexpected error firing sandbox webhook: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error firing sandbox webhook: {exc}",
            )


class PlaidService:
    """Service for ingesting and normalizing Plaid data."""

    def __init__(self, db_session: AsyncSession = Depends(get_db_session)):
        """Initialize the Plaid service.
        
        Args:
            db_session: The database session.
        """
        self.db_session = db_session

    async def get_client_for_tenant(self, tenant_id: uuid.UUID) -> PlaidClient:
        """Get a Plaid client for the specified tenant.
        
        Args:
            tenant_id: The tenant ID.
            
        Returns:
            PlaidClient: The Plaid client.
            
        Raises:
            HTTPException: If the tenant or Plaid settings are not found.
        """
        # Get the tenant and its Plaid settings
        tenant_result = await self.db_session.execute(
            select(Tenant).where(Tenant.id == tenant_id)
        )
        tenant = tenant_result.scalar_one_or_none()

        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tenant {tenant_id} not found",
            )

        if not tenant.plaid_settings:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Plaid not configured for tenant {tenant_id}",
            )

        plaid_settings = cast(Dict[str, Any], tenant.plaid_settings)
        
        # Create and return a client with the tenant's settings
        client = PlaidClient(
            tenant_id=tenant_id,
            client_id=settings.PLAID_CLIENT_ID or "",
            client_secret=settings.PLAID_SECRET.get_secret_value() if settings.PLAID_SECRET else "",
            environment=settings.PLAID_ENVIRONMENT,
        )
        
        # Set the tenant-specific access_token
        client.access_token = plaid_settings.get("access_token")
        client.item_id = plaid_settings.get("item_id")
        
        return client

    async def sync_transactions(
        self, 
        tenant_id: uuid.UUID, 
        start_date: Optional[datetime.date] = None, 
        end_date: Optional[datetime.date] = None,
        account_ids: Optional[List[str]] = None,
    ) -> Tuple[int, int, int]:
        """Sync transactions from Plaid.
        
        Args:
            tenant_id: The tenant ID.
            start_date: The start date for fetching transactions.
            end_date: The end date for fetching transactions.
            account_ids: Optional list of account IDs to filter by.
            
        Returns:
            Tuple[int, int, int]: A tuple of (fetched, created, updated) counts.
            
        Raises:
            HTTPException: If there's an error syncing transactions.
        """
        client = await self.get_client_for_tenant(tenant_id)
        
        try:
            # Fetch transactions from Plaid
            plaid_data = await client.fetch_transactions(start_date, end_date, account_ids)
            raw_transactions = plaid_data.get("transactions", [])
            
            # Get account mappings for source_account_id
            accounts_map = {account["account_id"]: account for account in plaid_data.get("accounts", [])}
            
            # Process and normalize transactions
            created_count, updated_count = await self._process_transactions(
                tenant_id, raw_transactions, accounts_map
            )
            
            return len(raw_transactions), created_count, updated_count
        except Exception as exc:
            logger.error(f"Error syncing Plaid transactions: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error syncing Plaid transactions: {exc}",
            )

    async def _process_transactions(
        self, 
        tenant_id: uuid.UUID, 
        raw_transactions: List[Dict[str, Any]],
        accounts_map: Dict[str, Dict[str, Any]],
    ) -> Tuple[int, int]:
        """Process and normalize Plaid transactions.
        
        Args:
            tenant_id: The tenant ID.
            raw_transactions: The list of raw transactions from Plaid.
            accounts_map: A mapping of account IDs to account details.
            
        Returns:
            Tuple[int, int]: A tuple of (created, updated) counts.
        """
        created_count = 0
        updated_count = 0
        
        for raw_txn in raw_transactions:
            # Convert Plaid format to our normalized format
            normalized_txn = self._normalize_transaction(raw_txn, accounts_map)
            
            # Check if the transaction already exists
            existing_txn_result = await self.db_session.execute(
                select(Transaction).where(
                    Transaction.tenant_id == tenant_id,
                    Transaction.transaction_id == normalized_txn["transaction_id"],
                    Transaction.source == DataSource.PLAID,
                )
            )
            existing_txn = existing_txn_result.scalar_one_or_none()
            
            if existing_txn:
                # Update existing transaction
                for key, value in normalized_txn.items():
                    if key != "transaction_id" and key != "source" and key != "tenant_id":
                        setattr(existing_txn, key, value)
                updated_count += 1
            else:
                # Create new transaction
                new_txn = Transaction(
                    tenant_id=tenant_id,
                    source=DataSource.PLAID,
                    **normalized_txn,
                )
                self.db_session.add(new_txn)
                created_count += 1
        
        # Commit all changes at once
        await self.db_session.commit()
        
        return created_count, updated_count

    def _normalize_transaction(
        self, 
        raw_txn: Dict[str, Any], 
        accounts_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Normalize a Plaid transaction to our data model.
        
        Args:
            raw_txn: The raw transaction from Plaid.
            accounts_map: A mapping of account IDs to account details.
            
        Returns:
            Dict[str, Any]: The normalized transaction data.
        """
        # Extract basic transaction information
        txn_id = raw_txn.get("transaction_id", "")
        account_id = raw_txn.get("account_id", "")
        
        # Extract account information
        source_account_id = account_id
        
        # Get additional account details if available
        account_detail = accounts_map.get(account_id, {})
        account_name = account_detail.get("name", "")
        if account_name:
            source_account_id = f"{account_id} ({account_name})"
        
        # Extract transaction amount (Plaid returns this as a positive value for credits)
        amount = raw_txn.get("amount", 0.0)
        # Invert the sign for consistency with our model (expenses are positive)
        amount = -amount
        
        # Extract transaction date
        transaction_date = datetime.datetime.now()
        txn_date = raw_txn.get("date")
        if txn_date:
            try:
                transaction_date = datetime.datetime.strptime(txn_date, "%Y-%m-%d")
            except ValueError:
                pass
        
        # Extract posting date if available
        posting_date = None
        authorized_date = raw_txn.get("authorized_date")
        if authorized_date:
            try:
                posting_date = datetime.datetime.strptime(authorized_date, "%Y-%m-%d")
            except ValueError:
                pass
        
        # Extract description and merchant name
        description = raw_txn.get("name", "")
        merchant_name = raw_txn.get("merchant_name", "")
        if not merchant_name:
            # Use the description as fallback for merchant name
            merchant_name = description
        
        # Extract category
        category = ""
        categories = raw_txn.get("category", [])
        if categories:
            # Use the most specific category (last in the list)
            category = categories[-1]
        
        # Extract payment channel
        payment_channel = raw_txn.get("payment_channel", "")
        if payment_channel and payment_channel != "other":
            if not description.endswith(f"({payment_channel})"):
                description = f"{description} ({payment_channel})"
        
        # Build the normalized transaction
        return {
            "transaction_id": txn_id,
            "source_account_id": source_account_id,
            "amount": amount,
            "currency": "USD",  # Plaid US typically uses USD
            "description": description,
            "category": category,
            "merchant_name": merchant_name,
            "transaction_date": transaction_date,
            "posting_date": posting_date,
            "raw_data": raw_txn,  # Store the entire raw transaction
        }


async def update_tenant_plaid_settings(
    db_session: AsyncSession, tenant_id: uuid.UUID, settings_data: Dict[str, Any]
) -> bool:
    """Update the Plaid settings for a tenant.
    
    Args:
        db_session: The database session.
        tenant_id: The tenant ID.
        settings_data: The Plaid settings data.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Get the tenant
        tenant_result = await db_session.execute(
            select(Tenant).where(Tenant.id == tenant_id)
        )
        tenant = tenant_result.scalar_one_or_none()
        
        if not tenant:
            logger.error(f"Tenant {tenant_id} not found")
            return False
        
        # Update the settings
        if tenant.plaid_settings:
            # Update existing settings
            tenant.plaid_settings.update(settings_data)
        else:
            # Create new settings
            tenant.plaid_settings = settings_data
        
        # Commit the changes
        await db_session.commit()
        return True
    except Exception as exc:
        logger.error(f"Error updating Plaid settings: {exc}")
        await db_session.rollback()
        return False


# Webhook handler for Plaid events
async def handle_plaid_webhook(
    db_session: AsyncSession, webhook_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Handle a Plaid webhook event.
    
    Args:
        db_session: The database session.
        webhook_data: The webhook payload.
        
    Returns:
        Dict[str, Any]: The response.
    """
    try:
        # Extract webhook information
        webhook_type = webhook_data.get("webhook_type")
        webhook_code = webhook_data.get("webhook_code")
        item_id = webhook_data.get("item_id")
        
        if not item_id:
            logger.error("Missing item ID in Plaid webhook")
            return {"status": "error", "message": "Missing item ID"}
        
        # Find the tenant with this item ID
        tenant_result = await db_session.execute(
            select(Tenant).where(
                Tenant.plaid_settings["item_id"].astext == item_id
            )
        )
        tenant = tenant_result.scalar_one_or_none()
        
        if not tenant:
            logger.error(f"No tenant found for Plaid item ID: {item_id}")
            return {"status": "error", "message": f"No tenant found for item ID: {item_id}"}
        
        # Process the webhook based on its type and code
        if webhook_type == "TRANSACTIONS":
            if webhook_code == "INITIAL_UPDATE":
                logger.info(f"Initial update for Plaid item {item_id}")
                # First transaction sync completed
                return {"status": "acknowledged", "message": "Initial update received"}
                
            elif webhook_code == "HISTORICAL_UPDATE":
                logger.info(f"Historical update for Plaid item {item_id}")
                # Historical transactions available
                return {"status": "acknowledged", "message": "Historical update received"}
                
            elif webhook_code == "DEFAULT_UPDATE":
                logger.info(f"Default update for Plaid item {item_id}")
                # New transactions are available
                
                # Schedule a sync for recent transactions
                service = PlaidService(db_session)
                start_date = datetime.date.today() - datetime.timedelta(days=30)
                end_date = datetime.date.today()
                
                await service.sync_transactions(
                    tenant.id,
                    start_date=start_date,
                    end_date=end_date,
                )
                
                return {"status": "success", "message": "Transactions synced"}
                
            elif webhook_code == "TRANSACTIONS_REMOVED":
                logger.info(f"Transactions removed for Plaid item {item_id}")
                # Transactions were removed
                # TODO: Mark transactions as deleted in our database
                return {"status": "acknowledged", "message": "Transactions removed event received"}
        
        # Handle other webhook types (AUTH, ITEM, etc.)
        return {
            "status": "acknowledged", 
            "message": f"Webhook {webhook_type}:{webhook_code} received"
        }
        
    except Exception as exc:
        logger.error(f"Error handling Plaid webhook: {exc}")
        return {"status": "error", "message": str(exc)} 