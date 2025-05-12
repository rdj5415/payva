"""QuickBooks ingestion module.

This module handles QuickBooks OAuth authentication, data fetching, and normalization.
"""
import datetime
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple, cast

import httpx
from fastapi import Depends, HTTPException, status
from intuitlib.client import AuthClient
from intuitlib.enums import Scopes
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


class QuickBooksClient:
    """Client for interacting with the QuickBooks API."""

    def __init__(
        self,
        tenant_id: uuid.UUID,
        client_id: str = settings.QUICKBOOKS_CLIENT_ID or "",
        client_secret: str = settings.QUICKBOOKS_CLIENT_SECRET.get_secret_value() if settings.QUICKBOOKS_CLIENT_SECRET else "",
        redirect_uri: str = settings.QUICKBOOKS_REDIRECT_URI or "",
        environment: str = settings.QUICKBOOKS_ENVIRONMENT,
    ):
        """Initialize the QuickBooks client.

        Args:
            tenant_id: The tenant ID.
            client_id: The QuickBooks client ID.
            client_secret: The QuickBooks client secret.
            redirect_uri: The redirect URI for OAuth.
            environment: The QuickBooks environment (sandbox or production).
        """
        if not client_id or not client_secret or not redirect_uri:
            raise ValueError("QuickBooks credentials are not configured")

        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.environment = environment
        self.auth_client = AuthClient(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            environment=environment,
        )
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.realm_id: Optional[str] = None

    def get_authorization_url(self, state: str) -> str:
        """Get the QuickBooks authorization URL.

        Args:
            state: A state parameter for CSRF protection.

        Returns:
            str: The authorization URL.
        """
        scopes = [
            Scopes.ACCOUNTING,
            Scopes.OPENID,
            Scopes.EMAIL,
            Scopes.PROFILE,
        ]
        return self.auth_client.get_authorization_url(scopes, state)

    async def exchange_code_for_token(self, code: str, realm_id: str) -> Dict[str, Any]:
        """Exchange the authorization code for an access token.

        Args:
            code: The authorization code.
            realm_id: The QuickBooks company ID.

        Returns:
            Dict[str, Any]: The token response.
        """
        self.auth_client.get_bearer_token(code, realm_id)
        self.access_token = self.auth_client.access_token
        self.refresh_token = self.auth_client.refresh_token
        self.realm_id = realm_id

        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "realm_id": self.realm_id,
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        reraise=True,
    )
    async def fetch_transactions(
        self, start_date: Optional[datetime.date] = None, end_date: Optional[datetime.date] = None
    ) -> List[Dict[str, Any]]:
        """Fetch transactions from QuickBooks.

        Args:
            start_date: The start date for fetching transactions.
            end_date: The end date for fetching transactions.

        Returns:
            List[Dict[str, Any]]: The list of transactions.

        Raises:
            HTTPException: If there's an error fetching transactions.
        """
        if not self.access_token or not self.realm_id:
            raise ValueError("Access token or realm ID is missing")

        # Default to last 90 days if no date range is provided
        if not start_date:
            start_date = datetime.date.today() - datetime.timedelta(days=90)
        if not end_date:
            end_date = datetime.date.today()

        # Convert dates to QuickBooks format
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        # Build the query - fetch all transactions including purchases and bills
        query = (
            f"SELECT * FROM Purchase WHERE TxnDate >= '{start_date_str}' AND "
            f"TxnDate <= '{end_date_str}' ORDER BY TxnDate DESC MAXRESULTS 1000"
        )

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
            "Content-Type": "application/text",
        }

        base_url = (
            "https://sandbox-quickbooks.api.intuit.com"
            if self.environment == "sandbox"
            else "https://quickbooks.api.intuit.com"
        )
        url = f"{base_url}/v3/company/{self.realm_id}/query"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, data=query)
                response.raise_for_status()
                data = response.json()
                return data.get("QueryResponse", {}).get("Purchase", [])
        except httpx.HTTPStatusError as exc:
            logger.error(f"HTTP error fetching QuickBooks transactions: {exc}")
            if exc.response.status_code == 401:
                await self.refresh_access_token()
                # Retry the request after refreshing the token
                return await self.fetch_transactions(start_date, end_date)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Error fetching transactions from QuickBooks: {exc}",
            )
        except httpx.RequestError as exc:
            logger.error(f"Network error fetching QuickBooks transactions: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Network error communicating with QuickBooks: {exc}",
            )
        except Exception as exc:
            logger.error(f"Unexpected error fetching QuickBooks transactions: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error fetching QuickBooks transactions: {exc}",
            )

    async def refresh_access_token(self) -> Dict[str, Any]:
        """Refresh the QuickBooks access token.

        Returns:
            Dict[str, Any]: The refreshed token response.
        """
        if not self.refresh_token:
            raise ValueError("Refresh token is missing")

        try:
            self.auth_client.refresh(refresh_token=self.refresh_token)
            self.access_token = self.auth_client.access_token
            self.refresh_token = self.auth_client.refresh_token

            return {
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
                "realm_id": self.realm_id,
            }
        except Exception as exc:
            logger.error(f"Error refreshing QuickBooks token: {exc}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Error refreshing QuickBooks token: {exc}",
            )


class QuickBooksService:
    """Service for ingesting and normalizing QuickBooks data."""

    def __init__(self, db_session: AsyncSession = Depends(get_db_session)):
        """Initialize the QuickBooks service.

        Args:
            db_session: The database session.
        """
        self.db_session = db_session

    async def get_client_for_tenant(self, tenant_id: uuid.UUID) -> QuickBooksClient:
        """Get a QuickBooks client for the specified tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            QuickBooksClient: The QuickBooks client.

        Raises:
            HTTPException: If the tenant or QuickBooks settings are not found.
        """
        # Get the tenant and its QuickBooks settings
        tenant_result = await self.db_session.execute(
            select(Tenant).where(Tenant.id == tenant_id)
        )
        tenant = tenant_result.scalar_one_or_none()

        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tenant {tenant_id} not found",
            )

        if not tenant.quickbooks_settings:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"QuickBooks not configured for tenant {tenant_id}",
            )

        qb_settings = cast(Dict[str, Any], tenant.quickbooks_settings)
        
        # Create and return a client with the tenant's settings
        client = QuickBooksClient(
            tenant_id=tenant_id,
            client_id=settings.QUICKBOOKS_CLIENT_ID or "",
            client_secret=settings.QUICKBOOKS_CLIENT_SECRET.get_secret_value() if settings.QUICKBOOKS_CLIENT_SECRET else "",
            redirect_uri=settings.QUICKBOOKS_REDIRECT_URI or "",
            environment=settings.QUICKBOOKS_ENVIRONMENT,
        )
        
        # Set the tenant-specific tokens and realm ID
        client.access_token = qb_settings.get("access_token")
        client.refresh_token = qb_settings.get("refresh_token")
        client.realm_id = qb_settings.get("realm_id")
        
        return client

    async def sync_transactions(
        self, tenant_id: uuid.UUID, start_date: Optional[datetime.date] = None, end_date: Optional[datetime.date] = None
    ) -> Tuple[int, int, int]:
        """Sync transactions from QuickBooks.

        Args:
            tenant_id: The tenant ID.
            start_date: The start date for fetching transactions.
            end_date: The end date for fetching transactions.

        Returns:
            Tuple[int, int, int]: A tuple of (fetched, created, updated) counts.

        Raises:
            HTTPException: If there's an error syncing transactions.
        """
        client = await self.get_client_for_tenant(tenant_id)
        
        try:
            # Fetch transactions from QuickBooks
            raw_transactions = await client.fetch_transactions(start_date, end_date)
            
            # Process and normalize transactions
            created_count, updated_count = await self._process_transactions(
                tenant_id, raw_transactions
            )
            
            return len(raw_transactions), created_count, updated_count
        except Exception as exc:
            logger.error(f"Error syncing QuickBooks transactions: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error syncing QuickBooks transactions: {exc}",
            )

    async def _process_transactions(
        self, tenant_id: uuid.UUID, raw_transactions: List[Dict[str, Any]]
    ) -> Tuple[int, int]:
        """Process and normalize QuickBooks transactions.

        Args:
            tenant_id: The tenant ID.
            raw_transactions: The list of raw transactions from QuickBooks.

        Returns:
            Tuple[int, int]: A tuple of (created, updated) counts.
        """
        created_count = 0
        updated_count = 0
        
        for raw_txn in raw_transactions:
            # Convert QuickBooks format to our normalized format
            normalized_txn = self._normalize_transaction(raw_txn)
            
            # Check if the transaction already exists
            existing_txn_result = await self.db_session.execute(
                select(Transaction).where(
                    Transaction.tenant_id == tenant_id,
                    Transaction.transaction_id == normalized_txn["transaction_id"],
                    Transaction.source == DataSource.QUICKBOOKS,
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
                    source=DataSource.QUICKBOOKS,
                    **normalized_txn,
                )
                self.db_session.add(new_txn)
                created_count += 1
        
        # Commit all changes at once
        await self.db_session.commit()
        
        return created_count, updated_count

    def _normalize_transaction(self, raw_txn: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a QuickBooks transaction to our data model.

        Args:
            raw_txn: The raw transaction from QuickBooks.

        Returns:
            Dict[str, Any]: The normalized transaction data.
        """
        # Extract basic transaction information
        txn_id = raw_txn.get("Id") or ""
        
        # Determine if it's a purchase or bill
        txn_type = "Purchase"
        if raw_txn.get("PaymentType") == "Check":
            txn_type = "Check"
        elif raw_txn.get("DocNumber", "").startswith("BILL"):
            txn_type = "Bill"
        
        # Extract account information
        account_id = ""
        account_ref = raw_txn.get("AccountRef")
        if account_ref:
            account_id = account_ref.get("value", "")
        
        # Extract vendor information
        merchant_name = ""
        entity_ref = raw_txn.get("EntityRef")
        if entity_ref:
            merchant_name = entity_ref.get("name", "")
        
        # Extract transaction amount
        amount = 0.0
        total_amount = raw_txn.get("TotalAmt")
        if total_amount is not None:
            amount = float(total_amount)
        
        # Extract transaction date
        transaction_date = datetime.datetime.now()
        txn_date = raw_txn.get("TxnDate")
        if txn_date:
            try:
                transaction_date = datetime.datetime.strptime(txn_date, "%Y-%m-%d")
            except ValueError:
                pass
        
        # Extract description and category
        description = raw_txn.get("PrivateNote") or ""
        category = ""
        
        # Combine all line items for a richer description if needed
        line_items = raw_txn.get("Line", [])
        if line_items and not description:
            descriptions = []
            for item in line_items:
                detail = item.get("Description", "")
                if detail:
                    descriptions.append(detail)
            if descriptions:
                description = " | ".join(descriptions)
        
        # Determine category from line items if available
        if line_items:
            for item in line_items:
                account_detail = item.get("AccountBasedExpenseLineDetail")
                if account_detail:
                    account_ref = account_detail.get("AccountRef")
                    if account_ref:
                        category = account_ref.get("name", "")
                        break
        
        # Build the normalized transaction
        return {
            "transaction_id": f"{txn_id}",
            "source_account_id": account_id,
            "amount": amount,
            "currency": "USD",  # QuickBooks US typically uses USD
            "description": description,
            "category": category,
            "merchant_name": merchant_name,
            "transaction_date": transaction_date,
            "posting_date": transaction_date,  # Use same date as transaction_date
            "raw_data": raw_txn,  # Store the entire raw transaction
        }


async def update_tenant_quickbooks_settings(
    db_session: AsyncSession, tenant_id: uuid.UUID, settings_data: Dict[str, Any]
) -> bool:
    """Update the QuickBooks settings for a tenant.

    Args:
        db_session: The database session.
        tenant_id: The tenant ID.
        settings_data: The QuickBooks settings data.

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
        if tenant.quickbooks_settings:
            # Update existing settings
            tenant.quickbooks_settings.update(settings_data)
        else:
            # Create new settings
            tenant.quickbooks_settings = settings_data
        
        # Commit the changes
        await db_session.commit()
        return True
    except Exception as exc:
        logger.error(f"Error updating QuickBooks settings: {exc}")
        await db_session.rollback()
        return False


# Webhook handler for QuickBooks events
async def handle_quickbooks_webhook(
    db_session: AsyncSession, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Handle a QuickBooks webhook event.

    Args:
        db_session: The database session.
        payload: The webhook payload.

    Returns:
        Dict[str, Any]: The response.
    """
    try:
        # Extract event information
        event_notification = payload.get("eventNotifications", [{}])[0]
        realm_id = event_notification.get("realmId")
        
        if not realm_id:
            logger.error("Missing realm ID in QuickBooks webhook")
            return {"status": "error", "message": "Missing realm ID"}
        
        # Find the tenant with this realm ID
        tenant_result = await db_session.execute(
            select(Tenant).where(
                Tenant.quickbooks_settings["realm_id"].astext == realm_id
            )
        )
        tenant = tenant_result.scalar_one_or_none()
        
        if not tenant:
            logger.error(f"No tenant found for QuickBooks realm ID: {realm_id}")
            return {"status": "error", "message": f"No tenant found for realm ID: {realm_id}"}
        
        # Process the event
        events = event_notification.get("dataChangeEvent", {}).get("entities", [])
        
        if not events:
            return {"status": "success", "message": "No events to process"}
        
        # Schedule background jobs to sync affected data
        for event in events:
            entity_name = event.get("name")
            operation = event.get("operation")
            
            # If it's a transaction-related entity, trigger a sync
            if entity_name in ["Purchase", "Bill", "Check", "JournalEntry"]:
                logger.info(f"QuickBooks {operation} event for {entity_name}")
                # TODO: Schedule a background job instead of synchronous processing
                service = QuickBooksService(db_session)
                await service.sync_transactions(
                    tenant.id,
                    start_date=datetime.date.today() - datetime.timedelta(days=30),
                    end_date=datetime.date.today(),
                )
        
        return {
            "status": "success",
            "message": f"Processed {len(events)} QuickBooks events",
        }
    except Exception as exc:
        logger.error(f"Error handling QuickBooks webhook: {exc}")
        return {"status": "error", "message": str(exc)} 