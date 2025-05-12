"""Plaid API client for financial data integration.

This module provides integration with the Plaid API for retrieving financial data
from various institutions.
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import plaid
from plaid.api import plaid_api
from plaid.model.country_code import CountryCode
from plaid.model.products import Products
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions
from plaid.model.accounts_get_request import AccountsGetRequest
from plaid.model.item_get_request import ItemGetRequest
from fastapi import Depends

from auditpulse_mvp.utils.settings import get_settings, Settings

logger = logging.getLogger(__name__)

class PlaidClient:
    """Client for interacting with the Plaid API."""
    
    def __init__(self, settings: Settings = Depends(get_settings)):
        """Initialize the Plaid client.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        
        # Configure Plaid client
        configuration = plaid.Configuration(
            host=self._get_plaid_environment(),
            api_key={
                "clientId": settings.PLAID_CLIENT_ID,
                "secret": settings.PLAID_SECRET,
            }
        )
        
        api_client = plaid.ApiClient(configuration)
        self.client = plaid_api.PlaidApi(api_client)
        
    def _get_plaid_environment(self) -> str:
        """Get the Plaid API environment URL based on settings.
        
        Returns:
            str: Plaid API environment URL
        """
        if self.settings.PLAID_ENVIRONMENT == "sandbox":
            return plaid.Environment.Sandbox
        elif self.settings.PLAID_ENVIRONMENT == "development":
            return plaid.Environment.Development
        elif self.settings.PLAID_ENVIRONMENT == "production":
            return plaid.Environment.Production
        else:
            # Default to sandbox
            return plaid.Environment.Sandbox
            
    async def create_link_token(self, user_id: str) -> Dict[str, Any]:
        """Create a Plaid Link token for connecting bank accounts.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict[str, Any]: Link token creation response
        """
        try:
            request = LinkTokenCreateRequest(
                user=LinkTokenCreateRequestUser(
                    client_user_id=user_id
                ),
                client_name=self.settings.APP_NAME,
                products=[Products("transactions")],
                country_codes=[CountryCode("US")],
                language="en",
            )
            
            # Create link token
            response = self.client.link_token_create(request)
            return response.to_dict()
            
        except plaid.ApiException as e:
            logger.error(f"Error creating link token: {e}")
            raise
            
    async def exchange_public_token(self, public_token: str) -> Dict[str, Any]:
        """Exchange a public token for an access token.
        
        Args:
            public_token: Public token from Plaid Link
            
        Returns:
            Dict[str, Any]: Access token exchange response
        """
        try:
            response = self.client.item_public_token_exchange(
                {"public_token": public_token}
            )
            return response.to_dict()
            
        except plaid.ApiException as e:
            logger.error(f"Error exchanging public token: {e}")
            raise
            
    async def get_accounts(self, access_token: str) -> Dict[str, Any]:
        """Get account information.
        
        Args:
            access_token: Plaid API access token
            
        Returns:
            Dict[str, Any]: Account information
        """
        try:
            request = AccountsGetRequest(access_token=access_token)
            response = self.client.accounts_get(request)
            return response.to_dict()
            
        except plaid.ApiException as e:
            logger.error(f"Error getting accounts: {e}")
            raise
            
    async def get_item(self, access_token: str) -> Dict[str, Any]:
        """Get item information.
        
        Args:
            access_token: Plaid API access token
            
        Returns:
            Dict[str, Any]: Item information
        """
        try:
            request = ItemGetRequest(access_token=access_token)
            response = self.client.item_get(request)
            return response.to_dict()
            
        except plaid.ApiException as e:
            logger.error(f"Error getting item: {e}")
            raise
            
    async def get_transactions(
        self,
        access_token: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        account_ids: Optional[List[str]] = None,
        count: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get transactions.
        
        Args:
            access_token: Plaid API access token
            start_date: Start date for transactions
            end_date: End date for transactions (defaults to today)
            account_ids: List of account IDs to filter by
            count: Number of transactions to fetch
            offset: Offset for pagination
            
        Returns:
            Dict[str, Any]: Transaction data
        """
        try:
            if end_date is None:
                end_date = datetime.now()
                
            # Set up options
            options = TransactionsGetRequestOptions(
                count=count,
                offset=offset,
            )
            
            if account_ids:
                options.account_ids = account_ids
                
            # Create request
            request = TransactionsGetRequest(
                access_token=access_token,
                start_date=start_date.date(),
                end_date=end_date.date(),
                options=options,
            )
            
            # Get transactions
            response = self.client.transactions_get(request)
            return response.to_dict()
            
        except plaid.ApiException as e:
            logger.error(f"Error getting transactions: {e}")
            raise
            
    async def get_all_transactions(
        self,
        access_token: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        account_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get all transactions with pagination handling.
        
        Args:
            access_token: Plaid API access token
            start_date: Start date for transactions
            end_date: End date for transactions (defaults to today)
            account_ids: List of account IDs to filter by
            
        Returns:
            List[Dict[str, Any]]: All transactions
        """
        transactions = []
        has_more = True
        offset = 0
        
        while has_more:
            # Get batch of transactions
            response = await self.get_transactions(
                access_token=access_token,
                start_date=start_date,
                end_date=end_date,
                account_ids=account_ids,
                count=100,
                offset=offset,
            )
            
            # Add transactions to list
            transactions.extend(response["transactions"])
            
            # Check if there are more transactions
            has_more = response["total_transactions"] > len(transactions)
            offset += len(response["transactions"])
            
        return transactions 