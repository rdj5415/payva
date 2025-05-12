"""Plaid API endpoints for financial data integration."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel, Field

from auditpulse_mvp.api.api_v1.auth import get_current_user
from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.database.models import User
from auditpulse_mvp.integrations.plaid.client import PlaidClient
from auditpulse_mvp.services.financial_account_service import FinancialAccountService

router = APIRouter(prefix="/plaid", tags=["Plaid Integration"])

# Pydantic models
class LinkTokenCreateResponse(BaseModel):
    """Link token creation response."""
    link_token: str = Field(..., description="Plaid Link token")
    expiration: datetime = Field(..., description="Token expiration time")
    request_id: str = Field(..., description="Request ID")

class PublicTokenExchangeRequest(BaseModel):
    """Public token exchange request."""
    public_token: str = Field(..., description="Public token from Plaid Link")
    institution_name: Optional[str] = Field(None, description="Institution name")
    institution_id: Optional[str] = Field(None, description="Institution ID")

class AccountResponse(BaseModel):
    """Account response."""
    id: str
    name: str
    mask: Optional[str]
    type: str
    subtype: Optional[str]
    balances: Dict[str, Any]
    institution_name: Optional[str]
    
class AccountsResponse(BaseModel):
    """Accounts response."""
    accounts: List[AccountResponse]
    item_id: str
    institution_id: Optional[str]
    
class TransactionResponse(BaseModel):
    """Transaction response."""
    id: str
    account_id: str
    amount: float
    date: datetime
    name: str
    merchant_name: Optional[str]
    category: List[str]
    pending: bool
    transaction_type: Optional[str]
    payment_channel: Optional[str]
    location: Optional[Dict[str, Any]]
    
class TransactionsResponse(BaseModel):
    """Transactions response."""
    transactions: List[TransactionResponse]
    accounts: List[AccountResponse]
    total_transactions: int
    
# Dependencies
async def get_plaid_client(
    user: User = Depends(get_current_user),
) -> PlaidClient:
    """Get Plaid client."""
    return PlaidClient()

async def get_financial_account_service(
    db = Depends(get_db_session),
    user: User = Depends(get_current_user),
) -> FinancialAccountService:
    """Get financial account service."""
    return FinancialAccountService(db_session=db, user=user)

# Endpoints
@router.post(
    "/link-token",
    response_model=LinkTokenCreateResponse,
    status_code=status.HTTP_200_OK,
)
async def create_link_token(
    plaid_client: PlaidClient = Depends(get_plaid_client),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Create a Plaid Link token for connecting bank accounts.
    
    This endpoint generates a Link token that can be used with the Plaid Link
    frontend integration to securely connect financial accounts.
    
    Returns:
        LinkTokenCreateResponse: Link token details
    """
    try:
        response = await plaid_client.create_link_token(user_id=str(user.id))
        return {
            "link_token": response["link_token"],
            "expiration": response["expiration"],
            "request_id": response["request_id"],
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create link token: {str(e)}",
        )

@router.post(
    "/exchange-token",
    response_model=Dict[str, str],
    status_code=status.HTTP_200_OK,
)
async def exchange_public_token(
    request: PublicTokenExchangeRequest,
    plaid_client: PlaidClient = Depends(get_plaid_client),
    financial_account_service: FinancialAccountService = Depends(get_financial_account_service),
    user: User = Depends(get_current_user),
) -> Dict[str, str]:
    """Exchange a public token for an access token and store account information.
    
    This endpoint exchanges a public token received from Plaid Link for an
    access token, which is then stored for future use.
    
    Args:
        request: Public token exchange request
        
    Returns:
        Dict[str, str]: Status message
    """
    try:
        # Exchange public token for access token
        exchange_response = await plaid_client.exchange_public_token(
            public_token=request.public_token
        )
        
        # Store access token and fetch account information
        await financial_account_service.store_plaid_access_token(
            access_token=exchange_response["access_token"],
            item_id=exchange_response["item_id"],
            institution_name=request.institution_name,
            institution_id=request.institution_id,
        )
        
        return {"status": "success", "message": "Account connected successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to exchange token: {str(e)}",
        )

@router.get(
    "/accounts",
    response_model=AccountsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_accounts(
    item_id: Optional[str] = Query(None, description="Item ID to filter by"),
    financial_account_service: FinancialAccountService = Depends(get_financial_account_service),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get connected financial accounts.
    
    This endpoint retrieves information about the user's connected financial accounts.
    
    Args:
        item_id: Optional item ID to filter by
        
    Returns:
        AccountsResponse: Account information
    """
    try:
        accounts = await financial_account_service.get_accounts(item_id=item_id)
        return accounts
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get accounts: {str(e)}",
        )

@router.get(
    "/transactions",
    response_model=TransactionsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_transactions(
    start_date: datetime = Query(..., description="Start date for transactions"),
    end_date: Optional[datetime] = Query(None, description="End date for transactions"),
    account_id: Optional[str] = Query(None, description="Account ID to filter by"),
    item_id: Optional[str] = Query(None, description="Item ID to filter by"),
    count: int = Query(100, ge=1, le=500, description="Number of transactions to fetch"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    financial_account_service: FinancialAccountService = Depends(get_financial_account_service),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get financial transactions.
    
    This endpoint retrieves financial transactions for the user's connected accounts.
    
    Args:
        start_date: Start date for transactions
        end_date: End date for transactions (defaults to today)
        account_id: Optional account ID to filter by
        item_id: Optional item ID to filter by
        count: Number of transactions to fetch
        offset: Pagination offset
        
    Returns:
        TransactionsResponse: Transaction data
    """
    try:
        if end_date is None:
            end_date = datetime.now()
            
        transactions = await financial_account_service.get_transactions(
            start_date=start_date,
            end_date=end_date,
            account_id=account_id,
            item_id=item_id,
            count=count,
            offset=offset,
        )
        
        return transactions
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get transactions: {str(e)}",
        )

@router.post(
    "/sync",
    response_model=Dict[str, str],
    status_code=status.HTTP_200_OK,
)
async def sync_transactions(
    days: int = Query(30, ge=1, le=90, description="Number of days to sync"),
    financial_account_service: FinancialAccountService = Depends(get_financial_account_service),
    user: User = Depends(get_current_user),
) -> Dict[str, str]:
    """Sync transactions for all connected accounts.
    
    This endpoint triggers a synchronization of financial transactions for all
    connected accounts for the specified number of days.
    
    Args:
        days: Number of days to sync
        
    Returns:
        Dict[str, str]: Status message
    """
    try:
        start_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()
        
        # Queue sync task
        await financial_account_service.queue_transaction_sync(
            start_date=start_date,
            end_date=end_date,
        )
        
        return {
            "status": "success",
            "message": f"Transaction sync queued for the last {days} days",
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync transactions: {str(e)}",
        ) 