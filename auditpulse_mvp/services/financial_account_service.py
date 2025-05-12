"""Financial account service for managing Plaid integrations and transaction data."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.database.models import User, FinancialInstitution, FinancialAccount, FinancialTransaction
from auditpulse_mvp.integrations.plaid.client import PlaidClient
from auditpulse_mvp.tasks.task_manager import TaskManager, TaskPriority

logger = logging.getLogger(__name__)

class FinancialAccountService:
    """Service for managing financial accounts and transactions."""
    
    def __init__(
        self,
        db_session: AsyncSession = Depends(get_db_session),
        user: User = Depends(),
        plaid_client: PlaidClient = Depends(),
        task_manager: TaskManager = Depends(),
    ):
        """Initialize the financial account service.
        
        Args:
            db_session: Database session
            user: Current user
            plaid_client: Plaid API client
            task_manager: Task manager instance
        """
        self.db = db_session
        self.user = user
        self.plaid_client = plaid_client
        self.task_manager = task_manager
        
    async def store_plaid_access_token(
        self,
        access_token: str,
        item_id: str,
        institution_name: Optional[str] = None,
        institution_id: Optional[str] = None,
    ) -> FinancialInstitution:
        """Store Plaid access token and institution information.
        
        Args:
            access_token: Plaid API access token
            item_id: Plaid item ID
            institution_name: Name of the financial institution
            institution_id: Plaid institution ID
            
        Returns:
            FinancialInstitution: Created or updated institution
        """
        # Check if institution already exists for this user and item ID
        stmt = (
            select(FinancialInstitution)
            .where(
                FinancialInstitution.user_id == self.user.id,
                FinancialInstitution.plaid_item_id == item_id,
            )
        )
        result = await self.db.execute(stmt)
        institution = result.scalar_one_or_none()
        
        if institution:
            # Update existing institution
            institution.plaid_access_token = access_token
            institution.plaid_institution_id = institution_id or institution.plaid_institution_id
            institution.name = institution_name or institution.name
            institution.last_updated = datetime.now()
        else:
            # Create new institution
            institution = FinancialInstitution(
                user_id=self.user.id,
                name=institution_name or "Financial Institution",
                plaid_access_token=access_token,
                plaid_item_id=item_id,
                plaid_institution_id=institution_id,
                is_active=True,
                created_at=datetime.now(),
                last_updated=datetime.now(),
            )
            self.db.add(institution)
            
        await self.db.commit()
        await self.db.refresh(institution)
        
        # Fetch and store account information
        await self._fetch_and_store_accounts(institution)
        
        return institution
        
    async def _fetch_and_store_accounts(self, institution: FinancialInstitution) -> List[FinancialAccount]:
        """Fetch and store account information for an institution.
        
        Args:
            institution: Financial institution
            
        Returns:
            List[FinancialAccount]: List of accounts
        """
        # Fetch accounts from Plaid
        accounts_response = await self.plaid_client.get_accounts(
            access_token=institution.plaid_access_token
        )
        
        accounts = []
        for account_data in accounts_response["accounts"]:
            # Check if account already exists
            stmt = (
                select(FinancialAccount)
                .where(
                    FinancialAccount.user_id == self.user.id,
                    FinancialAccount.institution_id == institution.id,
                    FinancialAccount.plaid_account_id == account_data["account_id"],
                )
            )
            result = await self.db.execute(stmt)
            account = result.scalar_one_or_none()
            
            if account:
                # Update existing account
                account.name = account_data["name"]
                account.official_name = account_data.get("official_name")
                account.type = account_data["type"]
                account.subtype = account_data.get("subtype")
                account.mask = account_data.get("mask")
                account.balances = account_data["balances"]
                account.last_updated = datetime.now()
            else:
                # Create new account
                account = FinancialAccount(
                    user_id=self.user.id,
                    institution_id=institution.id,
                    plaid_account_id=account_data["account_id"],
                    name=account_data["name"],
                    official_name=account_data.get("official_name"),
                    type=account_data["type"],
                    subtype=account_data.get("subtype"),
                    mask=account_data.get("mask"),
                    balances=account_data["balances"],
                    is_active=True,
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                )
                self.db.add(account)
                
            accounts.append(account)
            
        await self.db.commit()
        
        # Refresh accounts
        for account in accounts:
            await self.db.refresh(account)
            
        return accounts
        
    async def get_institutions(self) -> List[Dict[str, Any]]:
        """Get user's financial institutions.
        
        Returns:
            List[Dict[str, Any]]: List of financial institutions
        """
        # Get institutions from database
        stmt = (
            select(FinancialInstitution)
            .where(
                FinancialInstitution.user_id == self.user.id,
                FinancialInstitution.is_active == True,
            )
        )
        result = await self.db.execute(stmt)
        institutions = result.scalars().all()
        
        return [
            {
                "id": str(institution.id),
                "name": institution.name,
                "plaid_item_id": institution.plaid_item_id,
                "plaid_institution_id": institution.plaid_institution_id,
                "created_at": institution.created_at,
                "last_updated": institution.last_updated,
            }
            for institution in institutions
        ]
        
    async def get_accounts(self, item_id: Optional[str] = None) -> Dict[str, Any]:
        """Get user's financial accounts.
        
        Args:
            item_id: Optional Plaid item ID to filter by
            
        Returns:
            Dict[str, Any]: Account information
        """
        # Create query based on filters
        query = [
            FinancialAccount.user_id == self.user.id,
            FinancialAccount.is_active == True,
        ]
        
        if item_id:
            # Get institution first
            stmt = (
                select(FinancialInstitution)
                .where(
                    FinancialInstitution.user_id == self.user.id,
                    FinancialInstitution.plaid_item_id == item_id,
                )
            )
            result = await self.db.execute(stmt)
            institution = result.scalar_one_or_none()
            
            if not institution:
                return {"accounts": [], "item_id": item_id, "institution_id": None}
                
            query.append(FinancialAccount.institution_id == institution.id)
            institution_id = institution.plaid_institution_id
            item_id_result = institution.plaid_item_id
        else:
            # Use the first institution's info for the response
            stmt = (
                select(FinancialInstitution)
                .where(
                    FinancialInstitution.user_id == self.user.id,
                    FinancialInstitution.is_active == True,
                )
                .order_by(FinancialInstitution.created_at.desc())
                .limit(1)
            )
            result = await self.db.execute(stmt)
            institution = result.scalar_one_or_none()
            
            if not institution:
                return {"accounts": [], "item_id": None, "institution_id": None}
                
            institution_id = institution.plaid_institution_id
            item_id_result = institution.plaid_item_id
        
        # Get accounts
        stmt = (
            select(FinancialAccount)
            .where(and_(*query))
        )
        result = await self.db.execute(stmt)
        accounts = result.scalars().all()
        
        return {
            "accounts": [
                {
                    "id": account.plaid_account_id,
                    "name": account.name,
                    "mask": account.mask,
                    "type": account.type,
                    "subtype": account.subtype,
                    "balances": account.balances,
                    "institution_name": institution.name if institution else None,
                }
                for account in accounts
            ],
            "item_id": item_id_result,
            "institution_id": institution_id,
        }
        
    async def get_transactions(
        self,
        start_date: datetime,
        end_date: datetime,
        account_id: Optional[str] = None,
        item_id: Optional[str] = None,
        count: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get financial transactions.
        
        Args:
            start_date: Start date for transactions
            end_date: End date for transactions
            account_id: Optional account ID to filter by
            item_id: Optional item ID to filter by
            count: Number of transactions to fetch
            offset: Pagination offset
            
        Returns:
            Dict[str, Any]: Transaction data
        """
        # Create query based on filters
        query = [
            FinancialTransaction.user_id == self.user.id,
            FinancialTransaction.date >= start_date.date(),
            FinancialTransaction.date <= end_date.date(),
        ]
        
        account_ids = []
        if account_id:
            query.append(FinancialTransaction.account_id == account_id)
            account_ids = [account_id]
        elif item_id:
            # Get accounts for the item
            stmt = (
                select(FinancialInstitution)
                .where(
                    FinancialInstitution.user_id == self.user.id,
                    FinancialInstitution.plaid_item_id == item_id,
                )
            )
            result = await self.db.execute(stmt)
            institution = result.scalar_one_or_none()
            
            if institution:
                stmt = (
                    select(FinancialAccount.plaid_account_id)
                    .where(
                        FinancialAccount.user_id == self.user.id,
                        FinancialAccount.institution_id == institution.id,
                    )
                )
                result = await self.db.execute(stmt)
                account_ids = [row[0] for row in result.all()]
                
                if account_ids:
                    query.append(FinancialTransaction.account_id.in_(account_ids))
        
        # Get transactions count
        count_stmt = (
            select(func.count(FinancialTransaction.id))
            .where(and_(*query))
        )
        count_result = await self.db.execute(count_stmt)
        total_transactions = count_result.scalar_one()
        
        # Get transactions
        stmt = (
            select(FinancialTransaction)
            .where(and_(*query))
            .order_by(FinancialTransaction.date.desc())
            .offset(offset)
            .limit(count)
        )
        result = await self.db.execute(stmt)
        transactions = result.scalars().all()
        
        # Get accounts for the transactions
        accounts_response = await self.get_accounts(item_id=item_id)
        
        return {
            "transactions": [
                {
                    "id": transaction.transaction_id,
                    "account_id": transaction.account_id,
                    "amount": transaction.amount,
                    "date": transaction.date,
                    "name": transaction.name,
                    "merchant_name": transaction.merchant_name,
                    "category": transaction.category,
                    "pending": transaction.pending,
                    "transaction_type": transaction.transaction_type,
                    "payment_channel": transaction.payment_channel,
                    "location": transaction.location,
                }
                for transaction in transactions
            ],
            "accounts": accounts_response["accounts"],
            "total_transactions": total_transactions,
        }
        
    async def queue_transaction_sync(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> str:
        """Queue a task to sync transactions.
        
        Args:
            start_date: Start date for transactions
            end_date: End date for transactions
            
        Returns:
            str: Task ID
        """
        # Get all institutions
        stmt = (
            select(FinancialInstitution)
            .where(
                FinancialInstitution.user_id == self.user.id,
                FinancialInstitution.is_active == True,
            )
        )
        result = await self.db.execute(stmt)
        institutions = result.scalars().all()
        
        # Queue task for each institution
        task_ids = []
        for institution in institutions:
            task_id = await self.task_manager.schedule_task(
                task_name="sync_plaid_transactions",
                priority=TaskPriority.MEDIUM,
                kwargs={
                    "user_id": str(self.user.id),
                    "institution_id": str(institution.id),
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
            )
            task_ids.append(task_id)
            
        return task_ids[0] if task_ids else "No institutions to sync" 