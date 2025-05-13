"""Plaid synchronization tasks."""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.database.models import (
    FinancialInstitution,
    FinancialTransaction,
    User,
)
from auditpulse_mvp.integrations.plaid.client import PlaidClient
from auditpulse_mvp.utils.settings import get_settings, Settings

logger = logging.getLogger(__name__)


async def sync_plaid_transactions(
    user_id: str,
    institution_id: str,
    start_date: str,
    end_date: str,
    db_session: Optional[AsyncSession] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Sync transactions from Plaid for a financial institution.

    Args:
        user_id: User ID
        institution_id: Financial institution ID
        start_date: Start date for transactions (ISO format)
        end_date: End date for transactions (ISO format)
        db_session: Database session
        settings: Application settings

    Returns:
        Dict[str, Any]: Sync result
    """
    # Get database session if not provided
    if db_session is None:
        db_session = await anext(get_db_session())

    # Get settings if not provided
    if settings is None:
        settings = get_settings()

    try:
        # Parse dates
        start_date_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        end_date_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

        # Get user
        stmt = select(User).where(User.id == uuid.UUID(user_id))
        result = await db_session.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            logger.error(f"User not found: {user_id}")
            return {"status": "error", "message": "User not found"}

        # Get institution
        stmt = select(FinancialInstitution).where(
            FinancialInstitution.id == uuid.UUID(institution_id)
        )
        result = await db_session.execute(stmt)
        institution = result.scalar_one_or_none()

        if not institution:
            logger.error(f"Institution not found: {institution_id}")
            return {"status": "error", "message": "Institution not found"}

        # Initialize Plaid client
        plaid_client = PlaidClient(settings=settings)

        # Get transactions from Plaid
        transactions = await plaid_client.get_all_transactions(
            access_token=institution.plaid_access_token,
            start_date=start_date_dt,
            end_date=end_date_dt,
        )

        # Save transactions to database
        await store_transactions(
            db_session=db_session,
            user=user,
            transactions=transactions,
        )

        return {
            "status": "success",
            "transactions_count": len(transactions),
            "institution_id": str(institution.id),
            "user_id": str(user.id),
            "start_date": start_date,
            "end_date": end_date,
        }

    except Exception as e:
        logger.error(f"Error syncing Plaid transactions: {e}")
        return {
            "status": "error",
            "message": str(e),
            "user_id": user_id,
            "institution_id": institution_id,
            "start_date": start_date,
            "end_date": end_date,
        }


async def store_transactions(
    db_session: AsyncSession,
    user: User,
    transactions: List[Dict[str, Any]],
) -> int:
    """Store transactions in the database.

    Args:
        db_session: Database session
        user: User object
        transactions: List of transaction dictionaries from Plaid

    Returns:
        int: Number of transactions stored
    """
    stored_count = 0

    for transaction_data in transactions:
        # Check if transaction already exists
        stmt = select(FinancialTransaction).where(
            FinancialTransaction.transaction_id == transaction_data["transaction_id"]
        )
        result = await db_session.execute(stmt)
        transaction = result.scalar_one_or_none()

        if transaction:
            # Update existing transaction
            transaction.amount = transaction_data["amount"]
            transaction.name = transaction_data["name"]
            transaction.merchant_name = transaction_data.get("merchant_name")
            transaction.pending = transaction_data["pending"]
            transaction.category = transaction_data.get("category", [])
            transaction.category_id = transaction_data.get("category_id")
            transaction.transaction_type = transaction_data.get("transaction_type")
            transaction.payment_channel = transaction_data.get("payment_channel")
            transaction.location = transaction_data.get("location", {})
            transaction.last_updated = datetime.now()
        else:
            # Create new transaction
            transaction = FinancialTransaction(
                user_id=user.id,
                account_id=transaction_data["account_id"],
                transaction_id=transaction_data["transaction_id"],
                amount=transaction_data["amount"],
                date=datetime.fromisoformat(
                    transaction_data["date"].replace("Z", "+00:00")
                ),
                name=transaction_data["name"],
                merchant_name=transaction_data.get("merchant_name"),
                pending=transaction_data["pending"],
                category=transaction_data.get("category", []),
                category_id=transaction_data.get("category_id"),
                transaction_type=transaction_data.get("transaction_type"),
                payment_channel=transaction_data.get("payment_channel"),
                location=transaction_data.get("location", {}),
                created_at=datetime.now(),
                last_updated=datetime.now(),
            )
            db_session.add(transaction)
            stored_count += 1

    await db_session.commit()
    return stored_count
