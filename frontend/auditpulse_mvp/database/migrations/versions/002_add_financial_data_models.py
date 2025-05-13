"""Add financial data models.

Revision ID: 002
Revises: 001
Create Date: 2024-03-21 09:00:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade database schema."""
    # Create financial_institutions table
    op.create_table(
        "financial_institutions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("plaid_access_token", sa.String(), nullable=False),
        sa.Column("plaid_item_id", sa.String(), nullable=False),
        sa.Column("plaid_institution_id", sa.String(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, default=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("last_updated", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_financial_institutions_user_id",
        "financial_institutions",
        ["user_id"],
    )
    op.create_index(
        "ix_financial_institutions_plaid_item_id",
        "financial_institutions",
        ["plaid_item_id"],
    )

    # Create financial_accounts table
    op.create_table(
        "financial_accounts",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("institution_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("plaid_account_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("official_name", sa.String(), nullable=True),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("subtype", sa.String(), nullable=True),
        sa.Column("mask", sa.String(), nullable=True),
        sa.Column("balances", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, default=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("last_updated", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["institution_id"],
            ["financial_institutions.id"],
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_financial_accounts_user_id",
        "financial_accounts",
        ["user_id"],
    )
    op.create_index(
        "ix_financial_accounts_plaid_account_id",
        "financial_accounts",
        ["plaid_account_id"],
    )

    # Create financial_transactions table
    op.create_table(
        "financial_transactions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("account_id", sa.String(), nullable=False),
        sa.Column("transaction_id", sa.String(), nullable=False),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("date", sa.DateTime(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("merchant_name", sa.String(), nullable=True),
        sa.Column("pending", sa.Boolean(), nullable=False, default=False),
        sa.Column("category", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("category_id", sa.String(), nullable=True),
        sa.Column("transaction_type", sa.String(), nullable=True),
        sa.Column("payment_channel", sa.String(), nullable=True),
        sa.Column("location", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("last_updated", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_financial_transactions_user_id",
        "financial_transactions",
        ["user_id"],
    )
    op.create_index(
        "ix_financial_transactions_account_id",
        "financial_transactions",
        ["account_id"],
    )
    op.create_index(
        "ix_financial_transactions_transaction_id",
        "financial_transactions",
        ["transaction_id"],
        unique=True,
    )
    op.create_index(
        "ix_financial_transactions_date",
        "financial_transactions",
        ["date"],
    )


def downgrade():
    """Downgrade database schema."""
    op.drop_index("ix_financial_transactions_date", table_name="financial_transactions")
    op.drop_index(
        "ix_financial_transactions_transaction_id", table_name="financial_transactions"
    )
    op.drop_index(
        "ix_financial_transactions_account_id", table_name="financial_transactions"
    )
    op.drop_index(
        "ix_financial_transactions_user_id", table_name="financial_transactions"
    )
    op.drop_table("financial_transactions")

    op.drop_index(
        "ix_financial_accounts_plaid_account_id", table_name="financial_accounts"
    )
    op.drop_index("ix_financial_accounts_user_id", table_name="financial_accounts")
    op.drop_table("financial_accounts")

    op.drop_index(
        "ix_financial_institutions_plaid_item_id", table_name="financial_institutions"
    )
    op.drop_index(
        "ix_financial_institutions_user_id", table_name="financial_institutions"
    )
    op.drop_table("financial_institutions")
