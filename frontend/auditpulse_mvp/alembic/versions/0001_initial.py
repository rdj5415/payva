"""Initial database tables.

Revision ID: 0001
Revises:
Create Date: 2023-11-01 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial database tables."""
    # Create enum types
    op.execute(
        "CREATE TYPE datasource AS ENUM ('quickbooks', 'plaid', 'manual', 'netsuite', 'csv')"
    )
    op.execute("CREATE TYPE anomalytype AS ENUM ('rules_based', 'ml_based', 'manual')")
    op.execute(
        "CREATE TYPE feedbacktype AS ENUM ('true_positive', 'false_positive', 'ignore', 'needs_review')"
    )

    # Create tenants table
    op.create_table(
        "tenants",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, default=True),
        sa.Column(
            "subscription_tier", sa.String(50), nullable=False, default="standard"
        ),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "quickbooks_settings",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "plaid_settings", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("enable_ml_engine", sa.Boolean(), nullable=False, default=True),
        sa.Column(
            "enable_gpt_explanations", sa.Boolean(), nullable=False, default=True
        ),
        sa.Column("enable_demo_mode", sa.Boolean(), nullable=False, default=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug"),
    )

    # Create users table
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("role", sa.String(50), nullable=False, server_default="viewer"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("is_verified", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("last_login", sa.DateTime(), nullable=True),
        sa.Column(
            "email_notifications", sa.Boolean(), nullable=False, server_default="true"
        ),
        sa.Column(
            "slack_notifications", sa.Boolean(), nullable=False, server_default="false"
        ),
        sa.Column(
            "sms_notifications", sa.Boolean(), nullable=False, server_default="false"
        ),
        sa.Column("phone_number", sa.String(20), nullable=True),
        sa.Column("slack_user_id", sa.String(50), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_index(op.f("ix_users_tenant_id"), "users", ["tenant_id"], unique=False)

    # Create transactions table
    op.create_table(
        "transactions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("transaction_id", sa.String(255), nullable=False),
        sa.Column(
            "source",
            sa.Enum(
                "quickbooks", "plaid", "manual", "netsuite", "csv", name="datasource"
            ),
            nullable=False,
        ),
        sa.Column("source_account_id", sa.String(255), nullable=False),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("currency", sa.String(10), nullable=False, server_default="USD"),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", sa.String(255), nullable=True),
        sa.Column("merchant_name", sa.String(255), nullable=True),
        sa.Column("transaction_date", sa.DateTime(), nullable=False),
        sa.Column("posting_date", sa.DateTime(), nullable=True),
        sa.Column("raw_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "transaction_id",
            "source",
            name="uq_transaction_per_tenant_source",
        ),
    )
    op.create_index(
        op.f("ix_transactions_tenant_id"), "transactions", ["tenant_id"], unique=False
    )

    # Create anomalies table
    op.create_table(
        "anomalies",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("transaction_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "anomaly_type",
            sa.Enum("rules_based", "ml_based", "manual", name="anomalytype"),
            nullable=False,
        ),
        sa.Column("rule_name", sa.String(255), nullable=True),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("ai_explanation", sa.Text(), nullable=True),
        sa.Column(
            "feedback",
            sa.Enum(
                "true_positive",
                "false_positive",
                "ignore",
                "needs_review",
                name="feedbacktype",
            ),
            nullable=True,
        ),
        sa.Column("feedback_notes", sa.Text(), nullable=True),
        sa.Column("resolved", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("resolved_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        sa.Column(
            "notification_sent", sa.Boolean(), nullable=False, server_default="false"
        ),
        sa.Column("notification_sent_at", sa.DateTime(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["transaction_id"], ["transactions.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_anomalies_tenant_id"), "anomalies", ["tenant_id"], unique=False
    )

    # Create audit_logs table
    op.create_table(
        "audit_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("action", sa.String(255), nullable=False),
        sa.Column("resource_type", sa.String(255), nullable=False),
        sa.Column("resource_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.String(255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_audit_logs_tenant_id"), "audit_logs", ["tenant_id"], unique=False
    )


def downgrade() -> None:
    """Drop all tables in reverse order of creation."""
    op.drop_table("audit_logs")
    op.drop_table("anomalies")
    op.drop_table("transactions")
    op.drop_table("users")
    op.drop_table("tenants")
    op.execute("DROP TYPE feedbacktype")
    op.execute("DROP TYPE anomalytype")
    op.execute("DROP TYPE datasource")
