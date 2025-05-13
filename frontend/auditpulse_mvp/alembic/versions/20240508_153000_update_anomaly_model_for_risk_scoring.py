"""update_anomaly_model_for_risk_scoring

Revision ID: 0002
Revises: 0001_initial
Create Date: 2024-05-08 15:30:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from alembic import op


# revision identifiers, used by Alembic.
revision: str = '0002'
down_revision: Union[str, None] = '0001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create new enum type for AnomalyType with updated values
    anomaly_type_enum = sa.Enum(
        'large_amount',
        'unusual_vendor',
        'duplicate_transaction',
        'statistical_outlier',
        'unauthorized_approver',
        'weekend_transaction',
        'round_number',
        'other',
        name='anomalytype',
    )
    
    # Create new enum type for FeedbackType with updated values
    feedback_type_enum = sa.Enum(
        'true_positive',
        'false_positive',
        'ignore',
        name='feedbacktype',
    )
    
    # Create temporary tables to handle the enum changes
    op.execute('CREATE TABLE anomalies_temp AS SELECT * FROM anomalies')
    
    # Drop the old table
    op.drop_table('anomalies')
    
    # Create enum types
    op.create_type('anomalytype_new', postgresql.ENUM(
        'large_amount',
        'unusual_vendor',
        'duplicate_transaction',
        'statistical_outlier',
        'unauthorized_approver',
        'weekend_transaction',
        'round_number',
        'other',
        name='anomalytype_new',
    ))
    
    op.create_type('feedbacktype_new', postgresql.ENUM(
        'true_positive',
        'false_positive',
        'ignore',
        name='feedbacktype_new',
    ))
    
    # Create the new anomalies table with updated schema
    op.create_table(
        'anomalies',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('transaction_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('anomaly_type', sa.Enum('large_amount', 'unusual_vendor', 'duplicate_transaction', 
                                          'statistical_outlier', 'unauthorized_approver', 
                                          'weekend_transaction', 'round_number', 'other', 
                                          name='anomalytype_new'), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('ml_score', sa.Float(), nullable=True),
        sa.Column('risk_score', sa.Integer(), nullable=True),
        sa.Column('is_flagged', sa.Boolean(), default=False, nullable=False),
        sa.Column('detection_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('is_resolved', sa.Boolean(), default=False, nullable=False),
        sa.Column('feedback', sa.Enum('true_positive', 'false_positive', 'ignore', 
                                      name='feedbacktype_new'), nullable=True),
        sa.Column('feedback_notes', sa.Text(), nullable=True),
        sa.Column('resolved_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), 
                  onupdate=sa.text('now()'), nullable=False),
        sa.Column('notification_sent', sa.Boolean(), default=False, nullable=False),
        sa.Column('notification_sent_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['transaction_id'], ['transactions.id'], ondelete='CASCADE'),
    )
    
    # Migrate data from temp table to new table with transformations
    if op.get_bind().dialect.name == 'postgresql':
        op.execute("""
            INSERT INTO anomalies (
                id, tenant_id, transaction_id, 
                anomaly_type, description, confidence, ml_score,
                risk_score, is_flagged, detection_metadata, is_resolved,
                feedback, feedback_notes, resolved_by, resolved_at,
                created_at, updated_at, notification_sent, notification_sent_at
            )
            SELECT 
                id, tenant_id, transaction_id,
                CASE
                    WHEN anomaly_type = 'rules_based' THEN 'other'::anomalytype_new
                    WHEN anomaly_type = 'ml_based' THEN 'statistical_outlier'::anomalytype_new
                    WHEN anomaly_type = 'manual' THEN 'other'::anomalytype_new
                    ELSE 'other'::anomalytype_new
                END,
                description,
                confidence_score,  -- Map confidence_score to confidence
                NULL,  -- ml_score starts NULL
                CAST(risk_score * 100 AS INTEGER),  -- Convert float risk_score to integer 0-100
                false,  -- is_flagged starts false
                jsonb_build_object('rule_name', rule_name, 'ai_explanation', ai_explanation),  -- Move fields to JSON
                resolved,  -- Map resolved to is_resolved
                feedback,  -- Keep feedback
                feedback_notes,
                resolved_by,
                resolved_at,
                created_at,
                updated_at,
                notification_sent,
                notification_sent_at
            FROM anomalies_temp
        """)
    
    # For SQLite or other databases, handle the conversion differently
    else:
        # Simple mapping for non-PostgreSQL databases (limited functionality)
        # In a real implementation, we would handle this more carefully
        op.execute("""
            INSERT INTO anomalies (
                id, tenant_id, transaction_id, 
                anomaly_type, description, confidence,
                risk_score, is_resolved,
                feedback, feedback_notes, resolved_by, resolved_at,
                created_at, updated_at, notification_sent, notification_sent_at
            )
            SELECT 
                id, tenant_id, transaction_id,
                'other',  -- Convert all to 'other' since we can't do CASE with enums in SQLite
                description,
                confidence_score,
                CAST(risk_score * 100 AS INTEGER),
                resolved,
                feedback,
                feedback_notes,
                resolved_by,
                resolved_at,
                created_at,
                updated_at,
                notification_sent,
                notification_sent_at
            FROM anomalies_temp
        """)
    
    # Update the tenant table to add metadata column
    # First rename the existing config column to metadata
    op.alter_column('tenants', 'config', new_column_name='metadata', 
                   existing_type=postgresql.JSONB(astext_type=sa.Text()))
    
    # Drop the temporary table
    op.drop_table('anomalies_temp')
    
    # Drop old enum types
    op.execute('DROP TYPE IF EXISTS anomalytype')
    op.execute('DROP TYPE IF EXISTS feedbacktype')
    
    # Rename new enum types to standard names
    op.execute('ALTER TYPE anomalytype_new RENAME TO anomalytype')
    op.execute('ALTER TYPE feedbacktype_new RENAME TO feedbacktype')


def downgrade() -> None:
    # Create temporary table
    op.execute('CREATE TABLE anomalies_temp AS SELECT * FROM anomalies')
    
    # Drop the current table
    op.drop_table('anomalies')
    
    # Create old enum types
    op.create_type('anomalytype_old', postgresql.ENUM(
        'rules_based',
        'ml_based',
        'manual',
        name='anomalytype_old'
    ))
    
    op.create_type('feedbacktype_old', postgresql.ENUM(
        'true_positive',
        'false_positive',
        'ignore',
        'needs_review',
        name='feedbacktype_old'
    ))
    
    # Create the original anomalies table
    op.create_table(
        'anomalies',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('transaction_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('anomaly_type', sa.Enum('rules_based', 'ml_based', 'manual',
                                         name='anomalytype_old'), nullable=False),
        sa.Column('rule_name', sa.String(255), nullable=True),
        sa.Column('risk_score', sa.Float(), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('ai_explanation', sa.Text(), nullable=True),
        sa.Column('feedback', sa.Enum('true_positive', 'false_positive', 'ignore', 'needs_review',
                                     name='feedbacktype_old'), nullable=True),
        sa.Column('feedback_notes', sa.Text(), nullable=True),
        sa.Column('resolved', sa.Boolean(), default=False, nullable=False),
        sa.Column('resolved_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), 
                 onupdate=sa.text('now()'), nullable=False),
        sa.Column('notification_sent', sa.Boolean(), default=False, nullable=False),
        sa.Column('notification_sent_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['transaction_id'], ['transactions.id'], ondelete='CASCADE'),
    )
    
    # Migrate data back from temp table to original table with transformations
    if op.get_bind().dialect.name == 'postgresql':
        op.execute("""
            INSERT INTO anomalies (
                id, tenant_id, transaction_id,
                anomaly_type, rule_name, risk_score, confidence_score,
                description, ai_explanation, feedback, feedback_notes, resolved,
                resolved_by, resolved_at, created_at, updated_at,
                notification_sent, notification_sent_at
            )
            SELECT 
                id, tenant_id, transaction_id,
                CASE
                    WHEN anomaly_type = 'statistical_outlier' THEN 'ml_based'::anomalytype_old
                    ELSE 'rules_based'::anomalytype_old
                END,
                detection_metadata->>'rule_name',
                CASE WHEN risk_score IS NULL THEN 0.5 ELSE risk_score::float / 100.0 END,
                confidence,
                description,
                detection_metadata->>'ai_explanation',
                feedback,
                feedback_notes,
                is_resolved,
                resolved_by,
                resolved_at,
                created_at,
                updated_at,
                notification_sent,
                notification_sent_at
            FROM anomalies_temp
        """)
    else:
        # Simple mapping for non-PostgreSQL databases
        op.execute("""
            INSERT INTO anomalies (
                id, tenant_id, transaction_id,
                anomaly_type, risk_score, confidence_score,
                description, feedback, feedback_notes, resolved,
                resolved_by, resolved_at, created_at, updated_at,
                notification_sent, notification_sent_at
            )
            SELECT 
                id, tenant_id, transaction_id,
                'rules_based',  -- Convert all to 'rules_based'
                CASE WHEN risk_score IS NULL THEN 0.5 ELSE risk_score / 100.0 END,
                confidence,
                description,
                feedback,
                feedback_notes,
                is_resolved,
                resolved_by,
                resolved_at,
                created_at,
                updated_at,
                notification_sent,
                notification_sent_at
            FROM anomalies_temp
        """)
    
    # Rename the metadata column back to config
    op.alter_column('tenants', 'metadata', new_column_name='config',
                   existing_type=postgresql.JSONB(astext_type=sa.Text()))
    
    # Drop the temporary table
    op.drop_table('anomalies_temp')
    
    # Drop new enum types
    op.execute('DROP TYPE IF EXISTS anomalytype')
    op.execute('DROP TYPE IF EXISTS feedbacktype')
    
    # Rename old enum types to standard names
    op.execute('ALTER TYPE anomalytype_old RENAME TO anomalytype')
    op.execute('ALTER TYPE feedbacktype_old RENAME TO feedbacktype') 