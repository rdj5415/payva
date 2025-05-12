"""Database migration for notification models.

Revision ID: 003
Revises: 002
Create Date: 2023-07-15 14:00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade():
    # Create notification_templates table
    op.create_table(
        'notification_templates',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('template_id', sa.String(100), nullable=False, unique=True, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('subject', sa.String(255), nullable=False),
        sa.Column('body', sa.Text(), nullable=False),
        sa.Column('html_body', sa.Text(), nullable=True),
        sa.Column('placeholders', sa.Text(), nullable=True),
        sa.Column('version', sa.Integer(), nullable=False, server_default=sa.text('1')),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
    )
    
    # Create notifications table
    op.create_table(
        'notifications',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('template_id', sa.String(100), sa.ForeignKey('notification_templates.template_id'), nullable=False),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('recipient', JSONB(), nullable=False),
        sa.Column('template_data', JSONB(), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, server_default=sa.text("'pending'"), index=True),
        sa.Column('priority', sa.String(20), nullable=False, server_default=sa.text("'medium'")),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('scheduled_at', sa.DateTime(), nullable=True),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
    )
    
    # Create notification_delivery_attempts table
    op.create_table(
        'notification_delivery_attempts',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('notification_id', UUID(as_uuid=True), sa.ForeignKey('notifications.id'), nullable=False),
        sa.Column('channel', sa.String(50), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, server_default=sa.text("'pending'")),
        sa.Column('response', JSONB(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('attempt_number', sa.Integer(), nullable=False, server_default=sa.text('1')),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
    )
    
    # Create indexes
    op.create_index('ix_notifications_user_id', 'notifications', ['user_id'])
    op.create_index('ix_notification_delivery_attempts_notification_id', 'notification_delivery_attempts', ['notification_id'])
    
    # Add notification_preferences column to users table
    op.add_column('users', sa.Column('notification_preferences', JSONB(), nullable=True))
    
    # Create function to update updated_at timestamp on update
    op.execute("""
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
       NEW.updated_at = now();
       RETURN NEW;
    END;
    $$ language 'plpgsql';
    """)
    
    # Create triggers for updated_at
    op.execute("""
    CREATE TRIGGER update_notification_templates_updated_at BEFORE UPDATE
    ON notification_templates FOR EACH ROW EXECUTE PROCEDURE 
    update_updated_at_column();
    """)
    
    op.execute("""
    CREATE TRIGGER update_notifications_updated_at BEFORE UPDATE
    ON notifications FOR EACH ROW EXECUTE PROCEDURE 
    update_updated_at_column();
    """)
    
    op.execute("""
    CREATE TRIGGER update_notification_delivery_attempts_updated_at BEFORE UPDATE
    ON notification_delivery_attempts FOR EACH ROW EXECUTE PROCEDURE 
    update_updated_at_column();
    """)


def downgrade():
    # Drop triggers
    op.execute("DROP TRIGGER IF EXISTS update_notification_templates_updated_at ON notification_templates")
    op.execute("DROP TRIGGER IF EXISTS update_notifications_updated_at ON notifications")
    op.execute("DROP TRIGGER IF EXISTS update_notification_delivery_attempts_updated_at ON notification_delivery_attempts")
    
    # Drop tables
    op.drop_table('notification_delivery_attempts')
    op.drop_table('notifications')
    op.drop_table('notification_templates')
    
    # Remove notification_preferences column from users table
    op.drop_column('users', 'notification_preferences') 