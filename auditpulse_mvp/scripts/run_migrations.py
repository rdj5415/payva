"""Script to run database migrations.

This script runs Alembic migrations to update the database schema.
It should be run before starting the application.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from alembic.config import Config
from alembic import command


def run_migrations() -> None:
    """Run database migrations."""
    # Get database URL from environment variable
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")

    # Create Alembic configuration
    alembic_cfg = Config()
    alembic_cfg.set_main_option("script_location", "alembic")
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)

    # Run migrations
    print("Running database migrations...")
    command.upgrade(alembic_cfg, "head")
    print("Database migrations completed successfully")


if __name__ == "__main__":
    run_migrations() 