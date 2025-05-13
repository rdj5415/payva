#!/usr/bin/env python
"""Script to initialize the database with all tables."""

import asyncio
import argparse
import logging
import sys
import os

# Add parent directory to path to be able to import from auditpulse_mvp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alembic.config import Config
from alembic import command
from auditpulse_mvp.database.session import init_db
from auditpulse_mvp.utils.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


async def initialize_database():
    """Initialize the database by running all migrations."""
    logger.info("Initializing database")

    # Initialize database session
    await init_db()

    # Run Alembic migrations
    logger.info("Running database migrations")
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")

    logger.info("Database initialization complete")


def main():
    """Main function for database initialization."""
    parser = argparse.ArgumentParser(description="Initialize the database")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the database (drop and recreate all tables)",
    )
    args = parser.parse_args()

    if args.reset:
        logger.warning("--reset flag is set. This will drop and recreate all tables.")
        logger.warning("This operation cannot be undone.")
        confirmation = input("Are you sure you want to proceed? [y/N]: ")

        if confirmation.lower() != "y":
            logger.info("Operation aborted")
            return

        # Run Alembic downgrade to base
        logger.info("Dropping all tables")
        alembic_cfg = Config("alembic.ini")
        command.downgrade(alembic_cfg, "base")

    # Run initialization
    asyncio.run(initialize_database())

    logger.info("Done")


if __name__ == "__main__":
    main()
