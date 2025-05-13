"""Pytest configuration for the Risk Engine tests.

This module provides fixtures for testing the Risk Engine.
"""

import asyncio
import os
import sys
from typing import AsyncGenerator, Generator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import from auditpulse_mvp
from auditpulse_mvp.database.base import Base
from auditpulse_mvp.utils.settings import get_settings

# Get settings for tests
settings = get_settings()

# Create test engine
test_engine = create_async_engine(
    settings.DATABASE_TEST_URL or "sqlite+aiosqlite:///:memory:",
    echo=False,
)

# Create async session factory
TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=test_engine,
    class_=AsyncSession,
)


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an event loop for tests.

    Returns:
        Generator: The event loop.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def setup_database() -> None:
    """Set up the test database.

    Returns:
        None
    """
    # Create all tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Clean up after tests
    yield

    # Drop all tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture(scope="function")
async def db_session(setup_database) -> AsyncGenerator[AsyncSession, None]:
    """Get a database session for testing.

    Args:
        setup_database: Fixture to set up the database.

    Returns:
        AsyncGenerator[AsyncSession, None]: The database session.
    """
    # Create a new session for each test
    async with TestingSessionLocal() as session:
        # Begin a nested transaction
        async with session.begin():
            # Use a nested transaction for tests
            yield session

            # Rollback at the end of each test
            await session.rollback()
