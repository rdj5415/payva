"""Pytest configuration for the AuditPulse MVP project.

This module provides fixtures for testing.
"""

import asyncio
import os
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from auditpulse_mvp.database.base import Base
from auditpulse_mvp.utils.settings import settings


# Test database URL - ensure it's a string
TEST_DATABASE_URL = str(settings.DATABASE_TEST_URL or "sqlite+aiosqlite:///:memory:")


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for pytest-asyncio.

    Returns:
        asyncio.AbstractEventLoop: The event loop.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def db_engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create a SQLAlchemy engine for testing.

    Yields:
        AsyncEngine: The database engine.
    """
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        poolclass=NullPool,
    )

    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Drop all tables after tests
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Create a SQLAlchemy session for testing.

    Args:
        db_engine (AsyncEngine): The database engine.

    Yields:
        AsyncSession: The database session.
    """
    connection = await db_engine.connect()
    transaction = await connection.begin()

    async_session = async_sessionmaker(
        bind=connection, expire_on_commit=False, class_=AsyncSession
    )
    session = async_session()

    try:
        yield session
    finally:
        await session.close()
        await transaction.rollback()
        await connection.close()
