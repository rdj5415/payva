"""Test configuration for rules engine tests.

This module provides pytest fixtures for testing the rules engine.
"""

import asyncio
import os
import sys
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from auditpulse_mvp.database.base_class import Base

# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Create async engine for tests
test_engine = create_async_engine(TEST_DATABASE_URL, echo=True)
TestingSessionLocal = async_sessionmaker(
    bind=test_engine, class_=AsyncSession, expire_on_commit=False
)


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def setup_database() -> AsyncGenerator:
    """Set up test database."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session(setup_database) -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh database session for a test."""
    async with TestingSessionLocal() as session:
        yield session
        await session.rollback()
