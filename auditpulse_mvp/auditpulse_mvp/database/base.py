"""Base database configuration for SQLAlchemy.

This module provides the base declarative class, async engine, and session management.
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from auditpulse_mvp.utils.settings import settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    pass


# Create async engine with the provided database URL
engine = create_async_engine(
    str(settings.DATABASE_URL),
    echo=False,
    pool_pre_ping=True,
    pool_recycle=300,
)

# Create async session factory
async_session_factory = async_sessionmaker(
    engine, expire_on_commit=False, autoflush=False
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a new database session and ensure it's closed when done.

    Yields:
        AsyncSession: The database session.
    """
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
