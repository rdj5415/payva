"""Database session management for AuditPulse MVP.

This module provides functionality for managing database sessions,
including session creation, dependency injection, and context management.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from auditpulse_mvp.utils.settings import settings

# Create database engine
engine = create_async_engine(
    str(settings.DATABASE_URL),  # Convert URL to string to ensure compatibility
    echo=settings.ENVIRONMENT == "development",
    future=True,
)

# Create session factory using async_sessionmaker
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session.

    Yields:
        AsyncSession: Database session.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
