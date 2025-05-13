"""Test configuration for dashboard tests.

This module provides pytest fixtures for testing the dashboard.
"""

import asyncio
import os
import sys
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from auditpulse_mvp.database.models import Base

# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Create async engine for tests
test_engine = create_async_engine(TEST_DATABASE_URL, echo=True)
TestingSessionLocal = sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
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


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit functions for testing."""
    with patch("streamlit.session_state", {}) as mock_state, patch(
        "streamlit.sidebar"
    ) as mock_sidebar, patch("streamlit.columns") as mock_columns, patch(
        "streamlit.subheader"
    ) as mock_subheader, patch(
        "streamlit.write"
    ) as mock_write, patch(
        "streamlit.metric"
    ) as mock_metric, patch(
        "streamlit.plotly_chart"
    ) as mock_plotly_chart, patch(
        "streamlit.expander"
    ) as mock_expander, patch(
        "streamlit.button"
    ) as mock_button, patch(
        "streamlit.radio"
    ) as mock_radio, patch(
        "streamlit.text_area"
    ) as mock_text_area, patch(
        "streamlit.success"
    ) as mock_success, patch(
        "streamlit.error"
    ) as mock_error, patch(
        "streamlit.info"
    ) as mock_info:
        yield {
            "state": mock_state,
            "sidebar": mock_sidebar,
            "columns": mock_columns,
            "subheader": mock_subheader,
            "write": mock_write,
            "metric": mock_metric,
            "plotly_chart": mock_plotly_chart,
            "expander": mock_expander,
            "button": mock_button,
            "radio": mock_radio,
            "text_area": mock_text_area,
            "success": mock_success,
            "error": mock_error,
            "info": mock_info,
        }
