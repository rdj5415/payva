"""Tests for the monitoring module."""
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from prometheus_client import generate_latest

from auditpulse_mvp.utils.monitoring import HealthCheck, router


@pytest.fixture
def app():
    """Create a test FastAPI application."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def health_check():
    """Create a HealthCheck instance."""
    return HealthCheck()


@pytest.mark.asyncio
async def test_health_check_endpoint(app):
    """Test the health check endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "services" in data
        assert "system" in data
        
        assert "database" in data["services"]
        assert "redis" in data["services"]


@pytest.mark.asyncio
async def test_metrics_endpoint(app):
    """Test the metrics endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/metrics")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
        
        # Verify metrics content
        content = response.text
        assert "http_requests_total" in content
        assert "http_request_duration_seconds" in content


@pytest.mark.asyncio
async def test_system_metrics_endpoint(app):
    """Test the system metrics endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/system")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "cpu" in data
        assert "memory" in data
        assert "disk" in data
        assert "system" in data
        
        # Verify CPU metrics
        assert "percent" in data["cpu"]
        assert "count" in data["cpu"]
        
        # Verify memory metrics
        assert "total" in data["memory"]
        assert "available" in data["memory"]
        assert "percent" in data["memory"]
        
        # Verify disk metrics
        assert "total" in data["disk"]
        assert "free" in data["disk"]
        assert "percent" in data["disk"]
        
        # Verify system metrics
        assert "platform" in data["system"]
        assert "python_version" in data["system"]
        assert "uptime_seconds" in data["system"]


@pytest.mark.asyncio
async def test_check_database(health_check):
    """Test database health check."""
    with patch("auditpulse_mvp.database.session.get_db") as mock_get_db:
        mock_session = AsyncMock()
        mock_get_db.return_value.__aenter__.return_value = mock_session
        
        result = await health_check.check_database()
        
        assert "status" in result
        assert "latency_ms" in result
        assert result["status"] == "healthy"


@pytest.mark.asyncio
async def test_check_redis(health_check):
    """Test Redis health check."""
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = AsyncMock(status_code=200)
        
        result = await health_check.check_redis()
        
        assert "status" in result
        assert "latency_ms" in result
        assert result["status"] == "healthy"


@pytest.mark.asyncio
async def test_check_redis_failure(health_check):
    """Test Redis health check failure."""
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.side_effect = Exception("Connection failed")
        
        result = await health_check.check_redis()
        
        assert "status" in result
        assert result["status"] == "unhealthy"
        assert "error" in result


def test_get_system_metrics(health_check):
    """Test system metrics collection."""
    metrics = health_check.get_system_metrics()
    
    assert "cpu" in metrics
    assert "memory" in metrics
    assert "disk" in metrics
    assert "system" in metrics
    
    # Verify metrics structure
    assert isinstance(metrics["cpu"]["percent"], float)
    assert isinstance(metrics["cpu"]["count"], int)
    assert isinstance(metrics["memory"]["total"], int)
    assert isinstance(metrics["memory"]["available"], int)
    assert isinstance(metrics["disk"]["total"], int)
    assert isinstance(metrics["disk"]["free"], int)
    assert isinstance(metrics["system"]["uptime_seconds"], float) 