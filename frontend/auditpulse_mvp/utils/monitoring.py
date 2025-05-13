"""Monitoring and health check functionality for AuditPulse MVP.

This module provides health check endpoints and system metrics collection
for monitoring the application's health and performance.
"""
import asyncio
import logging
import platform
import psutil
from datetime import datetime
from typing import Dict, Any, List

import httpx
from fastapi import APIRouter, Depends, HTTPException
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from auditpulse_mvp.database.session import get_db
from auditpulse_mvp.utils.settings import settings
from auditpulse_mvp.utils.logging import get_logger

# Create logger
logger = get_logger(__name__)

# Create router
router = APIRouter()


class HealthCheck:
    """Health check functionality."""
    
    def __init__(self):
        """Initialize health check."""
        self.start_time = datetime.now()
    
    async def check_database(self) -> Dict[str, Any]:
        """Check database health.
        
        Returns:
            Dict[str, Any]: Database health status.
        """
        try:
            async with get_db() as db:
                # Execute simple query
                await db.execute("SELECT 1")
                return {
                    "status": "healthy",
                    "latency_ms": 0,  # TODO: Measure actual latency
                }
        except Exception as e:
            logger.error("database_health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
            }
    
    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis health.
        
        Returns:
            Dict[str, Any]: Redis health status.
        """
        try:
            async with httpx.AsyncClient() as client:
                start_time = datetime.now()
                response = await client.get(f"{settings.REDIS_URL}/ping")
                latency = (datetime.now() - start_time).total_seconds() * 1000
                
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "latency_ms": latency,
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"Unexpected status code: {response.status_code}",
                    }
        except Exception as e:
            logger.error("redis_health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
            }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics.
        
        Returns:
            Dict[str, Any]: System metrics.
        """
        try:
            return {
                "cpu": {
                    "percent": psutil.cpu_percent(interval=1),
                    "count": psutil.cpu_count(),
                },
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent,
                },
                "disk": {
                    "total": psutil.disk_usage("/").total,
                    "free": psutil.disk_usage("/").free,
                    "percent": psutil.disk_usage("/").percent,
                },
                "system": {
                    "platform": platform.platform(),
                    "python_version": platform.python_version(),
                    "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                },
            }
        except Exception as e:
            logger.error("system_metrics_failed", error=str(e))
            return {
                "error": str(e),
            }


# Create health check instance
health_check = HealthCheck()


@router.get("/health")
async def health_check_endpoint() -> Dict[str, Any]:
    """Health check endpoint.
    
    Returns:
        Dict[str, Any]: Health check results.
    """
    # Run health checks in parallel
    db_health, redis_health = await asyncio.gather(
        health_check.check_database(),
        health_check.check_redis(),
    )
    
    # Get system metrics
    system_metrics = health_check.get_system_metrics()
    
    # Determine overall health
    is_healthy = all(
        check["status"] == "healthy"
        for check in [db_health, redis_health]
    )
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": db_health,
            "redis": redis_health,
        },
        "system": system_metrics,
    }


@router.get("/metrics")
async def metrics_endpoint() -> Response:
    """Prometheus metrics endpoint.
    
    Returns:
        Response: Prometheus metrics.
    """
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@router.get("/system")
async def system_metrics_endpoint() -> Dict[str, Any]:
    """System metrics endpoint.
    
    Returns:
        Dict[str, Any]: System metrics.
    """
    return health_check.get_system_metrics() 