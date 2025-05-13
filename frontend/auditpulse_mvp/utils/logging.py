"""Logging and metrics configuration for AuditPulse MVP.

This module provides structured logging and metrics collection functionality,
integrating with Prometheus for metrics and JSON logging for better observability.
"""

import json
import logging
import time
from contextvars import ContextVar
from typing import Any, Dict, Optional
from uuid import UUID

import structlog
from prometheus_client import Counter, Histogram, Gauge
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from auditpulse_mvp.utils.settings import settings

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Create context variables for request tracking
request_id: ContextVar[str] = ContextVar("request_id", default="")
user_id: ContextVar[Optional[UUID]] = ContextVar("user_id", default=None)
tenant_id: ContextVar[Optional[UUID]] = ContextVar("tenant_id", default=None)

# Define Prometheus metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

active_users = Gauge(
    "active_users",
    "Number of active users",
    ["tenant_id"],
)

auth_attempts = Counter(
    "auth_attempts_total",
    "Total number of authentication attempts",
    ["method", "status"],
)

api_errors = Counter(
    "api_errors_total",
    "Total number of API errors",
    ["endpoint", "error_type"],
)

# Create logger
logger = structlog.get_logger()


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging and metrics collection."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Process the request and log metrics.

        Args:
            request: The incoming request.
            call_next: The next middleware or route handler.

        Returns:
            Response: The response from the application.
        """
        # Generate request ID
        req_id = request.headers.get("X-Request-ID", "")
        request_id.set(req_id)

        # Start timer
        start_time = time.time()

        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise e
        finally:
            # Calculate duration
            duration = time.time() - start_time

            # Log request
            logger.info(
                "http_request",
                request_id=req_id,
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration=duration,
                user_id=user_id.get(),
                tenant_id=tenant_id.get(),
            )

            # Record metrics
            http_requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status=status_code,
            ).inc()

            http_request_duration_seconds.labels(
                method=request.method,
                endpoint=request.url.path,
            ).observe(duration)

        return response


def log_auth_attempt(
    method: str, success: bool, user_id: Optional[UUID] = None
) -> None:
    """Log an authentication attempt.

    Args:
        method: Authentication method used.
        success: Whether authentication was successful.
        user_id: ID of the user attempting to authenticate.
    """
    logger.info(
        "auth_attempt",
        method=method,
        success=success,
        user_id=user_id,
    )

    auth_attempts.labels(
        method=method,
        status="success" if success else "failure",
    ).inc()


def log_api_error(
    endpoint: str,
    error_type: str,
    error_message: str,
    user_id: Optional[UUID] = None,
    tenant_id: Optional[UUID] = None,
) -> None:
    """Log an API error.

    Args:
        endpoint: The API endpoint where the error occurred.
        error_type: Type of error (e.g., "validation", "auth", "server").
        error_message: Error message.
        user_id: ID of the user who encountered the error.
        tenant_id: ID of the tenant where the error occurred.
    """
    logger.error(
        "api_error",
        endpoint=endpoint,
        error_type=error_type,
        error_message=error_message,
        user_id=user_id,
        tenant_id=tenant_id,
    )

    api_errors.labels(
        endpoint=endpoint,
        error_type=error_type,
    ).inc()


def update_active_users(tenant_id: UUID, count: int) -> None:
    """Update the number of active users for a tenant.

    Args:
        tenant_id: ID of the tenant.
        count: Number of active users.
    """
    active_users.labels(tenant_id=str(tenant_id)).set(count)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger instance with the given name.

    Args:
        name: Name of the logger.

    Returns:
        structlog.BoundLogger: Configured logger instance.
    """
    return structlog.get_logger(name)
