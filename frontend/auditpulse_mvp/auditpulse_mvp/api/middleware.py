"""API middleware for AuditPulse MVP.

This module defines middleware for the FastAPI application.
"""

import logging
import time
from typing import Callable, Dict, List, Optional, Set, Union
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from auditpulse_mvp.utils.settings import settings

logger = logging.getLogger(__name__)


class TenantMiddleware(BaseHTTPMiddleware):
    """Middleware for tenant isolation.

    This middleware enforces tenant isolation by checking for a tenant ID header
    and validating that the user has access to the requested tenant.
    """

    def __init__(
        self,
        app: ASGIApp,
        header_name: str = settings.TENANT_HEADER_NAME,
        exempt_paths: Optional[List[str]] = None,
    ):
        """Initialize the tenant middleware.

        Args:
            app: ASGI application
            header_name: Name of the header containing the tenant ID
            exempt_paths: List of paths exempt from tenant isolation
        """
        super().__init__(app)
        self.header_name = header_name
        self.exempt_paths = set(
            exempt_paths or ["/docs", "/redoc", "/openapi.json", "/api/v1/auth"]
        )

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process each request to enforce tenant isolation.

        Args:
            request: FastAPI request
            call_next: Next request handler

        Returns:
            Response: API response
        """
        # Skip tenant isolation for exempt paths
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)

        # Check if tenant isolation is enabled
        if not settings.ENABLE_TENANT_ISOLATION:
            return await call_next(request)

        # Check for tenant header if required
        if settings.REQUIRE_TENANT_HEADER:
            tenant_id = request.headers.get(self.header_name)

            if not tenant_id:
                # Log the error
                logger.warning(
                    f"No tenant ID provided in header '{self.header_name}' for path {request.url.path}"
                )

                # Return 400 Bad Request
                return Response(
                    content=f'{{"detail":"No tenant ID provided in header \'{self.header_name}\'"}}',
                    status_code=status.HTTP_400_BAD_REQUEST,
                    media_type="application/json",
                )

            # Validate tenant ID format
            try:
                UUID(tenant_id)
            except ValueError:
                # Log the error
                logger.warning(
                    f"Invalid tenant ID format in header '{self.header_name}': {tenant_id}"
                )

                # Return 400 Bad Request
                return Response(
                    content=f'{{"detail":"Invalid tenant ID format in header \'{self.header_name}\'"}}',
                    status_code=status.HTTP_400_BAD_REQUEST,
                    media_type="application/json",
                )

            # Store tenant ID in request state for use in dependencies
            request.state.tenant_id = tenant_id

        # Process the request
        response = await call_next(request)
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses.

    This middleware logs incoming requests and outgoing responses with timing information.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process each request to log timing information.

        Args:
            request: FastAPI request
            call_next: Next request handler

        Returns:
            Response: API response
        """
        # Start timer
        start_time = time.time()

        # Get request details
        method = request.method
        path = request.url.path
        query = request.url.query
        client_host = request.client.host if request.client else "unknown"

        # Log request
        logger.debug(f"Request: {method} {path}?{query} from {client_host}")

        # Process request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Log response
        logger.debug(
            f"Response: {method} {path} - Status: {response.status_code} - Time: {process_time:.3f}s"
        )

        # Add X-Process-Time header to response
        response.headers["X-Process-Time"] = f"{process_time:.3f}"

        return response


def setup_middlewares(app: FastAPI) -> None:
    """Configure middlewares for the FastAPI application.

    Args:
        app: FastAPI application instance
    """
    # Add CORS middleware
    if settings.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Add tenant middleware for multi-tenancy
    app.add_middleware(
        TenantMiddleware,
        header_name=settings.TENANT_HEADER_NAME,
        exempt_paths=[
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/auth/callback",
            "/api/v1/auth/settings",
            "/metrics",
            "/health",
        ],
    )

    # Add logging middleware
    app.add_middleware(LoggingMiddleware)
