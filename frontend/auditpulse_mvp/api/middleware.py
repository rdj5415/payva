from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from auditpulse_mvp.api.middleware import SecurityHeadersMiddleware


def setup_middlewares(app: FastAPI) -> None:
    """Configure middlewares for the FastAPI application."""
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
    # Add security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)
    # Add logging middleware
    app.add_middleware(LoggingMiddleware)
    # Add rate limiting middleware
    limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(
        RateLimitExceeded,
        lambda request, exc: JSONResponse(
            status_code=429, content={"detail": "Rate limit exceeded"}
        ),
    )
    app.middleware("http")(limiter.middleware)
