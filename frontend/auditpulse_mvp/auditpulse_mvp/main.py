"""Main application module for AuditPulse MVP.

This module initializes the FastAPI application with all configurations.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from auditpulse_mvp.api.api_v1.api import api_router
from auditpulse_mvp.api.middleware import (
    TenantMiddleware,
    AuthMiddleware,
    AccessLogMiddleware,
    SecurityHeadersMiddleware,
)
from auditpulse_mvp.database.session import engine, SessionLocal
from auditpulse_mvp.utils.settings import settings
from auditpulse_mvp.ml_engine.scheduler import start_ml_scheduler, stop_ml_scheduler
from auditpulse_mvp.learning.scheduler import (
    start_feedback_learning_scheduler,
    stop_feedback_learning_scheduler,
)

# Configure logging based on settings
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("auditpulse_mvp")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager to handle startup and shutdown events.

    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("AuditPulse MVP is starting up...")

    # Initialize ML scheduler if enabled
    if settings.enable_ml_engine and settings.enable_ml_scheduler:
        from auditpulse_mvp.ml_engine.scheduler import (
            get_ml_scheduler,
            start_ml_scheduler,
        )

        logger.info("Initializing ML scheduler")
        scheduler = get_ml_scheduler()
        await start_ml_scheduler()

    # Initialize notification scheduler if enabled
    if settings.enable_notifications:
        from auditpulse_mvp.alerts.scheduler import start_notification_scheduler

        logger.info("Initializing notification scheduler")
        await start_notification_scheduler()

    # Initialize feedback learning scheduler if enabled
    if settings.enable_feedback_learning:
        await start_feedback_learning_scheduler()

    # Log enabled features
    logger.info(f"ML Engine enabled: {settings.enable_ml_engine}")
    logger.info(f"ML Scheduler enabled: {settings.enable_ml_scheduler}")
    logger.info(f"Risk Engine enabled: {settings.enable_risk_engine}")
    logger.info(f"GPT Explanations enabled: {settings.enable_gpt_explanations}")
    logger.info(f"Notifications enabled: {settings.enable_notifications}")
    logger.info(f"Tenant isolation enabled: {settings.ENABLE_TENANT_ISOLATION}")
    logger.info(f"Auth0 integration enabled: {settings.ENABLE_AUTH0_LOGIN}")

    # Yield control back to FastAPI
    yield

    # Shutdown
    logger.info("AuditPulse MVP is shutting down...")

    # Stop ML scheduler if it was started
    if settings.enable_ml_engine and settings.enable_ml_scheduler:
        from auditpulse_mvp.ml_engine.scheduler import stop_ml_scheduler

        logger.info("Stopping ML scheduler")
        await stop_ml_scheduler()

    # Stop notification scheduler if it was started
    if settings.enable_notifications:
        from auditpulse_mvp.alerts.scheduler import stop_notification_scheduler

        logger.info("Stopping notification scheduler")
        await stop_notification_scheduler()

    # Stop feedback learning scheduler if it was started
    if settings.enable_feedback_learning:
        await stop_feedback_learning_scheduler()


def create_application() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="AuditPulse MVP API - Financial anomaly detection platform",
        version="0.1.0",
        lifespan=lifespan,
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Set up middlewares (CORS, tenant isolation, logging)
    setup_middlewares(app)

    # Add API router
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    # Add global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle unhandled exceptions globally.

        Args:
            request: FastAPI request
            exc: Exception raised

        Returns:
            JSONResponse: JSON response with error details
        """
        logger.exception(f"Unhandled exception: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    # Add health check endpoint
    @app.get("/health", tags=["health"])
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint.

        Returns:
            Dict: Health status information
        """
        return {
            "status": "healthy",
            "version": "0.1.0",
            "features": {
                "ml_engine": settings.enable_ml_engine,
                "risk_engine": settings.enable_risk_engine,
                "gpt_explanations": settings.enable_gpt_explanations,
                "notifications": settings.enable_notifications,
                "tenant_isolation": settings.ENABLE_TENANT_ISOLATION,
                "auth0_enabled": settings.ENABLE_AUTH0_LOGIN,
                "feedback_learning": settings.enable_feedback_learning,
            },
        }

    return app


# Create application instance
app = create_application()


if __name__ == "__main__":
    """Run application with Uvicorn when script is executed directly."""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
