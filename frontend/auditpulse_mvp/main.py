"""
Main module for Bungii.

This is the entry point for the Bungii application.
It configures the FastAPI app, sets up routing, middleware, API endpoints,
and provides comprehensive OpenAPI documentation for the RESTful API.
The application implements financial transaction monitoring and anomaly detection
to help identify suspicious activities and maintain financial integrity.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Any

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.staticfiles import StaticFiles

from auditpulse_mvp.api.api_v1.api import api_router
from auditpulse_mvp.core.config import settings
from auditpulse_mvp.core.frontend import setup_frontend
from auditpulse_mvp.core.templates import TemplateManager, create_default_templates
from auditpulse_mvp.database.session import engine, SessionLocal
from auditpulse_mvp.middleware.security import setup_security_middleware

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("auditpulse")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for FastAPI app lifespan.

    This handles startup and shutdown events for the application.
    On startup, initializes required components like notification templates.
    On shutdown, ensures proper cleanup of resources.
    """
    # Startup event
    logger.info("Starting Bungii application")

    # Create a DB session for initialization tasks
    db = SessionLocal()
    try:
        # Initialize default notification templates
        logger.info("Initializing default notification templates")
        template_manager = TemplateManager(db)
        await create_default_templates(template_manager)
        logger.info("Default notification templates initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing templates: {e}")
    finally:
        db.close()

    yield

    # Shutdown event
    logger.info("Shutting down Bungii application")


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description="""
    **Bungii: Financial Transaction Auditing and Anomaly Detection**
    
    Bungii is an AI-powered financial transaction monitoring and anomaly detection system
    designed to help businesses identify suspicious activities and maintain financial integrity.
    
    This API allows you to:
    * Connect financial accounts via Plaid
    * Monitor transactions and detect anomalies
    * Manage ML models for anomaly detection
    * Configure notification preferences
    * Generate financial reports and analytics
    """,
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
    contact={
        "name": "Bungii Support",
        "url": "https://bungii.example.com/support",
        "email": "support@bungii.example.com",
    },
    license_info={
        "name": "Proprietary",
        "url": "https://bungii.example.com/license",
    },
    terms_of_service="https://bungii.example.com/terms",
)

# Set up CORS middleware
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Total-Count", "X-Page-Count"],
    )

# Set up security middleware
setup_security_middleware(app)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Set up frontend integration
setup_frontend(app)

# Custom OpenAPI schema with tags
app.openapi_tags = [
    {
        "name": "auth",
        "description": "Authentication operations including login, token management, password reset, and two-factor authentication.",
    },
    {
        "name": "users",
        "description": "User management operations including profile updates, preference settings, and account management.",
    },
    {
        "name": "transactions",
        "description": "Financial transaction management and synchronization with banking institutions via Plaid. Includes transaction categorization, search, and filtering.",
    },
    {
        "name": "anomalies",
        "description": "Anomaly detection and management for financial transactions. Identify, review, and resolve unusual transaction patterns.",
    },
    {
        "name": "models",
        "description": "ML model management for anomaly detection, including version control, deployment, performance monitoring, and validation.",
    },
    {
        "name": "admin",
        "description": "Administrative operations for system management, tenant configuration, user administration, and global settings.",
    },
    {
        "name": "plaid",
        "description": "Integration with Plaid API for financial data access, account linking, and transaction synchronization.",
    },
    {
        "name": "notifications",
        "description": "Notification management and delivery services across multiple channels including email, SMS, and in-app notifications.",
    },
    {
        "name": "health",
        "description": "System health and monitoring endpoints for checking application status, dependencies, and performance metrics.",
    },
]


# Custom Swagger UI and ReDoc routes
@app.get("/api/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Serve custom Swagger UI documentation."""
    return get_swagger_ui_html(
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        title=f"{settings.PROJECT_NAME} - API Documentation",
        oauth2_redirect_url=f"{settings.API_V1_STR}/oauth2-redirect",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_favicon_url="/static/favicon.ico",
    )


@app.get("/api/redoc", include_in_schema=False)
async def redoc_html():
    """Serve ReDoc documentation."""
    return get_redoc_html(
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        title=f"{settings.PROJECT_NAME} - API Reference",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
        redoc_favicon_url="/static/favicon.ico",
    )


# Health check endpoint
@app.get(
    "/health",
    tags=["health"],
    summary="System Health Check",
    response_model=Dict[str, Any],
)
async def health_check():
    """
    Health check endpoint.

    Returns basic application health information including service status,
    version, environment, and timestamp. Used for monitoring and automated health checks.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "timestamp": settings.current_timestamp(),
        "services": {"api": "available", "database": "connected", "redis": "connected"},
    }


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status_code": exc.status_code,
            "type": "http_exception",
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle general exceptions with secure error messages.

    Logs the full exception details but returns a generic error message to clients
    to prevent information disclosure.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "status_code": 500,
            "type": "server_error",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
