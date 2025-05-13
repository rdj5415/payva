"""API Documentation for AuditPulse AI.

This module provides OpenAPI documentation and API specifications.
"""

from typing import Dict, Any
from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI

from auditpulse_mvp.main import app


def custom_openapi(app: FastAPI) -> Dict[str, Any]:
    """Generate custom OpenAPI schema with detailed documentation.

    Args:
        app: FastAPI application

    Returns:
        dict: OpenAPI schema
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="AuditPulse MVP API",
        version="1.0.0",
        description="""
        AuditPulse MVP API for financial transaction monitoring and anomaly detection.
        
        ## Authentication
        
        All endpoints require JWT authentication. Include the token in the Authorization header:
        ```
        Authorization: Bearer <token>
        ```
        
        ### OAuth2/SSO Providers
        
        The following OAuth2/SSO providers are supported:
        - Auth0 (default)
        - Google (enable with `ENABLE_GOOGLE_LOGIN=True` in settings)
        - Microsoft (enable with `ENABLE_MICROSOFT_LOGIN=True` in settings)
        
        The login page will display all enabled providers. OAuth2 flows are fully supported.
        
        ## Model Management
        
        The API provides endpoints for managing ML model versions and tracking performance:
        
        ### Model Versions
        
        - `POST /api/v1/models/versions`: Create a new model version
        - `GET /api/v1/models/versions/{model_type}`: List model versions
        - `GET /api/v1/models/versions/{model_type}/active`: Get active version
        - `POST /api/v1/models/versions/{model_type}/{version}/activate`: Activate version
        - `POST /api/v1/models/versions/{model_type}/{version}/rollback`: Rollback to version
        
        ### Performance Tracking
        
        - `POST /api/v1/models/performance`: Record performance metrics
        - `GET /api/v1/models/performance/{model_type}`: Get performance history
        - `GET /api/v1/models/performance/{model_type}/summary`: Get performance summary
        
        Model versions are tracked with semantic versioning (MAJOR.MINOR.PATCH). Only one version
        can be active at a time. Performance metrics are recorded for each evaluation run.
        """,
        routes=app.routes,
    )

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token for API authentication",
        },
        "oauth2": {
            "type": "oauth2",
            "flows": {
                "authorizationCode": {
                    "authorizationUrl": "/auth/oauth2/authorize",
                    "tokenUrl": "/auth/oauth2/token",
                    "scopes": {
                        "read": "Read access",
                        "write": "Write access",
                        "admin": "Admin access",
                    },
                }
            },
        },
    }

    # Add global security requirement
    openapi_schema["security"] = [{"bearerAuth": []}]

    # Add detailed response schemas
    openapi_schema["components"]["schemas"].update(
        {
            "Error": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Error code"},
                    "message": {"type": "string", "description": "Error message"},
                    "details": {
                        "type": "object",
                        "description": "Additional error details",
                    },
                },
            },
            "Pagination": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer", "description": "Current page number"},
                    "size": {"type": "integer", "description": "Page size"},
                    "total": {
                        "type": "integer",
                        "description": "Total number of items",
                    },
                    "pages": {
                        "type": "integer",
                        "description": "Total number of pages",
                    },
                },
            },
        }
    )

    # Add detailed endpoint documentation
    for path in openapi_schema["paths"].values():
        for operation in path.values():
            if "responses" in operation:
                operation["responses"].update(
                    {
                        "400": {
                            "description": "Bad Request",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            },
                        },
                        "401": {
                            "description": "Unauthorized",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            },
                        },
                        "403": {
                            "description": "Forbidden",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            },
                        },
                        "404": {
                            "description": "Not Found",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            },
                        },
                        "429": {
                            "description": "Too Many Requests",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            },
                        },
                        "500": {
                            "description": "Internal Server Error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            },
                        },
                    }
                )

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Set custom OpenAPI schema
app.openapi = custom_openapi
