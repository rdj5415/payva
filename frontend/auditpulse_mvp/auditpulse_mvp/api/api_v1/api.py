"""Main API router for AuditPulse MVP application.

This module sets up the main API router and includes all endpoint routers.
"""

from fastapi import APIRouter

from auditpulse_mvp.api.api_v1.endpoints import (
    anomalies,
    auth,
    dashboard,
    sync,
    tenants,
    transactions,
    config,
    learning,
)
from auditpulse_mvp.ml_engine.api.api import router as ml_router
from auditpulse_mvp.risk_engine.api.api import router as risk_router
from auditpulse_mvp.alerts.api import router as alerts_router

# Import other API routers here

# Create API router for v1
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(tenants.router, prefix="/tenants", tags=["tenants"])
api_router.include_router(
    transactions.router, prefix="/transactions", tags=["transactions"]
)
api_router.include_router(anomalies.router, prefix="/anomalies", tags=["anomalies"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
api_router.include_router(sync.router, prefix="/sync", tags=["sync"])
api_router.include_router(config.router, prefix="/config", tags=["config"])
api_router.include_router(learning.router, prefix="/learning", tags=["learning"])

# Include ML engine API
api_router.include_router(ml_router, prefix="/ml", tags=["ml"])

# Include Risk engine API
api_router.include_router(risk_router, prefix="/risk", tags=["risk"])

# Include Alert system API
api_router.include_router(alerts_router, prefix="/alerts", tags=["alerts"])
# Include other routers here
