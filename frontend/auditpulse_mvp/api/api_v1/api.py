"""API router, including all the API endpoints."""

from fastapi import APIRouter

from auditpulse_mvp.api.api_v1.endpoints import (
    anomalies,
    auth,
    config,
    tenants,
    transactions,
    users,
    learning,
)

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(tenants.router, prefix="/tenants", tags=["tenants"])
api_router.include_router(transactions.router, prefix="/transactions", tags=["transactions"])
api_router.include_router(anomalies.router, prefix="/anomalies", tags=["anomalies"])
api_router.include_router(config.router, prefix="/config", tags=["config"])
api_router.include_router(learning.router, prefix="/learning", tags=["learning"]) 