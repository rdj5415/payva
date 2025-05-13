"""
Frontend integration for AuditPulse MVP.

This module provides routes for serving the admin dashboard and static files,
allowing the frontend to be integrated with the FastAPI backend.
"""

from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path

from auditpulse_mvp.core.security import get_current_active_user
from auditpulse_mvp.schemas.user import User
from auditpulse_mvp.core.config import settings

# Create the router
router = APIRouter()

# Define the template directory
frontend_dir = Path("frontend")
templates_dir = frontend_dir / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Define the base URL for static files
STATIC_URL = "/static"

def is_admin(current_user: User = Depends(get_current_active_user)) -> User:
    """Check if the current user is an admin."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    return current_user

@router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, current_user: User = Depends(is_admin)):
    """
    Serve the admin dashboard HTML.
    
    This endpoint serves the main admin dashboard HTML page. Only accessible to admin users.
    """
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "user": current_user}
    )

@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve the favicon."""
    return FileResponse("static/favicon.ico")

def setup_frontend(app):
    """
    Set up the frontend routes and static files.
    
    This function should be called from the main application startup
    to mount the static files and register the frontend routes.
    """
    # Mount the static directory to serve static files
    static_dir = Path("frontend") / "static"
    if static_dir.exists():
        app.mount(STATIC_URL, StaticFiles(directory=str(static_dir)), name="frontend_static")
    
    # Mount the root static directory for favicon and other root assets
    root_static_dir = Path("static")
    if root_static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(root_static_dir)), name="static")
    
    # Include the frontend router
    app.include_router(router, tags=["frontend"]) 