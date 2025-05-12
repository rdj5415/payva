"""Authentication API endpoints.

This module provides API endpoints for user authentication and authorization.
"""
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, Any, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.security import OAuth2AuthorizationCodeBearer
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.api.deps import (
    get_current_user, require_admin, get_current_tenant, log_audit_action, AuditAction
)
from auditpulse_mvp.database.models import User, Tenant
from auditpulse_mvp.database.session import get_db
from auditpulse_mvp.utils.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


class UserRole(str, Enum):
    """User roles in the system."""
    ADMIN = "admin"
    AUDITOR = "auditor"
    VIEWER = "viewer"


class LoginProvider(str, Enum):
    """Authentication providers."""
    AUTH0 = "auth0"
    EMAIL = "email"
    GOOGLE = "google"
    MICROSOFT = "microsoft"


class AuthSettings(BaseModel):
    """Authentication settings."""
    auth0_domain: str = Field(settings.AUTH0_DOMAIN, description="Auth0 domain")
    auth0_client_id: str = Field(settings.AUTH0_CLIENT_ID, description="Auth0 client ID")
    auth0_audience: str = Field(settings.AUTH0_AUDIENCE, description="Auth0 API audience")
    login_providers: List[LoginProvider] = Field(
        ["auth0", "email"],
        description="Available login providers"
    )


class UserProfile(BaseModel):
    """User profile information."""
    id: UUID
    email: EmailStr
    full_name: str
    role: UserRole
    tenant_id: UUID
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    email_notifications: bool = True
    slack_notifications: bool = False
    sms_notifications: bool = False
    phone_number: Optional[str] = None
    slack_user_id: Optional[str] = None
    permissions: List[str] = []


class UserUpdate(BaseModel):
    """User profile update model."""
    full_name: Optional[str] = None
    email_notifications: Optional[bool] = None
    slack_notifications: Optional[bool] = None
    sms_notifications: Optional[bool] = None
    phone_number: Optional[str] = None
    slack_user_id: Optional[str] = None


@router.get(
    "/settings",
    response_model=AuthSettings,
    summary="Get authentication settings",
)
async def get_auth_settings() -> AuthSettings:
    """Get authentication settings for the frontend.
    
    Returns:
        AuthSettings: Authentication configuration
    """
    return AuthSettings(
        auth0_domain=settings.AUTH0_DOMAIN,
        auth0_client_id=settings.AUTH0_CLIENT_ID,
        auth0_audience=settings.AUTH0_AUDIENCE,
        login_providers=[
            provider for provider in LoginProvider 
            if getattr(settings, f"ENABLE_{provider.upper()}_LOGIN", True)
        ]
    )


@router.get(
    "/me",
    response_model=UserProfile,
    summary="Get current user profile",
)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
) -> UserProfile:
    """Get the current authenticated user's profile.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        UserProfile: User profile information
    """
    # Determine permissions based on role
    permissions = []
    if current_user.role == UserRole.ADMIN.value:
        permissions = [
            "read:anomalies", "write:anomalies", 
            "read:transactions", "write:transactions",
            "read:users", "write:users",
            "read:settings", "write:settings",
            "read:tenants", "write:tenants",
            "trigger:sync"
        ]
    elif current_user.role == UserRole.AUDITOR.value:
        permissions = [
            "read:anomalies", "write:anomalies", 
            "read:transactions",
            "read:users",
            "read:settings",
            "trigger:sync"
        ]
    elif current_user.role == UserRole.VIEWER.value:
        permissions = [
            "read:anomalies",
            "read:transactions",
            "read:settings"
        ]
    
    return UserProfile(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        tenant_id=current_user.tenant_id,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at,
        last_login=current_user.last_login,
        email_notifications=current_user.email_notifications,
        slack_notifications=current_user.slack_notifications,
        sms_notifications=current_user.sms_notifications,
        phone_number=current_user.phone_number,
        slack_user_id=current_user.slack_user_id,
        permissions=permissions,
    )


@router.patch(
    "/me",
    response_model=UserProfile,
    summary="Update current user profile",
)
async def update_current_user_profile(
    updates: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> UserProfile:
    """Update the current authenticated user's profile.
    
    Args:
        updates: User profile updates
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        UserProfile: Updated user profile
    """
    # Build update dictionary with only provided fields
    update_data = updates.dict(exclude_unset=True)
    
    if update_data:
        # Add updated_at timestamp
        update_data["updated_at"] = datetime.utcnow()
        
        # Execute update
        stmt = (
            update(User)
            .where(User.id == current_user.id)
            .values(**update_data)
        )
        await db.execute(stmt)
        await db.commit()
        
        # Refresh user object from database
        await db.refresh(current_user)
        
        # Log the change
        await log_audit_action(
            db=db,
            action=AuditAction(
                action="update_user_profile",
                user_id=current_user.id,
                tenant_id=current_user.tenant_id,
                resource_type="user",
                resource_id=current_user.id,
                details={"updated_fields": list(update_data.keys())},
            ),
        )
    
    # Return updated profile
    return await get_current_user_profile(current_user=current_user)


@router.post(
    "/callback",
    summary="Auth0 callback handler",
)
async def auth0_callback(
    request: Request,
    code: str = Query(..., description="Authorization code from Auth0"),
    state: str = Query(..., description="State from Auth0"),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Handle Auth0 callback after user authentication.
    
    Args:
        request: Request object
        code: Authorization code from Auth0
        state: State parameter from Auth0
        db: Database session
        
    Returns:
        Dict: User information and tokens
    """
    # In a real implementation, this would exchange the code for tokens
    # and create or update the user in the database based on the Auth0 profile
    
    # For the MVP, we'll just log the callback
    logger.info(f"Auth0 callback received with code={code}, state={state}")
    
    # Return a mock response
    return {
        "success": True,
        "message": "Auth0 callback processed",
        "next": "/dashboard"
    } 