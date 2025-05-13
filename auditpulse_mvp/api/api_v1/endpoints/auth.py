"""Authentication API endpoints for AuditPulse MVP.

This module provides API endpoints for user authentication,
including registration, login, and OAuth2 integration.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2AuthorizationCodeBearer
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.api.deps import (
    get_current_user,
    require_admin,
    get_current_tenant,
    log_audit_action,
    AuditAction,
)
from auditpulse_mvp.database.models import User, Tenant
from auditpulse_mvp.database.session import get_db
from auditpulse_mvp.utils.settings import settings
from auditpulse_mvp.auth.auth_service import (
    AuthService,
    Token,
    UserCreate,
    UserResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class LoginRequest(BaseModel):
    """Login request model."""

    email: EmailStr
    password: str
    tenant_id: UUID


class Auth0CallbackRequest(BaseModel):
    """Auth0 callback request model."""

    code: str
    state: str


class TokenRefreshRequest(BaseModel):
    """Token refresh request model."""

    refresh_token: str


@router.post("/register", response_model=UserResponse)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Register a new user.

    Args:
        user_data: User creation data.
        db: Database session.

    Returns:
        Created user response.

    Raises:
        HTTPException: If registration fails.
    """
    # Initialize service
    service = AuthService(db)

    # Create user
    return service.create_user(user_data)


@router.post("/login", response_model=Token)
async def login(
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> Token:
    """Login a user.

    Args:
        login_data: Login request data.
        db: Database session.

    Returns:
        Access token.

    Raises:
        HTTPException: If login fails.
    """
    # Initialize service
    service = AuthService(db)

    # Authenticate user
    user, token = service.authenticate_user(
        email=login_data.email,
        password=login_data.password,
        tenant_id=login_data.tenant_id,
    )

    # Log successful login
    await log_audit_action(
        db=db,
        user_id=user.id,
        tenant_id=user.tenant_id,
        action=AuditAction.LOGIN,
        resource_type="user",
        resource_id=user.id,
    )

    return token


@router.post("/auth0/callback", response_model=Token)
async def auth0_callback(
    callback_data: Auth0CallbackRequest,
    db: AsyncSession = Depends(get_db),
) -> Token:
    """Handle Auth0 OAuth2 callback.

    Args:
        callback_data: Auth0 callback data.
        db: Database session.

    Returns:
        Access token.

    Raises:
        HTTPException: If callback handling fails.
    """
    try:
        # Initialize service
        service = AuthService(db)

        # Exchange code for token
        token_data = await service.exchange_auth0_code(
            code=callback_data.code,
            state=callback_data.state,
        )

        # Get or create user
        user = await service.get_or_create_auth0_user(token_data)

        # Create session token
        token = service.create_session_token(user)

        # Log successful login
        await log_audit_action(
            db=db,
            user_id=user.id,
            tenant_id=user.tenant_id,
            action=AuditAction.LOGIN,
            resource_type="user",
            resource_id=user.id,
            metadata={"provider": "auth0"},
        )

        return token

    except Exception as e:
        logger.error(f"Auth0 callback failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
        )


@router.post("/token/refresh", response_model=Token)
async def refresh_token(
    refresh_data: TokenRefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> Token:
    """Refresh an access token.

    Args:
        refresh_data: Token refresh data.
        db: Database session.

    Returns:
        New access token.

    Raises:
        HTTPException: If token refresh fails.
    """
    try:
        # Initialize service
        service = AuthService(db)

        # Refresh token
        token = await service.refresh_token(refresh_data.refresh_token)

        return token

    except Exception as e:
        logger.error(f"Token refresh failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )


@router.get("/settings", response_model=Dict[str, Any])
async def get_auth_settings() -> Dict[str, Any]:
    """Get authentication settings for the frontend.

    Returns:
        Dict[str, Any]: Authentication configuration
    """
    return {
        "auth0_domain": settings.AUTH0_DOMAIN,
        "auth0_client_id": settings.AUTH0_CLIENT_ID,
        "auth0_audience": settings.AUTH0_AUDIENCE,
        "login_providers": [
            "auth0" if settings.ENABLE_AUTH0_LOGIN else None,
            "email" if settings.ENABLE_EMAIL_LOGIN else None,
            "google" if settings.ENABLE_GOOGLE_LOGIN else None,
            "microsoft" if settings.ENABLE_MICROSOFT_LOGIN else None,
        ],
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    """Get current user information.

    Args:
        current_user: Current user.

    Returns:
        User response.
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        tenant_id=current_user.tenant_id,
        role=current_user.role,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
    )


@router.post("/change-password")
async def change_password(
    current_password: str,
    new_password: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Change user password.

    Args:
        current_password: Current password.
        new_password: New password.
        current_user: Current user.
        db: Database session.

    Returns:
        Success message.

    Raises:
        HTTPException: If password change fails.
    """
    # Initialize service
    service = AuthService(db)

    # Verify current password
    if not service._verify_password(current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=400,
            detail="Invalid current password",
        )

    # Update password
    current_user.hashed_password = service._hash_password(new_password)
    await db.commit()

    return {"message": "Password changed successfully"}
