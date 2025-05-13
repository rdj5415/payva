"""Authentication and authorization dependencies for the API.

This module provides FastAPI dependency functions for authentication and authorization.
"""
import logging
from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2AuthorizationCodeBearer, SecurityScopes
from jose import JWTError, jwt
from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.models import User, Tenant
from auditpulse_mvp.database.session import get_db
from auditpulse_mvp.utils.settings import settings

logger = logging.getLogger(__name__)

# Configure OAuth2 with Auth0 integration
oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=f"{settings.AUTH0_DOMAIN}/authorize",
    tokenUrl=f"{settings.AUTH0_DOMAIN}/oauth/token",
    scopes={
        "openid": "OpenID Connect",
        "profile": "User profile",
        "email": "User email",
    },
    auto_error=True,
)


class TokenData(BaseModel):
    """Model for decoded token data."""
    sub: str  # Subject (user ID)
    scope: Optional[str] = None
    tenant_id: Optional[str] = None
    permissions: Optional[list[str]] = None
    exp: Optional[int] = None  # Expiration time


class AuditAction(BaseModel):
    """Model for audit log entries."""
    action: str
    user_id: UUID
    tenant_id: UUID
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    details: Optional[dict] = None


async def get_token_data(
    security_scopes: SecurityScopes, 
    token: str = Depends(oauth2_scheme)
) -> TokenData:
    """Decode and validate JWT token.
    
    Args:
        security_scopes: Security scopes required for the endpoint
        token: JWT token from the Authorization header
        
    Returns:
        TokenData: Decoded token data
        
    Raises:
        HTTPException: If token is invalid or does not have required scopes
    """
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{" ".join(security_scopes.scopes)}"'
    else:
        authenticate_value = "Bearer"
        
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )
    
    try:
        # Decode JWT token
        payload = jwt.decode(
            token, 
            settings.AUTH0_CLIENT_SECRET.get_secret_value(), 
            algorithms=[settings.ALGORITHM],
            audience=settings.AUTH0_API_AUDIENCE,
            issuer=settings.AUTH0_DOMAIN,
        )
        
        # Extract user ID (subject)
        sub: str = payload.get("sub")
        if not sub:
            raise credentials_exception
            
        # Extract scopes
        scope = payload.get("scope", "")
        
        # Extract tenant ID (custom claim)
        tenant_id = payload.get("https://auditpulse.ai/tenant_id")
        
        # Extract permissions
        permissions = payload.get("permissions", [])
        
        # Create token data
        token_data = TokenData(
            sub=sub,
            scope=scope,
            tenant_id=tenant_id,
            permissions=permissions,
            exp=payload.get("exp"),
        )
        
        # Check if token has required scopes
        if security_scopes.scopes:
            token_scopes = scope.split() if scope else []
            for scope in security_scopes.scopes:
                if scope not in token_scopes:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Not enough permissions. Required scope: {scope}",
                        headers={"WWW-Authenticate": authenticate_value},
                    )
        
        return token_data
        
    except (JWTError, ValidationError) as e:
        logger.error(f"Token validation error: {str(e)}")
        raise credentials_exception


async def get_current_user(
    db: AsyncSession = Depends(get_db),
    token_data: TokenData = Depends(get_token_data),
) -> User:
    """Get the current authenticated user.
    
    Args:
        db: Database session
        token_data: Decoded token data
        
    Returns:
        User: Current authenticated user
        
    Raises:
        HTTPException: If user not found or inactive
    """
    # Get user from database
    auth0_id = token_data.sub
    
    # Find user by Auth0 ID (sub)
    stmt = select(User).where(
        User.auth0_id == auth0_id,
        User.is_active == True,
    )
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found or inactive",
        )
    
    # Check if tenant ID matches
    if token_data.tenant_id and str(user.tenant_id) != token_data.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant mismatch",
        )
    
    return user


async def get_current_tenant(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Tenant:
    """Get the current tenant from the authenticated user.
    
    Args:
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Tenant: Current tenant
        
    Raises:
        HTTPException: If tenant not found or inactive
    """
    # Get tenant from database
    stmt = select(Tenant).where(
        Tenant.id == current_user.tenant_id,
        Tenant.is_active == True,
    )
    result = await db.execute(stmt)
    tenant = result.scalar_one_or_none()
    
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found or inactive",
        )
    
    return tenant


async def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require user to have admin role.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User: Current authenticated user with admin role
        
    Raises:
        HTTPException: If user does not have admin role
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions. Admin role required.",
        )
    
    return current_user


async def require_auditor(current_user: User = Depends(get_current_user)) -> User:
    """Require user to have admin or auditor role.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User: Current authenticated user with admin or auditor role
        
    Raises:
        HTTPException: If user does not have admin or auditor role
    """
    if current_user.role not in ["admin", "auditor"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions. Admin or auditor role required.",
        )
    
    return current_user


async def log_audit_action(
    db: AsyncSession,
    action: AuditAction,
) -> None:
    """Log an audit action in the database.
    
    Args:
        db: Database session
        action: Audit action to log
    """
    # In a real implementation, this would store the audit action in the database
    # For now, we'll just log it
    logger.info(
        f"AUDIT: {action.action} - User {action.user_id} - "
        f"Tenant {action.tenant_id} - {action.resource_type} {action.resource_id}"
    )
    
    # Add to audit log table
    # from auditpulse_mvp.database.models import AuditLog
    # audit_log = AuditLog(
    #     action=action.action,
    #     user_id=action.user_id,
    #     tenant_id=action.tenant_id,
    #     resource_type=action.resource_type,
    #     resource_id=action.resource_id,
    #     details=action.details,
    # )
    # db.add(audit_log)
    # await db.commit() 