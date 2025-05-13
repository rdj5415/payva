"""Authentication service for AuditPulse MVP.

This module provides functionality for user authentication and authorization,
including JWT token management, password hashing, and user session handling.
"""
import datetime
import logging
from typing import Optional, Tuple, Dict, Any
from uuid import UUID

import httpx
from fastapi import HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, UUID4
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from auditpulse_mvp.database.models import User, UserRole
from auditpulse_mvp.utils.settings import settings
from auditpulse_mvp.utils.logging import get_logger, log_auth_attempt, log_api_error

# Configure logging
logger = get_logger(__name__)

# Configure password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Configure OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class Token(BaseModel):
    """Token model for authentication."""
    access_token: str
    token_type: str
    expires_at: datetime.datetime
    refresh_token: Optional[str] = None


class TokenData(BaseModel):
    """Model for decoded token data."""
    user_id: UUID
    tenant_id: UUID
    role: UserRole
    exp: Optional[int] = None


class UserCreate(BaseModel):
    """User creation model."""
    email: EmailStr
    password: str
    full_name: str
    tenant_id: UUID4
    role: UserRole = UserRole.USER


class UserResponse(BaseModel):
    """User response model."""
    id: UUID4
    email: EmailStr
    full_name: str
    tenant_id: UUID4
    role: UserRole
    is_active: bool
    created_at: datetime.datetime


class AuthService:
    """Service for handling user authentication and authorization."""
    
    def __init__(self, db_session: AsyncSession):
        """Initialize the authentication service.
        
        Args:
            db_session: Database session.
        """
        self.db_session = db_session
    
    async def exchange_auth0_code(self, code: str, state: str) -> Dict[str, Any]:
        """Exchange Auth0 authorization code for tokens.
        
        Args:
            code: Authorization code from Auth0.
            state: State parameter for CSRF protection.
            
        Returns:
            Dict[str, Any]: Token data from Auth0.
            
        Raises:
            HTTPException: If token exchange fails.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.AUTH0_DOMAIN}/oauth/token",
                    data={
                        "grant_type": "authorization_code",
                        "client_id": settings.AUTH0_CLIENT_ID,
                        "client_secret": settings.AUTH0_CLIENT_SECRET.get_secret_value(),
                        "code": code,
                        "redirect_uri": settings.AUTH0_CALLBACK_URL,
                    },
                )
                
                if response.status_code != 200:
                    log_api_error(
                        endpoint="/auth0/callback",
                        error_type="auth",
                        error_message="Failed to exchange code for token",
                    )
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Failed to exchange code for token",
                    )
                
                return response.json()
                
        except Exception as e:
            logger.error("auth0_token_exchange_failed", error=str(e))
            log_api_error(
                endpoint="/auth0/callback",
                error_type="auth",
                error_message=str(e),
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed",
            )
    
    async def get_or_create_auth0_user(self, token_data: Dict[str, Any]) -> User:
        """Get or create a user from Auth0 token data.
        
        Args:
            token_data: Token data from Auth0.
            
        Returns:
            User: The user object.
            
        Raises:
            HTTPException: If user creation fails.
        """
        try:
            # Get user info from Auth0
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{settings.AUTH0_DOMAIN}/userinfo",
                    headers={"Authorization": f"Bearer {token_data['access_token']}"},
                )
                
                if response.status_code != 200:
                    log_api_error(
                        endpoint="/auth0/userinfo",
                        error_type="auth",
                        error_message="Failed to get user info",
                    )
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Failed to get user info",
                    )
                
                user_info = response.json()
            
            # Check if user exists
            stmt = select(User).where(User.email == user_info["email"])
            result = await self.db_session.execute(stmt)
            user = result.scalar_one_or_none()
            
            if user:
                # Update user info
                user.full_name = user_info.get("name", user.full_name)
                user.picture = user_info.get("picture")
                await self.db_session.commit()
                log_auth_attempt("auth0", True, user.id)
                return user
            
            # Create new user
            user = User(
                email=user_info["email"],
                full_name=user_info.get("name", ""),
                picture=user_info.get("picture"),
                role=UserRole.VIEWER,  # Default role
                is_active=True,
                created_at=datetime.datetime.now(),
            )
            
            self.db_session.add(user)
            await self.db_session.commit()
            await self.db_session.refresh(user)
            
            log_auth_attempt("auth0", True, user.id)
            return user
            
        except Exception as e:
            logger.error("auth0_user_creation_failed", error=str(e))
            log_api_error(
                endpoint="/auth0/user",
                error_type="auth",
                error_message=str(e),
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user",
            )
    
    async def refresh_token(self, refresh_token: str) -> Token:
        """Refresh an access token.
        
        Args:
            refresh_token: The refresh token.
            
        Returns:
            Token: New access token.
            
        Raises:
            HTTPException: If token refresh fails.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.AUTH0_DOMAIN}/oauth/token",
                    data={
                        "grant_type": "refresh_token",
                        "client_id": settings.AUTH0_CLIENT_ID,
                        "client_secret": settings.AUTH0_CLIENT_SECRET.get_secret_value(),
                        "refresh_token": refresh_token,
                    },
                )
                
                if response.status_code != 200:
                    log_api_error(
                        endpoint="/token/refresh",
                        error_type="auth",
                        error_message="Invalid refresh token",
                    )
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid refresh token",
                    )
                
                token_data = response.json()
                
                return Token(
                    access_token=token_data["access_token"],
                    token_type="bearer",
                    expires_at=datetime.datetime.now() + datetime.timedelta(
                        seconds=token_data["expires_in"]
                    ),
                    refresh_token=token_data.get("refresh_token"),
                )
                
        except Exception as e:
            logger.error("token_refresh_failed", error=str(e))
            log_api_error(
                endpoint="/token/refresh",
                error_type="auth",
                error_message=str(e),
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )
    
    def create_session_token(self, user: User) -> Token:
        """Create a session token for a user.
        
        Args:
            user: The user to create token for.
            
        Returns:
            Token: The session token.
        """
        # Set expiration
        expires_at = datetime.datetime.now() + datetime.timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        )
        
        # Create token data
        token_data = TokenData(
            user_id=user.id,
            tenant_id=user.tenant_id,
            role=user.role,
            exp=int(expires_at.timestamp()),
        )
        
        # Create token
        access_token = jwt.encode(
            token_data.dict(),
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_at=expires_at,
        )
    
    def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create a new user.
        
        Args:
            user_data: User creation data.
            
        Returns:
            Created user response.
            
        Raises:
            HTTPException: If user creation fails.
        """
        try:
            # Check if user exists
            existing_user = self.db_session.query(User).filter(
                User.email == user_data.email,
                User.tenant_id == user_data.tenant_id,
            ).first()
            
            if existing_user:
                log_api_error(
                    endpoint="/register",
                    error_type="validation",
                    error_message="User already exists",
                )
                raise HTTPException(
                    status_code=400,
                    detail="User already exists",
                )
            
            # Create user
            user = User(
                email=user_data.email,
                hashed_password=self._hash_password(user_data.password),
                full_name=user_data.full_name,
                tenant_id=user_data.tenant_id,
                role=user_data.role,
                is_active=True,
                created_at=datetime.datetime.now(),
            )
            
            self.db_session.add(user)
            self.db_session.commit()
            self.db_session.refresh(user)
            
            logger.info("user_created", user_id=user.id, email=user.email)
            return UserResponse(
                id=user.id,
                email=user.email,
                full_name=user.full_name,
                tenant_id=user.tenant_id,
                role=user.role,
                is_active=user.is_active,
                created_at=user.created_at,
            )
            
        except Exception as e:
            self.db_session.rollback()
            logger.error("user_creation_failed", error=str(e))
            log_api_error(
                endpoint="/register",
                error_type="server",
                error_message=str(e),
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to create user",
            )
    
    def authenticate_user(
        self,
        email: str,
        password: str,
        tenant_id: UUID4,
    ) -> Tuple[User, Token]:
        """Authenticate a user.
        
        Args:
            email: User email.
            password: User password.
            tenant_id: Tenant ID.
            
        Returns:
            Tuple of authenticated user and access token.
            
        Raises:
            HTTPException: If authentication fails.
        """
        # Get user
        user = self.db_session.query(User).filter(
            User.email == email,
            User.tenant_id == tenant_id,
        ).first()
        
        if not user or not self._verify_password(password, user.hashed_password):
            log_auth_attempt("password", False)
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials",
            )
        
        if not user.is_active:
            log_auth_attempt("password", False, user.id)
            raise HTTPException(
                status_code=401,
                detail="User is inactive",
            )
        
        # Create token
        token = self._create_access_token(user)
        
        log_auth_attempt("password", True, user.id)
        return user, token
    
    def get_current_user(
        self,
        token: str = Security(oauth2_scheme),
    ) -> User:
        """Get current user from token.
        
        Args:
            token: JWT token.
            
        Returns:
            Current user.
            
        Raises:
            HTTPException: If token is invalid or user not found.
        """
        try:
            # Decode token
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM],
            )
            token_data = TokenData(**payload)
            
            # Get user
            user = self.db_session.query(User).filter(
                User.id == token_data.user_id,
                User.tenant_id == token_data.tenant_id,
            ).first()
            
            if not user:
                log_api_error(
                    endpoint="/me",
                    error_type="auth",
                    error_message="User not found",
                )
                raise HTTPException(
                    status_code=401,
                    detail="User not found",
                )
            
            if not user.is_active:
                log_api_error(
                    endpoint="/me",
                    error_type="auth",
                    error_message="User is inactive",
                )
                raise HTTPException(
                    status_code=401,
                    detail="User is inactive",
                )
            
            return user
            
        except JWTError:
            log_api_error(
                endpoint="/me",
                error_type="auth",
                error_message="Invalid token",
            )
            raise HTTPException(
                status_code=401,
                detail="Invalid token",
            )
    
    def _hash_password(self, password: str) -> str:
        """Hash a password.
        
        Args:
            password: Plain text password.
            
        Returns:
            Hashed password.
        """
        return pwd_context.hash(password)
    
    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password.
        
        Args:
            plain_password: Plain text password.
            hashed_password: Hashed password.
            
        Returns:
            True if password is valid, False otherwise.
        """
        return pwd_context.verify(plain_password, hashed_password)
    
    def _create_access_token(self, user: User) -> Token:
        """Create an access token.
        
        Args:
            user: User to create token for.
            
        Returns:
            Access token.
        """
        # Set expiration
        expires_at = datetime.datetime.now() + datetime.timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        )
        
        # Create token data
        token_data = TokenData(
            user_id=user.id,
            tenant_id=user.tenant_id,
            role=user.role,
        )
        
        # Create token
        access_token = jwt.encode(
            token_data.dict(),
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_at=expires_at,
        ) 