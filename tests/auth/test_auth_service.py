"""Tests for the authentication service.

This module contains tests for the authentication service functionality,
including user creation, authentication, and token management.
"""

import datetime
import uuid
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session
from httpx import Response

from auditpulse_mvp.auth.auth_service import AuthService, UserCreate, Token
from auditpulse_mvp.database.models import User, UserRole
from auditpulse_mvp.utils.settings import settings


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = MagicMock(spec=Session)
    return session


@pytest.fixture
def mock_user():
    """Create a mock user."""
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.email = "test@example.com"
    user.hashed_password = (
        "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiAYMyzJ/I6e"  # "password123"
    )
    user.full_name = "Test User"
    user.tenant_id = uuid.uuid4()
    user.role = UserRole.USER
    user.is_active = True
    user.created_at = datetime.datetime.now()
    return user


def test_create_user_success(
    mock_db_session,
    mock_user,
):
    """Test successful user creation."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.first.return_value = None

    # Create user data
    user_data = UserCreate(
        email="new@example.com",
        password="password123",
        full_name="New User",
        tenant_id=uuid.uuid4(),
        role=UserRole.USER,
    )

    # Initialize service
    service = AuthService(mock_db_session)

    # Create user
    with patch("auditpulse_mvp.auth.auth_service.User", return_value=mock_user):
        response = service.create_user(user_data)

    # Verify response
    assert response.id == mock_user.id
    assert response.email == mock_user.email
    assert response.full_name == mock_user.full_name
    assert response.tenant_id == mock_user.tenant_id
    assert response.role == mock_user.role
    assert response.is_active == mock_user.is_active
    assert response.created_at == mock_user.created_at

    # Verify database operations
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()
    mock_db_session.refresh.assert_called_once()


def test_create_user_already_exists(
    mock_db_session,
    mock_user,
):
    """Test user creation with existing email."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.first.return_value = (
        mock_user
    )

    # Create user data
    user_data = UserCreate(
        email=mock_user.email,
        password="password123",
        full_name="New User",
        tenant_id=mock_user.tenant_id,
        role=UserRole.USER,
    )

    # Initialize service
    service = AuthService(mock_db_session)

    # Attempt to create user
    with pytest.raises(HTTPException) as exc_info:
        service.create_user(user_data)

    # Verify exception
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "User already exists"


def test_authenticate_user_success(
    mock_db_session,
    mock_user,
):
    """Test successful user authentication."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.first.return_value = (
        mock_user
    )

    # Initialize service
    service = AuthService(mock_db_session)

    # Authenticate user
    user, token = service.authenticate_user(
        email=mock_user.email,
        password="password123",
        tenant_id=mock_user.tenant_id,
    )

    # Verify user
    assert user == mock_user

    # Verify token
    assert token.access_token is not None
    assert token.token_type == "bearer"
    assert token.expires_at > datetime.datetime.now()


def test_authenticate_user_invalid_credentials(
    mock_db_session,
    mock_user,
):
    """Test authentication with invalid credentials."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.first.return_value = (
        mock_user
    )

    # Initialize service
    service = AuthService(mock_db_session)

    # Attempt to authenticate
    with pytest.raises(HTTPException) as exc_info:
        service.authenticate_user(
            email=mock_user.email,
            password="wrongpassword",
            tenant_id=mock_user.tenant_id,
        )

    # Verify exception
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid credentials"


def test_authenticate_user_inactive(
    mock_db_session,
    mock_user,
):
    """Test authentication with inactive user."""
    # Setup
    mock_user.is_active = False
    mock_db_session.query.return_value.filter.return_value.first.return_value = (
        mock_user
    )

    # Initialize service
    service = AuthService(mock_db_session)

    # Attempt to authenticate
    with pytest.raises(HTTPException) as exc_info:
        service.authenticate_user(
            email=mock_user.email,
            password="password123",
            tenant_id=mock_user.tenant_id,
        )

    # Verify exception
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "User is inactive"


def test_get_current_user_success(
    mock_db_session,
    mock_user,
):
    """Test successful current user retrieval."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.first.return_value = (
        mock_user
    )

    # Initialize service
    service = AuthService(mock_db_session)

    # Create token
    token = service._create_access_token(mock_user)

    # Get current user
    with patch("auditpulse_mvp.auth.auth_service.jwt.decode") as mock_decode:
        mock_decode.return_value = {
            "user_id": str(mock_user.id),
            "tenant_id": str(mock_user.tenant_id),
            "role": mock_user.role.value,
        }
        user = service.get_current_user(token.access_token)

    # Verify user
    assert user == mock_user


def test_get_current_user_invalid_token(
    mock_db_session,
):
    """Test current user retrieval with invalid token."""
    # Initialize service
    service = AuthService(mock_db_session)

    # Attempt to get current user
    with pytest.raises(HTTPException) as exc_info:
        service.get_current_user("invalid_token")

    # Verify exception
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid token"


@pytest.fixture
def auth_service(db_session):
    """Create an AuthService instance."""
    return AuthService(db_session)


@pytest.fixture
def mock_auth0_response():
    """Mock Auth0 token response."""
    return {
        "access_token": "mock_access_token",
        "refresh_token": "mock_refresh_token",
        "expires_in": 3600,
        "token_type": "bearer",
    }


@pytest.fixture
def mock_user_info():
    """Mock Auth0 user info."""
    return {
        "email": "test@example.com",
        "name": "Test User",
        "picture": "https://example.com/avatar.jpg",
    }


@pytest.mark.asyncio
async def test_exchange_auth0_code_success(auth_service, mock_auth0_response):
    """Test successful Auth0 code exchange."""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = Response(
            status_code=200,
            json=mock_auth0_response,
        )

        result = await auth_service.exchange_auth0_code("mock_code", "mock_state")

        assert result == mock_auth0_response
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_exchange_auth0_code_failure(auth_service):
    """Test failed Auth0 code exchange."""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = Response(
            status_code=401,
            json={"error": "invalid_grant"},
        )

        with pytest.raises(HTTPException) as exc_info:
            await auth_service.exchange_auth0_code("mock_code", "mock_state")

        assert exc_info.value.status_code == 401
        assert "Failed to exchange code for token" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_get_or_create_auth0_user_new(auth_service, mock_user_info):
    """Test creating a new user from Auth0."""
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = Response(
            status_code=200,
            json=mock_user_info,
        )

        token_data = {"access_token": "mock_token"}
        user = await auth_service.get_or_create_auth0_user(token_data)

        assert user.email == mock_user_info["email"]
        assert user.full_name == mock_user_info["name"]
        assert user.picture == mock_user_info["picture"]
        assert user.role == UserRole.VIEWER
        assert user.is_active is True


@pytest.mark.asyncio
async def test_get_or_create_auth0_user_existing(auth_service, mock_user_info):
    """Test getting an existing user from Auth0."""
    # Create existing user
    existing_user = User(
        email=mock_user_info["email"],
        full_name="Old Name",
        role=UserRole.ADMIN,
        is_active=True,
        created_at=datetime.datetime.now(),
    )
    auth_service.db_session.add(existing_user)
    await auth_service.db_session.commit()

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = Response(
            status_code=200,
            json=mock_user_info,
        )

        token_data = {"access_token": "mock_token"}
        user = await auth_service.get_or_create_auth0_user(token_data)

        assert user.id == existing_user.id
        assert user.full_name == mock_user_info["name"]
        assert user.picture == mock_user_info["picture"]
        assert user.role == UserRole.ADMIN  # Role should not change


@pytest.mark.asyncio
async def test_refresh_token_success(auth_service, mock_auth0_response):
    """Test successful token refresh."""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = Response(
            status_code=200,
            json=mock_auth0_response,
        )

        result = await auth_service.refresh_token("mock_refresh_token")

        assert isinstance(result, Token)
        assert result.access_token == mock_auth0_response["access_token"]
        assert result.refresh_token == mock_auth0_response["refresh_token"]
        assert result.token_type == "bearer"


@pytest.mark.asyncio
async def test_refresh_token_failure(auth_service):
    """Test failed token refresh."""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = Response(
            status_code=401,
            json={"error": "invalid_grant"},
        )

        with pytest.raises(HTTPException) as exc_info:
            await auth_service.refresh_token("mock_refresh_token")

        assert exc_info.value.status_code == 401
        assert "Invalid refresh token" in str(exc_info.value.detail)


def test_create_session_token(auth_service):
    """Test creating a session token."""
    user = User(
        email="test@example.com",
        full_name="Test User",
        role=UserRole.ADMIN,
        is_active=True,
        created_at=datetime.datetime.now(),
    )

    token = auth_service.create_session_token(user)

    assert isinstance(token, Token)
    assert token.token_type == "bearer"
    assert token.expires_at > datetime.datetime.now()
    assert token.refresh_token is None
