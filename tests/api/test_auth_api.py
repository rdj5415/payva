"""Tests for the authentication API endpoints.

This module contains tests for the authentication API endpoints,
including registration, login, and user information retrieval.
"""
import datetime
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from auditpulse_mvp.api.api_v1.endpoints.auth import router
from auditpulse_mvp.auth.auth_service import Token, UserCreate, UserResponse
from auditpulse_mvp.database.models import User, UserRole


@pytest.fixture
def app():
    """Create a test FastAPI application."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


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
    user.hashed_password = "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyNiAYMyzJ/I6e"  # "password123"
    user.full_name = "Test User"
    user.tenant_id = uuid.uuid4()
    user.role = UserRole.USER
    user.is_active = True
    user.created_at = datetime.datetime.now()
    return user


def test_register_success(
    client,
    mock_db_session,
    mock_user,
):
    """Test successful user registration."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.first.return_value = None
    
    # Create user data
    user_data = {
        "email": "new@example.com",
        "password": "password123",
        "full_name": "New User",
        "tenant_id": str(uuid.uuid4()),
        "role": UserRole.USER.value,
    }
    
    with patch("auditpulse_mvp.api.api_v1.endpoints.auth.get_db_session", return_value=mock_db_session), \
         patch("auditpulse_mvp.api.api_v1.endpoints.auth.AuthService") as mock_service:
        
        # Mock service response
        mock_service.return_value.create_user.return_value = UserResponse(
            id=mock_user.id,
            email=user_data["email"],
            full_name=user_data["full_name"],
            tenant_id=uuid.UUID(user_data["tenant_id"]),
            role=UserRole.USER,
            is_active=True,
            created_at=datetime.datetime.now(),
        )
        
        # Register user
        response = client.post("/api/v1/register", json=user_data)
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == user_data["email"]
        assert data["full_name"] == user_data["full_name"]
        assert data["tenant_id"] == user_data["tenant_id"]
        assert data["role"] == user_data["role"]


def test_register_user_exists(
    client,
    mock_db_session,
    mock_user,
):
    """Test user registration with existing email."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_user
    
    # Create user data
    user_data = {
        "email": mock_user.email,
        "password": "password123",
        "full_name": "New User",
        "tenant_id": str(mock_user.tenant_id),
        "role": UserRole.USER.value,
    }
    
    with patch("auditpulse_mvp.api.api_v1.endpoints.auth.get_db_session", return_value=mock_db_session), \
         patch("auditpulse_mvp.api.api_v1.endpoints.auth.AuthService") as mock_service:
        
        # Mock service exception
        mock_service.return_value.create_user.side_effect = Exception("User already exists")
        
        # Register user
        response = client.post("/api/v1/register", json=user_data)
        
        # Verify response
        assert response.status_code == 500


def test_login_success(
    client,
    mock_db_session,
    mock_user,
):
    """Test successful user login."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_user
    
    # Create login data
    login_data = {
        "email": mock_user.email,
        "password": "password123",
        "tenant_id": str(mock_user.tenant_id),
    }
    
    with patch("auditpulse_mvp.api.api_v1.endpoints.auth.get_db_session", return_value=mock_db_session), \
         patch("auditpulse_mvp.api.api_v1.endpoints.auth.AuthService") as mock_service:
        
        # Mock service response
        mock_service.return_value.authenticate_user.return_value = (
            mock_user,
            Token(
                access_token="test_token",
                token_type="bearer",
                expires_at=datetime.datetime.now() + datetime.timedelta(minutes=30),
            ),
        )
        
        # Login user
        response = client.post("/api/v1/login", json=login_data)
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["access_token"] == "test_token"
        assert data["token_type"] == "bearer"
        assert "expires_at" in data


def test_login_invalid_credentials(
    client,
    mock_db_session,
    mock_user,
):
    """Test login with invalid credentials."""
    # Setup
    mock_db_session.query.return_value.filter.return_value.first.return_value = mock_user
    
    # Create login data
    login_data = {
        "email": mock_user.email,
        "password": "wrongpassword",
        "tenant_id": str(mock_user.tenant_id),
    }
    
    with patch("auditpulse_mvp.api.api_v1.endpoints.auth.get_db_session", return_value=mock_db_session), \
         patch("auditpulse_mvp.api.api_v1.endpoints.auth.AuthService") as mock_service:
        
        # Mock service exception
        mock_service.return_value.authenticate_user.side_effect = Exception("Invalid credentials")
        
        # Login user
        response = client.post("/api/v1/login", json=login_data)
        
        # Verify response
        assert response.status_code == 500


def test_get_current_user_info(
    client,
    mock_user,
):
    """Test getting current user information."""
    with patch("auditpulse_mvp.api.api_v1.endpoints.auth.get_current_user", return_value=mock_user):
        # Get user info
        response = client.get("/api/v1/me")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(mock_user.id)
        assert data["email"] == mock_user.email
        assert data["full_name"] == mock_user.full_name
        assert data["tenant_id"] == str(mock_user.tenant_id)
        assert data["role"] == mock_user.role.value
        assert data["is_active"] == mock_user.is_active


def test_change_password_success(
    client,
    mock_db_session,
    mock_user,
):
    """Test successful password change."""
    with patch("auditpulse_mvp.api.api_v1.endpoints.auth.get_current_user", return_value=mock_user), \
         patch("auditpulse_mvp.api.api_v1.endpoints.auth.get_db_session", return_value=mock_db_session), \
         patch("auditpulse_mvp.api.api_v1.endpoints.auth.AuthService") as mock_service:
        
        # Mock password verification
        mock_service.return_value._verify_password.return_value = True
        
        # Change password
        response = client.post(
            "/api/v1/change-password",
            json={
                "current_password": "password123",
                "new_password": "newpassword123",
            },
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Password changed successfully"
        
        # Verify database operations
        mock_db_session.commit.assert_called_once()


def test_change_password_invalid_current(
    client,
    mock_db_session,
    mock_user,
):
    """Test password change with invalid current password."""
    with patch("auditpulse_mvp.api.api_v1.endpoints.auth.get_current_user", return_value=mock_user), \
         patch("auditpulse_mvp.api.api_v1.endpoints.auth.get_db_session", return_value=mock_db_session), \
         patch("auditpulse_mvp.api.api_v1.endpoints.auth.AuthService") as mock_service:
        
        # Mock password verification
        mock_service.return_value._verify_password.return_value = False
        
        # Change password
        response = client.post(
            "/api/v1/change-password",
            json={
                "current_password": "wrongpassword",
                "new_password": "newpassword123",
            },
        )
        
        # Verify response
        assert response.status_code == 400
        data = response.json()
        assert data["detail"] == "Invalid current password" 