"""Python SDK for AuditPulse AI API.

This module provides a client for interacting with the AuditPulse AI API.
"""
from typing import Any, Dict, List, Optional, Union
import aiohttp
import asyncio
from datetime import datetime
from pydantic import BaseModel, Field

from auditpulse_mvp.sdk.models import (
    Transaction,
    Anomaly,
    User,
    Tenant,
    RiskScore,
    Feedback,
    Notification,
    PaginatedResponse,
)


class AuditPulseClient:
    """Client for interacting with the AuditPulse AI API."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        token: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize the client.

        Args:
            base_url: Base URL of the API
            api_key: API key for authentication
            token: JWT token for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.token = token
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Create aiohttp session when entering context."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session when exiting context."""
        if self._session:
            await self._session.close()
            self._session = None

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> Any:
        """Make an HTTP request to the API.

        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional arguments for aiohttp.ClientSession.request

        Returns:
            API response data

        Raises:
            AuditPulseError: If the API request fails
        """
        if not self._session:
            raise RuntimeError("Client session not initialized. Use async with context.")

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = self._get_headers()

        try:
            async with self._session.request(
                method, url, headers=headers, **kwargs
            ) as response:
                data = await response.json()

                if not response.ok:
                    raise AuditPulseError(
                        code=data.get("code", "unknown"),
                        message=data.get("message", "Unknown error"),
                        details=data.get("details"),
                        status_code=response.status,
                    )

                return data
        except aiohttp.ClientError as e:
            raise AuditPulseError(
                code="request_error",
                message=str(e),
                status_code=500,
            )

    # Authentication methods
    async def login(self, email: str, password: str) -> Dict[str, str]:
        """Login and get access token.

        Args:
            email: User email
            password: User password

        Returns:
            Dictionary containing access token and refresh token
        """
        data = await self._request(
            "POST",
            "/auth/login",
            json={"email": email, "password": password},
        )
        self.token = data["access_token"]
        return data

    async def refresh_token(self, refresh_token: str) -> Dict[str, str]:
        """Refresh access token.

        Args:
            refresh_token: Refresh token

        Returns:
            Dictionary containing new access token and refresh token
        """
        data = await self._request(
            "POST",
            "/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        self.token = data["access_token"]
        return data

    # Transaction methods
    async def get_transactions(
        self,
        tenant_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        page: int = 1,
        size: int = 100,
    ) -> PaginatedResponse[Transaction]:
        """Get transactions for a tenant.

        Args:
            tenant_id: Tenant ID
            start_date: Start date filter
            end_date: End date filter
            page: Page number
            size: Page size

        Returns:
            Paginated list of transactions
        """
        params = {
            "tenant_id": tenant_id,
            "page": page,
            "size": size,
        }
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()

        data = await self._request("GET", "/transactions", params=params)
        return PaginatedResponse[Transaction].parse_obj(data)

    # Anomaly methods
    async def get_anomalies(
        self,
        tenant_id: str,
        status: Optional[str] = None,
        page: int = 1,
        size: int = 100,
    ) -> PaginatedResponse[Anomaly]:
        """Get anomalies for a tenant.

        Args:
            tenant_id: Tenant ID
            status: Anomaly status filter
            page: Page number
            size: Page size

        Returns:
            Paginated list of anomalies
        """
        params = {
            "tenant_id": tenant_id,
            "page": page,
            "size": size,
        }
        if status:
            params["status"] = status

        data = await self._request("GET", "/anomalies", params=params)
        return PaginatedResponse[Anomaly].parse_obj(data)

    async def update_anomaly(
        self,
        anomaly_id: str,
        status: str,
        resolution: Optional[str] = None,
    ) -> Anomaly:
        """Update anomaly status and resolution.

        Args:
            anomaly_id: Anomaly ID
            status: New status
            resolution: Resolution details

        Returns:
            Updated anomaly
        """
        data = await self._request(
            "PATCH",
            f"/anomalies/{anomaly_id}",
            json={"status": status, "resolution": resolution},
        )
        return Anomaly.parse_obj(data)

    # Feedback methods
    async def submit_feedback(
        self,
        anomaly_id: str,
        feedback: Feedback,
    ) -> Feedback:
        """Submit feedback for an anomaly.

        Args:
            anomaly_id: Anomaly ID
            feedback: Feedback data

        Returns:
            Created feedback
        """
        data = await self._request(
            "POST",
            f"/anomalies/{anomaly_id}/feedback",
            json=feedback.dict(),
        )
        return Feedback.parse_obj(data)

    # Notification methods
    async def get_notifications(
        self,
        tenant_id: str,
        page: int = 1,
        size: int = 100,
    ) -> PaginatedResponse[Notification]:
        """Get notifications for a tenant.

        Args:
            tenant_id: Tenant ID
            page: Page number
            size: Page size

        Returns:
            Paginated list of notifications
        """
        params = {
            "tenant_id": tenant_id,
            "page": page,
            "size": size,
        }

        data = await self._request("GET", "/notifications", params=params)
        return PaginatedResponse[Notification].parse_obj(data)

    async def update_notification_settings(
        self,
        tenant_id: str,
        settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update notification settings for a tenant.

        Args:
            tenant_id: Tenant ID
            settings: Notification settings

        Returns:
            Updated settings
        """
        data = await self._request(
            "PUT",
            f"/tenants/{tenant_id}/notification-settings",
            json=settings,
        )
        return data


class AuditPulseError(Exception):
    """Exception raised for AuditPulse API errors."""

    def __init__(
        self,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500,
    ):
        """Initialize the error.

        Args:
            code: Error code
            message: Error message
            details: Additional error details
            status_code: HTTP status code
        """
        self.code = code
        self.message = message
        self.details = details
        self.status_code = status_code
        super().__init__(f"{code}: {message}") 