"""
Security middleware for AuditPulse MVP.

This module provides security middlewares for the FastAPI application,
including rate limiting, CSRF protection, security headers, IP allowlisting,
and comprehensive audit logging.
"""

import time
import secrets
import ipaddress
from typing import Callable, Dict, List, Optional, Tuple, Union, Set
from fastapi import Request, Response, HTTPException, status, Depends
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import JSONResponse
import logging
from datetime import datetime, timedelta
import re
import json

# Configure logging
logger = logging.getLogger("auditpulse.security")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting API requests.

    This middleware limits the number of requests a client can make
    within a specified time window.
    """

    def __init__(
        self,
        app,
        requests_limit: int = 100,
        window_seconds: int = 60,
        exclude_paths: List[str] = None,
        admin_requests_limit: int = 200,  # Higher limit for admin endpoints
        burst_limit: int = 10,  # Maximum requests allowed in a burst (1 second)
        block_duration_seconds: int = 300,  # Block excessive requests for 5 minutes
    ):
        """
        Initialize the rate limit middleware.

        Args:
            app: The FastAPI application
            requests_limit: Maximum number of requests allowed in the time window
            window_seconds: Time window in seconds
            exclude_paths: List of paths to exclude from rate limiting
            admin_requests_limit: Maximum number of requests for admin endpoints
            burst_limit: Maximum requests allowed in a 1-second window
            block_duration_seconds: How long to block IPs that exceed limits
        """
        super().__init__(app)
        self.requests_limit = requests_limit
        self.admin_requests_limit = admin_requests_limit
        self.window_seconds = window_seconds
        self.exclude_paths = exclude_paths or ["/health", "/api/docs", "/api/redoc"]
        self.requests = {}  # Dict to track requests: {ip: [(timestamp, path), ...]}
        self.blocked_ips = {}  # Dict to track blocked IPs: {ip: unblock_timestamp}
        self.burst_limit = burst_limit
        self.block_duration_seconds = block_duration_seconds
        self.cleanup_counter = 0

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process incoming requests and apply rate limiting.

        Args:
            request: The incoming request
            call_next: The next middleware or route handler

        Returns:
            The response from the next middleware or route handler
        """
        # Skip rate limiting for excluded paths
        path = request.url.path
        if any(path.startswith(exclude_path) for exclude_path in self.exclude_paths):
            return await call_next(request)

        # Get client identifier (IP address or API key if available)
        client_id = self._get_client_id(request)

        # Check if client is currently blocked
        current_time = time.time()
        if client_id in self.blocked_ips:
            unblock_time = self.blocked_ips[client_id]
            if current_time < unblock_time:
                remaining_time = int(unblock_time - current_time)
                logger.warning(
                    f"Blocked request from {client_id}, remaining block time: {remaining_time}s"
                )
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": f"Too many requests. IP blocked for {remaining_time} seconds.",
                        "retry_after": remaining_time,
                    },
                )
            else:
                del self.blocked_ips[client_id]

        # Record the request
        if client_id not in self.requests:
            self.requests[client_id] = []

        # Add the current request to the list
        self.requests[client_id].append((current_time, path))

        # Remove old requests outside the time window
        self.requests[client_id] = [
            req
            for req in self.requests[client_id]
            if current_time - req[0] < self.window_seconds
        ]

        # Get the appropriate rate limit based on the path
        is_admin_path = path.startswith("/api/v1/admin")
        limit = self.admin_requests_limit if is_admin_path else self.requests_limit

        # Periodically clean up the requests dictionary
        self.cleanup_counter += 1
        if self.cleanup_counter > 1000:
            self._cleanup_old_requests()
            self.cleanup_counter = 0

        # Check if the client has exceeded the rate limit
        if len(self.requests[client_id]) > limit:
            logger.warning(f"Rate limit exceeded for client {client_id}")
            # Block the IP if it repeatedly exceeds the rate limit
            self.blocked_ips[client_id] = current_time + self.block_duration_seconds
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": f"Rate limit exceeded. IP blocked for {self.block_duration_seconds} seconds.",
                    "retry_after": self.block_duration_seconds,
                },
            )

        # Check for burst (too many requests in 1 second)
        last_second_requests = len(
            [req for req in self.requests[client_id] if current_time - req[0] < 1]
        )

        if last_second_requests > self.burst_limit:
            logger.warning(f"Burst limit exceeded for client {client_id}")
            # Block the IP for a short time
            self.blocked_ips[client_id] = current_time + 30  # Block for 30 seconds
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Request rate too high. Please slow down.",
                    "retry_after": 30,
                },
            )

        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, limit - len(self.requests[client_id]))
        )
        response.headers["X-RateLimit-Reset"] = str(
            int(current_time + self.window_seconds)
        )

        return response

    def _get_client_id(self, request: Request) -> str:
        """
        Get a unique identifier for the client.

        Uses the client's IP address or API key if available.

        Args:
            request: The incoming request

        Returns:
            A unique identifier for the client
        """
        # Try to get an API key from the authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return f"token:{auth_header[7:]}"  # Return the token as the ID

        # Fall back to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Get the original client's IP if behind a proxy
            return forwarded_for.split(",")[0].strip()

        # Use the client's direct IP
        return request.client.host if request.client else "unknown"

    def _cleanup_old_requests(self):
        """
        Clean up the requests dictionary by removing entries
        for clients who haven't made requests recently.
        """
        current_time = time.time()
        cutoff_time = current_time - (2 * self.window_seconds)

        # Find inactive clients
        inactive_clients = []
        for client_id, requests in self.requests.items():
            if not requests or max(r[0] for r in requests) < cutoff_time:
                inactive_clients.append(client_id)

        # Remove inactive clients
        for client_id in inactive_clients:
            del self.requests[client_id]

        # Clean up expired blocks
        expired_blocks = [
            ip
            for ip, unblock_time in self.blocked_ips.items()
            if current_time > unblock_time
        ]
        for ip in expired_blocks:
            del self.blocked_ips[ip]


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to responses.

    These headers help protect against common web vulnerabilities.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Add security headers to the response.

        Args:
            request: The incoming request
            call_next: The next middleware or route handler

        Returns:
            The response with added security headers
        """
        response = await call_next(request)

        # Content-Security-Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' https://cdn.jsdelivr.net; "
            "style-src 'self' https://cdn.jsdelivr.net; "
            "img-src 'self' data:; "
            "font-src 'self' https://cdn.jsdelivr.net; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "form-action 'self';"
        )

        # Other security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=()"
        )

        # Remove the Server header to avoid information disclosure
        response.headers.pop("Server", None)

        return response


class CSRFMiddleware(BaseHTTPMiddleware):
    """
    Middleware for CSRF protection.

    Implements the Double Submit Cookie pattern for CSRF protection.
    """

    def __init__(
        self,
        app,
        cookie_name: str = "csrf_token",
        header_name: str = "X-CSRF-Token",
        cookie_max_age: int = 3600,  # 1 hour
        exclude_methods: List[str] = None,
        exclude_paths: List[str] = None,
        samesite: str = "strict",
        secure: bool = True,
    ):
        """
        Initialize the CSRF middleware.

        Args:
            app: The FastAPI application
            cookie_name: Name of the CSRF cookie
            header_name: Name of the CSRF header
            cookie_max_age: Max age of the CSRF cookie in seconds
            exclude_methods: HTTP methods to exclude from CSRF protection
            exclude_paths: URL paths to exclude from CSRF protection
            samesite: SameSite cookie attribute value
            secure: Whether cookies should be secure
        """
        super().__init__(app)
        self.cookie_name = cookie_name
        self.header_name = header_name
        self.cookie_max_age = cookie_max_age
        self.exclude_methods = exclude_methods or ["GET", "HEAD", "OPTIONS"]
        self.exclude_paths = exclude_paths or [
            "/api/v1/auth/login",
            "/api/v1/auth/refresh",
            "/health",
        ]
        self.samesite = samesite
        self.secure = secure

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process incoming requests and apply CSRF protection.

        Args:
            request: The incoming request
            call_next: The next middleware or route handler

        Returns:
            The response from the next middleware or route handler
        """
        # Skip CSRF check for excluded methods
        if request.method in self.exclude_methods:
            response = await call_next(request)
            # For GET requests, set the CSRF token
            if request.method == "GET" and request.url.path not in [
                "/api/docs",
                "/api/redoc",
            ]:
                response = self._set_csrf_token(response)
            return response

        # Skip CSRF check for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Verify CSRF token
        csrf_cookie = request.cookies.get(self.cookie_name)
        csrf_header = request.headers.get(self.header_name)

        if not csrf_cookie or not csrf_header or csrf_cookie != csrf_header:
            logger.warning(f"CSRF validation failed for {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "CSRF token missing or invalid"},
            )

        # Process the request
        response = await call_next(request)

        # Refresh the CSRF token
        response = self._set_csrf_token(response)

        return response

    def _set_csrf_token(self, response: Response) -> Response:
        """
        Set a CSRF token in the response.

        Args:
            response: The response to add the CSRF token to

        Returns:
            The response with the CSRF token cookie
        """
        # Generate a secure token
        token = secrets.token_hex(32)

        # Set the token as a cookie
        response.set_cookie(
            key=self.cookie_name,
            value=token,
            max_age=self.cookie_max_age,
            httponly=False,  # JavaScript needs to access it to add it to headers
            secure=self.secure,
            samesite=self.samesite,
        )

        return response


class IPAllowListMiddleware(BaseHTTPMiddleware):
    """
    Middleware for IP address allowlisting.

    Restricts access to admin endpoints based on IP address.
    """

    def __init__(
        self,
        app,
        protected_paths: List[str] = None,
        allowed_ips: List[str] = None,
        allowed_ip_ranges: List[str] = None,
        admin_api_key_header: str = "X-Admin-API-Key",
        admin_api_keys: List[str] = None,
    ):
        """
        Initialize the IP allowlist middleware.

        Args:
            app: The FastAPI application
            protected_paths: List of path prefixes to protect
            allowed_ips: List of allowed IP addresses
            allowed_ip_ranges: List of allowed IP ranges in CIDR notation
            admin_api_key_header: Header name for admin API key
            admin_api_keys: List of valid admin API keys
        """
        super().__init__(app)
        self.protected_paths = protected_paths or ["/api/v1/admin"]
        self.allowed_ips = set(allowed_ips or [])
        self.allowed_ip_ranges = []
        self.admin_api_key_header = admin_api_key_header
        self.admin_api_keys = set(admin_api_keys or [])

        # Parse CIDR ranges
        for ip_range in allowed_ip_ranges or []:
            try:
                self.allowed_ip_ranges.append(ipaddress.ip_network(ip_range))
            except ValueError:
                logger.error(f"Invalid IP range: {ip_range}")

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Check if the request IP is allowed for protected paths.

        Args:
            request: The incoming request
            call_next: The next middleware or route handler

        Returns:
            The response from the next middleware or route handler
        """
        # Check if the path is protected
        path = request.url.path
        is_protected = any(path.startswith(prefix) for prefix in self.protected_paths)

        if is_protected:
            # Check for admin API key
            api_key = request.headers.get(self.admin_api_key_header)
            if api_key and api_key in self.admin_api_keys:
                return await call_next(request)

            # Get client IP
            client_ip = self._get_client_ip(request)

            # Check if IP is allowed
            if not self._is_ip_allowed(client_ip):
                logger.warning(f"Access denied for IP {client_ip} to {path}")
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "Access denied from your IP address"},
                )

        return await call_next(request)

    def _get_client_ip(self, request: Request) -> str:
        """Get the client's IP address."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _is_ip_allowed(self, ip: str) -> bool:
        """Check if the IP is in the allowlist."""
        if ip in self.allowed_ips:
            return True

        try:
            ip_obj = ipaddress.ip_address(ip)
            for network in self.allowed_ip_ranges:
                if ip_obj in network:
                    return True
        except ValueError:
            return False

        return False


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging security-relevant events.

    Records security-related events such as authentication attempts,
    access to sensitive resources, and potential security violations.
    """

    def __init__(
        self,
        app,
        sensitive_paths: List[str] = None,
        sensitive_params: List[str] = None,
        log_request_body: bool = False,
    ):
        """
        Initialize the audit logging middleware.

        Args:
            app: The FastAPI application
            sensitive_paths: Paths to consider sensitive for logging
            sensitive_params: Request parameters to mask in logs
            log_request_body: Whether to log request bodies
        """
        super().__init__(app)
        self.sensitive_paths = sensitive_paths or [
            "/api/v1/admin",
            "/api/v1/users",
            "/api/v1/auth",
            "/api/v1/plaid",
        ]
        self.sensitive_params = sensitive_params or [
            "password",
            "token",
            "key",
            "secret",
            "credential",
            "auth",
        ]
        self.log_request_body = log_request_body

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Log security-relevant events.

        Args:
            request: The incoming request
            call_next: The next middleware or route handler

        Returns:
            The response from the next middleware or route handler
        """
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        user_id = "anonymous"
        path = request.url.path
        method = request.method

        # Try to extract user ID from the request if available
        if hasattr(request.state, "user") and hasattr(request.state.user, "id"):
            user_id = request.state.user.id

        # Extract and mask query parameters if needed
        query_params = dict(request.query_params)
        self._mask_sensitive_params(query_params)

        # Log request details for sensitive paths
        if any(path.startswith(prefix) for prefix in self.sensitive_paths):
            request_log = {
                "timestamp": datetime.now().isoformat(),
                "client_ip": client_ip,
                "user_id": user_id,
                "method": method,
                "path": path,
                "query_params": query_params,
                "headers": self._get_safe_headers(request),
            }

            # Log request body for non-GET requests if enabled
            if self.log_request_body and method not in ["GET", "HEAD"]:
                try:
                    # This is a bit tricky as we can't read the body without consuming it
                    # For a real implementation, we'd need to handle this differently
                    # This is just for illustration
                    body = getattr(request, "_body", None)
                    if body:
                        try:
                            body_dict = json.loads(body)
                            self._mask_sensitive_params(body_dict)
                            request_log["body"] = body_dict
                        except:
                            request_log["body"] = "[binary data]"
                except:
                    request_log["body"] = "[error reading body]"

            logger.info(f"Security audit: {json.dumps(request_log)}")

        # Process the request
        try:
            response = await call_next(request)

            # Log successful requests to sensitive endpoints
            if self._is_sensitive_endpoint(path):
                self._log_sensitive_access(
                    request, response, client_ip, user_id, time.time() - start_time
                )

            # Log failed authentication attempts
            if (
                path == "/api/v1/auth/login"
                and response.status_code == status.HTTP_401_UNAUTHORIZED
            ):
                self._log_auth_failure(request, client_ip)

            return response

        except Exception as e:
            # Log exceptions
            logger.error(
                f"Exception during request processing: {str(e)}, "
                f"path={path}, method={method}, "
                f"client_ip={client_ip}, user_id={user_id}"
            )
            raise

    def _get_client_ip(self, request: Request) -> str:
        """Get the client's IP address."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _is_sensitive_endpoint(self, path: str) -> bool:
        """Check if the path is a sensitive endpoint."""
        sensitive_patterns = [
            "/api/v1/admin/",
            "/api/v1/users/",
            "/api/v1/models/",
            "/api/v1/transactions/",
            "/api/v1/anomalies/",
        ]
        return any(path.startswith(pattern) for pattern in sensitive_patterns)

    def _log_sensitive_access(
        self,
        request: Request,
        response: Response,
        client_ip: str,
        user_id: str,
        duration: float,
    ) -> None:
        """Log access to sensitive endpoints."""
        logger.info(
            f"Sensitive resource access: path={request.url.path}, "
            f"method={request.method}, status_code={response.status_code}, "
            f"client_ip={client_ip}, user_id={user_id}, duration={duration:.3f}s"
        )

    def _log_auth_failure(self, request: Request, client_ip: str) -> None:
        """Log failed authentication attempts."""
        try:
            body = request.json()
            username = body.get("username", "unknown")
        except:
            username = "unknown"

        logger.warning(
            f"Authentication failure: username={username}, "
            f"client_ip={client_ip}, path={request.url.path}"
        )

    def _mask_sensitive_params(self, params: Dict) -> None:
        """Mask sensitive parameters in place."""
        if not params:
            return

        for key in params:
            for sensitive in self.sensitive_params:
                if sensitive.lower() in key.lower():
                    params[key] = "[REDACTED]"

    def _get_safe_headers(self, request: Request) -> Dict:
        """Get headers with sensitive information removed."""
        headers = dict(request.headers)
        sensitive_headers = ["authorization", "cookie", "x-api-key", "x-admin-api-key"]

        for header in sensitive_headers:
            if header in headers:
                headers[header] = "[REDACTED]"

        return headers


def setup_security_middleware(app, config=None):
    """
    Set up all security middleware for the application.

    Args:
        app: The FastAPI application
        config: Optional configuration dictionary
    """
    config = config or {}

    # Add IP allowlisting for admin endpoints
    allowed_ips = config.get("allowed_ips", [])
    allowed_ip_ranges = config.get("allowed_ip_ranges", ["127.0.0.1/32", "10.0.0.0/8"])
    admin_api_keys = config.get("admin_api_keys", [])

    app.add_middleware(
        IPAllowListMiddleware,
        allowed_ips=allowed_ips,
        allowed_ip_ranges=allowed_ip_ranges,
        admin_api_keys=admin_api_keys,
    )

    # Add rate limiting
    app.add_middleware(
        RateLimitMiddleware,
        requests_limit=config.get("rate_limit", 100),
        window_seconds=config.get("rate_limit_window", 60),
        exclude_paths=config.get(
            "rate_limit_exclude_paths", ["/health", "/api/docs", "/api/redoc"]
        ),
        burst_limit=config.get("burst_limit", 10),
    )

    # Add security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # Add CSRF protection
    app.add_middleware(
        CSRFMiddleware,
        cookie_name=config.get("csrf_cookie_name", "auditpulse_csrf_token"),
        header_name=config.get("csrf_header_name", "X-CSRF-Token"),
        secure=config.get("secure_cookies", True),
        samesite=config.get("cookie_samesite", "lax"),
    )

    # Add audit logging
    app.add_middleware(
        AuditLoggingMiddleware,
        log_request_body=config.get("log_request_body", False),
    )
