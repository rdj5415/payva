"""Application settings module for AuditPulse MVP.

This module defines and validates all application settings using Pydantic.
"""
import os
import secrets
from typing import Literal, Optional, Union, List, Dict, Any
from functools import lru_cache

from pydantic import Field, PostgresDsn, SecretStr, AnyUrl, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Environment
    ENVIRONMENT: Literal["development", "test", "production"] = "development"

    # Database
    DATABASE_URL: Union[PostgresDsn, AnyUrl]
    DATABASE_TEST_URL: Optional[Union[PostgresDsn, AnyUrl]] = None
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10

    # Security
    SECRET_KEY: SecretStr
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # QuickBooks
    QUICKBOOKS_CLIENT_ID: str | None = None
    QUICKBOOKS_CLIENT_SECRET: SecretStr | None = None
    QUICKBOOKS_REDIRECT_URI: str | None = None
    QUICKBOOKS_ENVIRONMENT: Literal["sandbox", "production"] = "sandbox"

    # Plaid
    PLAID_CLIENT_ID: str | None = None
    PLAID_SECRET: SecretStr | None = None
    PLAID_ENVIRONMENT: Literal["sandbox", "development", "production"] = "sandbox"

    # OpenAI
    OPENAI_API_KEY: SecretStr | None = None

    # Notification Services
    SENDGRID_API_KEY: SecretStr | None = None
    TWILIO_ACCOUNT_SID: str | None = None
    TWILIO_AUTH_TOKEN: SecretStr | None = None
    SLACK_BOT_TOKEN: SecretStr | None = None

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Encryption
    ENCRYPTION_KEY: SecretStr | None = None

    # Feature Flags
    ENABLE_ML_ENGINE: bool = True
    ENABLE_GPT_EXPLANATIONS: bool = True
    ENABLE_DEMO_MODE: bool = False

    # Logging & Monitoring
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    ENABLE_PROMETHEUS: bool = True

    # Auth0
    AUTH0_DOMAIN: str
    AUTH0_CLIENT_ID: str
    AUTH0_CLIENT_SECRET: SecretStr
    AUTH0_CALLBACK_URL: str
    AUTH0_AUDIENCE: Optional[str] = None

    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: List[str] = ["*"]
    CORS_HEADERS: List[str] = ["*"]

    # Authentication
    ENABLE_GOOGLE_LOGIN: bool = False
    ENABLE_MICROSOFT_LOGIN: bool = False

    # Metrics
    ENABLE_METRICS: bool = True

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds

    # Base settings
    PROJECT_NAME: str = "Bungii"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[AnyUrl] = []
    
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        return v
        
    # Database settings
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "auditpulse"
    POSTGRES_PORT: str = "5432"
    DATABASE_URL: Optional[PostgresDsn] = None
    
    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            user=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_SERVER"),
            port=values.get("POSTGRES_PORT"),
            path=f"/{values.get('POSTGRES_DB') or ''}",
        )
        
    # Task settings
    TASK_RETRY_DELAY: int = 60  # 1 minute
    TASK_MAX_RETRIES: int = 3
    
    # Notification settings
    # Email
    SMTP_TLS: bool = True
    SMTP_PORT: int = 587
    SMTP_HOST: str = ""
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    EMAILS_FROM_EMAIL: str = ""
    EMAILS_FROM_NAME: str = "Bungii Notifications"
    
    # Slack
    SLACK_WEBHOOK_URL: Optional[str] = None
    
    # Twilio (SMS)
    TWILIO_FROM_NUMBER: Optional[str] = None
    
    # Webhook
    WEBHOOK_TIMEOUT_SECONDS: int = 10
    
    # Notification defaults
    DEFAULT_NOTIFICATION_CHANNEL: str = "email"
    NOTIFICATION_TASK_PRIORITY: str = "high"
    
    # Auth
    AUTH_REQUIRED: bool = True
    
    # Metrics
    METRICS_ENABLED: bool = True
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_TIMEFRAME: int = 60  # seconds


# Check if we're in test mode
if os.environ.get("ENVIRONMENT") == "test":
    # For tests, allow SQLite
    class TestSettings(Settings):
        """Test settings with SQLite support."""
        
        model_config = SettingsConfigDict(
            env_file=".env.test", env_file_encoding="utf-8", extra="ignore"
        )
        DATABASE_URL: AnyUrl
        DATABASE_TEST_URL: Optional[AnyUrl] = None
    
    settings = TestSettings()
else:
    # Normal settings for production/development
    settings = Settings()


@lru_cache
def get_settings() -> Settings:
    """Dependency function for FastAPI to get settings.

    Returns:
        Settings: Application settings.
    """
    return settings 