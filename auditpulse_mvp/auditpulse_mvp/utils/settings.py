"""Application settings module for AuditPulse MVP.

This module defines and validates all application settings using Pydantic.
"""

from typing import Literal, Optional, Any, Dict, List, Union

from pydantic import Field, SecretStr, field_validator, AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
import secrets
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with validation."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Database
    DATABASE_URL: str
    DATABASE_TEST_URL: Optional[str] = None

    @field_validator("DATABASE_URL", "DATABASE_TEST_URL", mode="before")
    def validate_database_url(cls, v: Any) -> str:
        """Validate that the database URL is a string.

        Args:
            v: The value to validate.

        Returns:
            str: The validated database URL.
        """
        if not v:
            return v
        if not isinstance(v, str):
            raise ValueError("Database URL must be a string")
        return v

    # Security
    SECRET_KEY: SecretStr
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Auth0
    AUTH0_DOMAIN: str = "auditpulse.us.auth0.com"
    AUTH0_CLIENT_ID: str = ""
    AUTH0_CLIENT_SECRET: Optional[SecretStr] = None
    AUTH0_AUDIENCE: str = "https://api.auditpulse.ai"
    AUTH0_CALLBACK_URL: str = "http://localhost:3000/callback"
    ENABLE_AUTH0_LOGIN: bool = True
    ENABLE_EMAIL_LOGIN: bool = True
    ENABLE_GOOGLE_LOGIN: bool = False
    ENABLE_MICROSOFT_LOGIN: bool = False

    # Multi-tenancy
    ENABLE_TENANT_ISOLATION: bool = True
    REQUIRE_TENANT_HEADER: bool = True
    TENANT_HEADER_NAME: str = "X-Tenant-ID"

    # QuickBooks
    QUICKBOOKS_CLIENT_ID: Optional[str] = None
    QUICKBOOKS_CLIENT_SECRET: Optional[SecretStr] = None
    QUICKBOOKS_REDIRECT_URI: Optional[str] = None
    QUICKBOOKS_ENVIRONMENT: Literal["sandbox", "production"] = "sandbox"

    # Plaid
    PLAID_CLIENT_ID: Optional[str] = None
    PLAID_SECRET: Optional[SecretStr] = None
    PLAID_ENVIRONMENT: Literal["sandbox", "development", "production"] = "sandbox"

    # OpenAI
    OPENAI_API_KEY: Optional[SecretStr] = None

    # Notification Services
    SENDGRID_API_KEY: Optional[SecretStr] = None
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[SecretStr] = None
    SLACK_BOT_TOKEN: Optional[SecretStr] = None
    FRONTEND_URL: str = "http://localhost:3000"
    NOTIFICATION_CHECK_INTERVAL_MINUTES: int = 15
    DEFAULT_FROM_EMAIL: str = "alerts@auditpulse.ai"
    DEFAULT_FROM_PHONE: str = (
        "+15555555555"  # Replace with actual default in production
    )

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Encryption
    ENCRYPTION_KEY: Optional[SecretStr] = None

    # Feature Flags
    ENABLE_ML_ENGINE: bool = True
    ENABLE_GPT_EXPLANATIONS: bool = True
    ENABLE_DEMO_MODE: bool = False
    ENABLE_RISK_ENGINE: bool = True
    ENABLE_NOTIFICATIONS: bool = True

    # Logging & Monitoring
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    ENABLE_PROMETHEUS: bool = True

    # ML Engine settings
    ENABLE_ML_SCHEDULER: bool = True
    ML_RETRAINING_HOUR: int = 2  # 2 AM by default
    ML_RETRAINING_MINUTE: int = 0
    ML_SCORE_THRESHOLD: float = 0.7  # Threshold for ML-detected anomalies
    ML_DEFAULT_TRAINING_DAYS: int = 365  # Use 1 year of data by default

    # Risk Engine settings
    RISK_SCORE_THRESHOLD: int = 50  # Default threshold for flagging risk (0-100)
    RISK_RESCORING_DAYS: int = 30  # Default days to rescore anomalies

    # Rules Engine settings
    ENABLE_RULES_ENGINE: bool = True

    # API configuration
    PROJECT_NAME: str = "AuditPulse MVP"
    API_V1_PREFIX: str = "/api/v1"
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    # Testing
    TESTING: bool = False

    # Feedback Learning settings
    ENABLE_FEEDBACK_LEARNING: bool = True
    FEEDBACK_LEARNING_HOUR: int = 3  # 3 AM
    FEEDBACK_LEARNING_MINUTE: int = 0  # 0 minutes
    FALSE_POSITIVE_THRESHOLD: int = 5  # Trigger learning after 5 false positives
    FEEDBACK_MIN_COUNT: int = 5  # Minimum feedback count to adjust rules
    FEEDBACK_LOOKBACK_DAYS: int = 30  # Look back 30 days for feedback

    # Derived settings with lowercase for ease of use
    @property
    def database_url(self) -> str:
        """Get the database URL with test suffix if in test mode."""
        if self.TESTING:
            # Special case for SQLite - use a different file for testing
            if self.DATABASE_URL.startswith("sqlite"):
                return self.DATABASE_URL.replace("sqlite:", "sqlite:")

            # For other databases, ensure we're using a test database
            if "test" not in self.DATABASE_URL.lower():
                if "?" in self.DATABASE_URL:
                    return f"{self.DATABASE_URL}&test=true"
                else:
                    return f"{self.DATABASE_URL}?test=true"
        return self.DATABASE_URL

    @property
    def enable_ml_engine(self) -> bool:
        """Get the ML engine enabled status."""
        return self.ENABLE_ML_ENGINE

    @property
    def enable_ml_scheduler(self) -> bool:
        """Get the ML scheduler enabled status."""
        return self.ENABLE_ML_SCHEDULER

    @property
    def enable_risk_engine(self) -> bool:
        """Get the risk engine enabled status."""
        return self.ENABLE_RISK_ENGINE

    @property
    def enable_gpt_explanations(self) -> bool:
        """Get the GPT explanations enabled status."""
        return self.ENABLE_GPT_EXPLANATIONS

    @property
    def enable_notifications(self) -> bool:
        """Get the notifications enabled status."""
        return self.ENABLE_NOTIFICATIONS

    @property
    def ml_retraining_hour(self) -> int:
        """Get the ML retraining hour."""
        return self.ML_RETRAINING_HOUR

    @property
    def ml_retraining_minute(self) -> int:
        """Get the ML retraining minute."""
        return self.ML_RETRAINING_MINUTE

    @property
    def notification_check_interval_minutes(self) -> int:
        """Get the notification check interval in minutes."""
        return self.NOTIFICATION_CHECK_INTERVAL_MINUTES

    @property
    def enable_feedback_learning(self) -> bool:
        """Get feedback learning enabled status."""
        return self.ENABLE_FEEDBACK_LEARNING

    @property
    def feedback_learning_hour(self) -> int:
        """Get feedback learning hour."""
        return self.FEEDBACK_LEARNING_HOUR

    @property
    def feedback_learning_minute(self) -> int:
        """Get feedback learning minute."""
        return self.FEEDBACK_LEARNING_MINUTE

    @property
    def false_positive_threshold(self) -> int:
        """Get false positive threshold."""
        return self.FALSE_POSITIVE_THRESHOLD


# Create global settings object
settings = Settings()


def get_settings() -> Settings:
    """Dependency function for FastAPI to get settings.

    Returns:
        Settings: Application settings.
    """
    return settings


# Check for custom config file
config_file = os.getenv("APP_CONFIG_FILE")
if config_file:
    if os.path.exists(config_file):
        # Create settings from custom config file
        settings = Settings(_env_file=config_file)

# Define project base path
BASE_DIR = Path(__file__).resolve().parent.parent
