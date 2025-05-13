"""Tenant and user administration classes.

This module provides classes for managing tenants and users in the admin interface.
"""

import logging
from typing import Any, Dict, List, Optional, Type

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import UUID4, EmailStr

from auditpulse_mvp.database.models import Tenant, User
from auditpulse_mvp.schemas.admin import TenantCreate, TenantRead, TenantUpdate
from auditpulse_mvp.schemas.admin import UserCreate, UserRead, UserUpdate
from auditpulse_mvp.utils.security import get_password_hash, verify_password


# Configure logging
logger = logging.getLogger(__name__)


class TenantAdmin:
    """Admin class for managing tenants."""

    def __init__(self, db_session: AsyncSession):
        """Initialize the tenant admin.

        Args:
            db_session: Database session.
        """
        self.db = db_session

    async def get_all(self, skip: int = 0, limit: int = 100) -> List[Tenant]:
        """Get all tenants with pagination.

        Args:
            skip: Number of records to skip.
            limit: Maximum number of records to return.

        Returns:
            List of tenants.
        """
        stmt = select(Tenant).offset(skip).limit(limit)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_by_id(self, tenant_id: UUID4) -> Optional[Tenant]:
        """Get a tenant by ID.

        Args:
            tenant_id: The tenant ID.

        Returns:
            The tenant or None if not found.
        """
        stmt = select(Tenant).where(Tenant.id == tenant_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_slug(self, slug: str) -> Optional[Tenant]:
        """Get a tenant by slug.

        Args:
            slug: The tenant slug.

        Returns:
            The tenant or None if not found.
        """
        stmt = select(Tenant).where(Tenant.slug == slug)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def create(self, tenant_data: TenantCreate) -> Tenant:
        """Create a new tenant.

        Args:
            tenant_data: The tenant creation data.

        Returns:
            The created tenant.

        Raises:
            ValueError: If a tenant with the same slug already exists.
        """
        # Check if tenant with same slug already exists
        existing = await self.get_by_slug(tenant_data.slug)
        if existing:
            raise ValueError(f"Tenant with slug '{tenant_data.slug}' already exists")

        # Create tenant
        db_tenant = Tenant(**tenant_data.dict())
        self.db.add(db_tenant)
        await self.db.commit()
        await self.db.refresh(db_tenant)

        logger.info(f"Created tenant: {db_tenant.name} (ID: {db_tenant.id})")
        return db_tenant

    async def update(
        self, tenant_id: UUID4, tenant_data: TenantUpdate
    ) -> Optional[Tenant]:
        """Update a tenant.

        Args:
            tenant_id: The tenant ID.
            tenant_data: The tenant update data.

        Returns:
            The updated tenant or None if not found.
        """
        db_tenant = await self.get_by_id(tenant_id)
        if not db_tenant:
            return None

        # Update tenant attributes
        update_data = tenant_data.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_tenant, key, value)

        await self.db.commit()
        await self.db.refresh(db_tenant)

        logger.info(f"Updated tenant: {db_tenant.name} (ID: {db_tenant.id})")
        return db_tenant

    async def delete(self, tenant_id: UUID4) -> bool:
        """Delete a tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            True if deleted, False if not found.
        """
        db_tenant = await self.get_by_id(tenant_id)
        if not db_tenant:
            return False

        # Instead of hard deleting, mark as inactive and renamed
        db_tenant.is_active = False
        db_tenant.name = f"DELETED_{db_tenant.name}"
        db_tenant.slug = f"deleted_{db_tenant.slug}_{db_tenant.id}"

        await self.db.commit()

        logger.info(f"Marked tenant as deleted: {tenant_id}")
        return True

    async def count_active(self) -> int:
        """Count active tenants.

        Returns:
            Number of active tenants.
        """
        stmt = select(Tenant).where(Tenant.is_active == True)
        result = await self.db.execute(stmt)
        return len(list(result.scalars().all()))


class UserAdmin:
    """Admin class for managing users."""

    def __init__(self, db_session: AsyncSession):
        """Initialize the user admin.

        Args:
            db_session: Database session.
        """
        self.db = db_session

    async def get_all(self, skip: int = 0, limit: int = 100) -> List[User]:
        """Get all users with pagination.

        Args:
            skip: Number of records to skip.
            limit: Maximum number of records to return.

        Returns:
            List of users.
        """
        stmt = select(User).offset(skip).limit(limit)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_by_id(self, user_id: UUID4) -> Optional[User]:
        """Get a user by ID.

        Args:
            user_id: The user ID.

        Returns:
            The user or None if not found.
        """
        stmt = select(User).where(User.id == user_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_email(self, email: EmailStr) -> Optional[User]:
        """Get a user by email.

        Args:
            email: The user email.

        Returns:
            The user or None if not found.
        """
        stmt = select(User).where(User.email == email)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_tenant_users(
        self, tenant_id: UUID4, skip: int = 0, limit: int = 100
    ) -> List[User]:
        """Get all users for a tenant with pagination.

        Args:
            tenant_id: The tenant ID.
            skip: Number of records to skip.
            limit: Maximum number of records to return.

        Returns:
            List of users.
        """
        stmt = select(User).where(User.tenant_id == tenant_id).offset(skip).limit(limit)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def create(self, user_data: UserCreate, tenant_id: UUID4) -> User:
        """Create a new user.

        Args:
            user_data: The user creation data.
            tenant_id: The tenant ID for the user.

        Returns:
            The created user.

        Raises:
            ValueError: If a user with the same email already exists.
        """
        # Check if user with same email already exists
        existing = await self.get_by_email(user_data.email)
        if existing:
            raise ValueError(f"User with email '{user_data.email}' already exists")

        # Create user with tenant_id
        user_dict = user_data.dict()
        password = user_dict.pop("password")

        db_user = User(
            **user_dict,
            tenant_id=tenant_id,
            hashed_password=get_password_hash(password),
        )

        self.db.add(db_user)
        await self.db.commit()
        await self.db.refresh(db_user)

        logger.info(f"Created user: {db_user.email} (ID: {db_user.id})")
        return db_user

    async def update(self, user_id: UUID4, user_data: UserUpdate) -> Optional[User]:
        """Update a user.

        Args:
            user_id: The user ID.
            user_data: The user update data.

        Returns:
            The updated user or None if not found.
        """
        db_user = await self.get_by_id(user_id)
        if not db_user:
            return None

        # Update user attributes
        update_data = user_data.dict(exclude_unset=True, exclude={"password"})
        for key, value in update_data.items():
            setattr(db_user, key, value)

        # Update password if provided
        if user_data.password:
            db_user.hashed_password = get_password_hash(user_data.password)

        await self.db.commit()
        await self.db.refresh(db_user)

        logger.info(f"Updated user: {db_user.email} (ID: {db_user.id})")
        return db_user

    async def deactivate(self, user_id: UUID4) -> bool:
        """Deactivate a user.

        Args:
            user_id: The user ID.

        Returns:
            True if deactivated, False if not found.
        """
        db_user = await self.get_by_id(user_id)
        if not db_user:
            return False

        db_user.is_active = False
        await self.db.commit()

        logger.info(f"Deactivated user: {db_user.email} (ID: {db_user.id})")
        return True

    async def count_by_tenant(self, tenant_id: UUID4) -> int:
        """Count users for a tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            Number of users.
        """
        stmt = select(User).where(User.tenant_id == tenant_id)
        result = await self.db.execute(stmt)
        return len(list(result.scalars().all()))
