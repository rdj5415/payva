"""Script to create the initial admin user.

This script creates the first admin user in the system.
It should be run after database migrations.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from auditpulse_mvp.database.models import User, UserRole
from auditpulse_mvp.database.session import async_session_maker
from auditpulse_mvp.core.security import get_password_hash


async def create_admin_user(
    email: str,
    password: str,
    first_name: str,
    last_name: str,
) -> None:
    """Create the initial admin user.

    Args:
        email: Admin user's email
        password: Admin user's password
        first_name: Admin user's first name
        last_name: Admin user's last name
    """
    async with async_session_maker() as session:
        # Check if admin user already exists
        existing_admin = await session.get(User, email)
        if existing_admin:
            print(f"Admin user {email} already exists")
            return

        # Create admin user
        admin = User(
            email=email,
            hashed_password=get_password_hash(password),
            first_name=first_name,
            last_name=last_name,
            role=UserRole.ADMIN,
            is_active=True,
        )
        session.add(admin)
        await session.commit()
        print(f"Created admin user {email}")


async def main() -> None:
    """Main function to create admin user."""
    # Get admin user details from environment variables
    email = os.getenv("ADMIN_EMAIL", "admin@auditpulse.ai")
    password = os.getenv("ADMIN_PASSWORD", "admin123")  # Change in production
    first_name = os.getenv("ADMIN_FIRST_NAME", "Admin")
    last_name = os.getenv("ADMIN_LAST_NAME", "User")

    # Create admin user
    await create_admin_user(email, password, first_name, last_name)


if __name__ == "__main__":
    asyncio.run(main()) 