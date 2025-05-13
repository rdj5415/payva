"""Notification template management system."""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from jinja2 import Template, Environment, PackageLoader, select_autoescape

from auditpulse_mvp.database.models import (
    NotificationTemplate as DBNotificationTemplate,
)
from auditpulse_mvp.database.session import get_db
from auditpulse_mvp.utils.settings import Settings

logger = logging.getLogger(__name__)


class TemplateManager:
    """Manager for notification templates."""

    def __init__(self, settings: Settings):
        """Initialize the template manager.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.template_cache = {}
        self.jinja_env = Environment(
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    async def get_template(
        self, template_id: str, refresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get a template by ID.

        Args:
            template_id: Template ID
            refresh: Whether to refresh the cache

        Returns:
            Optional[Dict[str, Any]]: Template data if found
        """
        if not refresh and template_id in self.template_cache:
            return self.template_cache[template_id]

        async with get_db() as db:
            template = (
                await db.query(DBNotificationTemplate)
                .filter(DBNotificationTemplate.template_id == template_id)
                .first()
            )

            if not template:
                logger.warning(f"Template not found: {template_id}")
                return None

            template_data = {
                "template_id": template.template_id,
                "name": template.name,
                "description": template.description,
                "subject": template.subject,
                "body": template.body,
                "html_body": template.html_body,
                "placeholders": (
                    json.loads(template.placeholders) if template.placeholders else {}
                ),
                "created_at": template.created_at,
                "updated_at": template.updated_at,
                "version": template.version,
            }

            self.template_cache[template_id] = template_data
            return template_data

    async def render_template(
        self,
        template_id: str,
        template_data: Dict[str, Any],
        include_html: bool = True,
    ) -> Dict[str, str]:
        """Render a template with the provided data.

        Args:
            template_id: Template ID
            template_data: Data for template rendering
            include_html: Whether to include HTML version

        Returns:
            Dict[str, str]: Rendered template with subject and body
        """
        template = await self.get_template(template_id)

        if not template:
            logger.error(f"Cannot render template, not found: {template_id}")
            return {
                "subject": f"Notification from AuditPulse",
                "body": f"(Template {template_id} not found)",
                "html_body": (
                    f"<p>(Template {template_id} not found)</p>"
                    if include_html
                    else None
                ),
            }

        try:
            subject_template = self.jinja_env.from_string(template["subject"])
            body_template = self.jinja_env.from_string(template["body"])

            result = {
                "subject": subject_template.render(**template_data),
                "body": body_template.render(**template_data),
            }

            if include_html and template.get("html_body"):
                html_template = self.jinja_env.from_string(template["html_body"])
                result["html_body"] = html_template.render(**template_data)

            return result

        except Exception as e:
            logger.error(f"Error rendering template {template_id}: {e}")
            return {
                "subject": f"Notification from AuditPulse",
                "body": f"An error occurred while rendering the notification template.",
                "html_body": (
                    f"<p>An error occurred while rendering the notification template.</p>"
                    if include_html
                    else None
                ),
            }

    async def create_template(
        self,
        template_id: str,
        name: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        description: Optional[str] = None,
        placeholders: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create or update a notification template.

        Args:
            template_id: Template ID
            name: Template name
            subject: Template subject
            body: Template body
            html_body: Optional HTML body
            description: Optional description
            placeholders: Optional placeholder descriptions

        Returns:
            Dict[str, Any]: Created or updated template
        """
        async with get_db() as db:
            # Check if template exists
            template = (
                await db.query(DBNotificationTemplate)
                .filter(DBNotificationTemplate.template_id == template_id)
                .first()
            )

            if template:
                # Update existing template
                template.name = name
                template.subject = subject
                template.body = body
                template.html_body = html_body
                template.description = description
                template.placeholders = (
                    json.dumps(placeholders) if placeholders else None
                )
                template.updated_at = datetime.utcnow()
                template.version += 1
            else:
                # Create new template
                template = DBNotificationTemplate(
                    template_id=template_id,
                    name=name,
                    subject=subject,
                    body=body,
                    html_body=html_body,
                    description=description,
                    placeholders=json.dumps(placeholders) if placeholders else None,
                    version=1,
                )
                db.add(template)

            await db.commit()
            await db.refresh(template)

            # Update cache
            template_data = {
                "template_id": template.template_id,
                "name": template.name,
                "description": template.description,
                "subject": template.subject,
                "body": template.body,
                "html_body": template.html_body,
                "placeholders": (
                    json.loads(template.placeholders) if template.placeholders else {}
                ),
                "created_at": template.created_at,
                "updated_at": template.updated_at,
                "version": template.version,
            }

            self.template_cache[template_id] = template_data
            return template_data

    async def list_templates(self) -> List[Dict[str, Any]]:
        """List all notification templates.

        Returns:
            List[Dict[str, Any]]: List of templates
        """
        async with get_db() as db:
            templates = await db.query(DBNotificationTemplate).all()

            result = []
            for template in templates:
                template_data = {
                    "template_id": template.template_id,
                    "name": template.name,
                    "description": template.description,
                    "subject": template.subject,
                    "created_at": template.created_at,
                    "updated_at": template.updated_at,
                    "version": template.version,
                }
                result.append(template_data)

            return result

    async def delete_template(self, template_id: str) -> bool:
        """Delete a notification template.

        Args:
            template_id: Template ID

        Returns:
            bool: Whether the template was deleted
        """
        async with get_db() as db:
            template = (
                await db.query(DBNotificationTemplate)
                .filter(DBNotificationTemplate.template_id == template_id)
                .first()
            )

            if not template:
                return False

            await db.delete(template)
            await db.commit()

            # Remove from cache
            if template_id in self.template_cache:
                del self.template_cache[template_id]

            return True
