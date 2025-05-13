# AuditPulse AI Deployment

This directory contains deployment configurations for AuditPulse AI.

## Prerequisites

- Docker and Docker Compose
- PostgreSQL 15
- Redis 7
- Python 3.11+

## Environment Variables

Create a `.env` file in the deployment directory with the following variables:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/auditpulse

# API
JWT_SECRET=your_jwt_secret_here
OPENAI_API_KEY=your_openai_api_key_here

# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password_here

# Slack
SLACK_WEBHOOK_URL=your_slack_webhook_url_here
```

## Deployment

1. Build and start the services:
```bash
docker-compose up -d
```

2. Run database migrations:
```bash
docker-compose exec api alembic upgrade head
```

3. Create initial admin user:
```bash
docker-compose exec api python -m auditpulse_mvp.scripts.create_admin
```

## Services

- **API**: FastAPI backend service (port 8000)
- **Dashboard**: Streamlit dashboard (port 8501)
- **Database**: PostgreSQL database (port 5432)
- **Redis**: Task queue (port 6379)
- **Worker**: Background task processor

## Monitoring

- API logs: `docker-compose logs -f api`
- Dashboard logs: `docker-compose logs -f dashboard`
- Worker logs: `docker-compose logs -f worker`

## Scaling

To scale the worker service:
```bash
docker-compose up -d --scale worker=3
```

## Backup

1. Database backup:
```bash
docker-compose exec db pg_dump -U postgres auditpulse > backup.sql
```

2. Restore database:
```bash
docker-compose exec -T db psql -U postgres auditpulse < backup.sql
```

## Security

- All services run as non-root users
- Environment variables for sensitive data
- HTTPS recommended for production
- Regular security updates
- Security headers are enforced for all API responses
- Rate limiting is enabled (default: 100 requests/minute per IP)
- CORS is configurable and enabled
- All input is validated using Pydantic models
- Use strong, unique secrets for all keys and tokens
- Regularly review and update dependencies

## Troubleshooting

1. Check service status:
```