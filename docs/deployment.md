# AuditPulse Deployment Guide

This document provides comprehensive instructions for deploying the AuditPulse application in different environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Deployment Options](#deployment-options)
   - [Local Development](#local-development)
   - [Staging](#staging)
   - [Production](#production)
4. [CI/CD Pipeline](#cicd-pipeline)
5. [SSL Certificate Setup](#ssl-certificate-setup)
6. [Database Management](#database-management)
7. [Monitoring Setup](#monitoring-setup)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

Before deploying AuditPulse, ensure you have the following:

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **Git**: Latest version
- **Server Requirements**:
  - Production: 4+ CPU cores, 16GB+ RAM, 100GB+ SSD
  - Staging: 2+ CPU cores, 8GB+ RAM, 50GB+ SSD
  - Development: 2+ CPU cores, 8GB+ RAM, 20GB+ SSD

## Environment Setup

### Clone the Repository

```bash
git clone https://github.com/yourusername/auditpulse-mvp.git
cd auditpulse-mvp
```

### Environment Variables

Create a `.env` file based on the template:

```bash
cp .env.example .env
```

Edit the `.env` file to include:

```
# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=auditpulse
DATABASE_URL=postgresql://postgres:your_secure_password@db:5432/auditpulse

# Redis
REDIS_PASSWORD=your_secure_redis_password
REDIS_URL=redis://default:your_secure_redis_password@redis:6379/0

# Application
SECRET_KEY=your_secure_random_string
ENVIRONMENT=development  # development, staging, or production
CORS_ORIGINS=http://localhost:3000,https://app.example.com

# Plaid Integration
PLAID_CLIENT_ID=your_plaid_client_id
PLAID_SECRET=your_plaid_secret
PLAID_ENVIRONMENT=sandbox  # sandbox, development, or production

# Email
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=noreply@example.com
SMTP_PASSWORD=your_smtp_password
EMAIL_SENDER=noreply@example.com

# SMS (Twilio)
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=+1234567890

# Monitoring
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=your_grafana_password

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Deployment Options

### Local Development

For local development:

```bash
docker-compose up -d
```

Access the API at: http://localhost:8000

Access Swagger documentation at: http://localhost:8000/docs

### Staging

For staging deployment:

```bash
docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d
```

Additional staging-specific configurations are in `docker-compose.staging.yml`.

### Production

For production deployment:

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

Production deployment includes:
- Higher replica counts for services
- Resource limits for containers
- Automated database backups
- Enhanced monitoring
- HTTPS with SSL certificates

## CI/CD Pipeline

AuditPulse includes GitHub Actions workflows for CI/CD:

### Continuous Integration (`ci.yml`)

The CI pipeline runs on every push to `main` and `develop` branches, and on pull requests:

1. **Linting**: Runs flake8, black, and mypy
2. **Testing**: Runs unit and integration tests
3. **Security Scan**: Runs bandit and safety checks
4. **Build**: Builds and pushes Docker images

### Continuous Deployment (`cd.yml`)

The CD pipeline runs after successful CI jobs:

1. **Staging Deployment**: Automatically deploys to staging from the `develop` branch
2. **Production Deployment**: Automatically deploys to production from the `main` branch
3. **Rollback**: Automatically rolls back production if deployment fails

## SSL Certificate Setup

### Self-Signed Certificates (Development/Staging)

```bash
mkdir -p nginx/ssl

# Generate self-signed certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/auditpulse.key \
  -out nginx/ssl/auditpulse.crt
```

### Let's Encrypt (Production)

For production, use Let's Encrypt:

```bash
# Install certbot
apt-get update
apt-get install -y certbot python3-certbot-nginx

# Obtain certificate
certbot --nginx -d app.auditpulse.example.com -d monitoring.auditpulse.example.com -d prometheus.auditpulse.example.com

# Auto-renewal
certbot renew --dry-run
```

Add to crontab for automatic renewal:

```
0 3 * * * /usr/bin/certbot renew --quiet
```

## Database Management

### Initialize Database

```bash
docker-compose exec api python -m auditpulse_mvp.scripts.initialize_db
```

### Database Migrations

After schema changes:

```bash
# Generate migration
docker-compose exec api alembic revision --autogenerate -m "Description of changes"

# Apply migration
docker-compose exec api alembic upgrade head
```

### Database Backup and Restore

Manual backup:

```bash
docker-compose exec db pg_dump -U postgres auditpulse | gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz
```

Restore from backup:

```bash
gunzip -c your-backup-file.sql.gz | docker-compose exec -T db psql -U postgres -d auditpulse
```

## Monitoring Setup

AuditPulse includes Prometheus and Grafana for monitoring.

### Access Monitoring

- Grafana: https://monitoring.auditpulse.example.com
  - Default username: admin
  - Default password: Set in `.env` file (GF_SECURITY_ADMIN_PASSWORD)

- Prometheus: https://prometheus.auditpulse.example.com
  - Protected with basic authentication (set during deployment)

### Alerting

To set up alerting in Grafana:

1. Log in to Grafana
2. Go to Alerting > Notification channels
3. Add channels for Slack, Email, PagerDuty, etc.
4. Create alert rules in dashboards

## Scaling

### Horizontal Scaling

To scale services horizontally:

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale api=5 --scale worker=3
```

### Vertical Scaling

To adjust resource limits, modify the Docker Compose files:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

## Troubleshooting

### Check Container Logs

```bash
# Check logs for specific service
docker-compose logs api

# Follow logs
docker-compose logs -f

# Check last 100 lines
docker-compose logs --tail=100 api
```

### Common Issues

#### Database Connection Failures

```bash
# Check if database is running
docker-compose ps db

# Check database logs
docker-compose logs db

# Verify connection string in .env
cat .env | grep DATABASE_URL
```

#### API Errors

```bash
# Check API logs
docker-compose logs api

# Inspect API container
docker-compose exec api bash

# Test database connection from inside container
python -c "import psycopg2; conn = psycopg2.connect('$DATABASE_URL'); print('Connected!')"
```

#### Nginx Issues

```bash
# Check Nginx configuration
docker-compose exec nginx nginx -t

# Check Nginx logs
docker-compose logs nginx
```

#### Monitoring Issues

```bash
# Check Prometheus configuration
docker-compose exec prometheus promtool check config /etc/prometheus/prometheus.yml

# Check targets in Prometheus
curl -s http://localhost:9090/api/v1/targets | jq .
```

## Additional Resources

- [AuditPulse Admin Guide](./admin_guide.md)
- [AuditPulse User Guide](./user_guide.md)
- [AuditPulse API Documentation](./api_documentation.md)
- [Docker Documentation](https://docs.docker.com/)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)
- [Grafana Documentation](https://grafana.com/docs/) 