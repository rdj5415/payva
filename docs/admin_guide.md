# AuditPulse Administrator Guide

## Introduction

This guide is intended for system administrators responsible for deploying, configuring, and maintaining AuditPulse. AuditPulse is an AI-powered financial transaction monitoring and anomaly detection system designed to help businesses identify suspicious activities and maintain financial integrity.

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8GB
- **Disk**: 50GB SSD
- **Operating System**: Linux (Ubuntu 20.04 LTS or higher recommended)
- **Docker**: version 20.10 or higher
- **Docker Compose**: version 2.0 or higher

### Recommended Requirements for Production
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Disk**: 100GB+ SSD
- **Operating System**: Linux (Ubuntu 22.04 LTS recommended)
- **Docker**: version 20.10 or higher
- **Docker Compose**: version 2.0 or higher

## Installation

### Using Docker Compose (Recommended)

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/auditpulse-mvp.git
   cd auditpulse-mvp
   ```

2. Create a `.env` file by copying the example:
   ```
   cp .env.example .env
   ```

3. Edit the `.env` file to configure your environment:
   ```
   # Set strong passwords
   POSTGRES_PASSWORD=your-secure-db-password
   REDIS_PASSWORD=your-secure-redis-password
   SECRET_KEY=your-secure-secret-key
   
   # Set Plaid credentials (if using Plaid)
   PLAID_CLIENT_ID=your-plaid-client-id
   PLAID_SECRET=your-plaid-secret
   
   # Set email configuration
   SMTP_HOST=smtp.example.com
   SMTP_PORT=587
   SMTP_USER=noreply@example.com
   SMTP_PASSWORD=your-smtp-password
   EMAIL_SENDER=noreply@example.com
   ```

4. Generate SSL certificates for HTTPS (production environment):
   ```
   mkdir -p nginx/ssl
   
   # Generate self-signed certificate (replace with proper certificates in production)
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
     -keyout nginx/ssl/auditpulse.key \
     -out nginx/ssl/auditpulse.crt
   
   # For monitoring subdomains
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
     -keyout nginx/ssl/auditpulse-monitoring.key \
     -out nginx/ssl/auditpulse-monitoring.crt
   ```

5. Create password for Prometheus access:
   ```
   mkdir -p nginx/production
   htpasswd -c nginx/production/.htpasswd admin
   ```

6. Start the services:
   - For development:
     ```
     docker-compose up -d
     ```
   - For staging:
     ```
     docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d
     ```
   - For production:
     ```
     docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
     ```

7. Create admin user:
   ```
   docker-compose exec api python -m auditpulse_mvp.scripts.create_admin \
     --email admin@example.com \
     --password securepassword \
     --first-name Admin \
     --last-name User
   ```

## Configuration Options

### Database

Database settings are controlled via environment variables:

```
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-secure-password
POSTGRES_DB=auditpulse
DATABASE_URL=postgresql://postgres:your-secure-password@db:5432/auditpulse
```

### Redis

Redis settings:

```
REDIS_URL=redis://default:your-secure-password@redis:6379/0
```

### Security

Security settings:

```
SECRET_KEY=your-secure-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
```

### Email

Email notification settings:

```
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=noreply@example.com
SMTP_PASSWORD=your-smtp-password
EMAIL_SENDER=noreply@example.com
USE_SSL=False
```

### SMS (Twilio)

SMS notification settings:

```
TWILIO_ACCOUNT_SID=your-account-sid
TWILIO_AUTH_TOKEN=your-auth-token
TWILIO_PHONE_NUMBER=+1234567890
```

### Plaid Integration

Settings for Plaid API:

```
PLAID_CLIENT_ID=your-plaid-client-id
PLAID_SECRET=your-plaid-secret
PLAID_ENVIRONMENT=sandbox # sandbox, development, or production
```

## Monitoring

AuditPulse includes comprehensive monitoring capabilities:

### Prometheus

- Access URL: `https://prometheus.auditpulse.example.com`
- Username: `admin` (or as configured)
- Password: Set during installation

### Grafana

- Access URL: `https://monitoring.auditpulse.example.com`
- Default username: `admin`
- Default password: Configured in `.env` file

## Backup and Restore

### Database Backup

In the production environment, automatic backups are configured to run every day at 2 AM and are stored in the `backups` directory. The last 30 days of backups are retained.

To manually create a backup:

```
docker-compose exec db pg_dump -U postgres auditpulse | gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz
```

### Database Restore

To restore from a backup:

```
gunzip -c your-backup-file.sql.gz | docker-compose exec -T db psql -U postgres -d auditpulse
```

## Scaling

### Horizontal Scaling

To scale horizontally, increase the number of worker and API instances:

```
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale worker=4 --scale api=3
```

### Vertical Scaling

Adjust resource limits in `docker-compose.prod.yml` for vertical scaling.

## Troubleshooting

### Checking Logs

```
# API logs
docker-compose logs api

# Worker logs
docker-compose logs worker

# Database logs
docker-compose logs db

# All logs with follow
docker-compose logs -f
```

### Common Issues

1. **Database Connection Errors**
   - Check database credentials in `.env` file
   - Ensure database service is running: `docker-compose ps db`

2. **Email Sending Failures**
   - Verify SMTP settings in `.env` file
   - Check network connectivity to SMTP server

3. **High CPU/Memory Usage**
   - Check monitoring for resource bottlenecks
   - Consider scaling horizontally or vertically

4. **API Request Timeouts**
   - Check database query performance
   - Increase timeout settings in Nginx configuration

## Security Best Practices

1. **Keep Secrets Secure**
   - Never commit `.env` files to version control
   - Use a secure password manager for credentials

2. **Regular Updates**
   - Update dependencies regularly
   - Apply security patches promptly

3. **Access Control**
   - Use strong, unique passwords
   - Implement IP allowlisting for admin access

4. **SSL/TLS**
   - Use proper certificates (not self-signed) in production
   - Configure SSL/TLS with strong ciphers

5. **Firewall Configuration**
   - Only expose necessary ports
   - Use network security groups or firewall rules

## Upgrade Procedure

1. Backup the database
2. Pull the latest code from the repository
3. Check for changes to environment variables
4. Apply database migrations
5. Restart services

```
# Backup database
docker-compose exec db pg_dump -U postgres auditpulse | gzip > backup_before_upgrade.sql.gz

# Get latest code
git pull

# Apply migrations
docker-compose exec api python -m auditpulse_mvp.scripts.initialize_db

# Restart services
docker-compose -f docker-compose.yml -f docker-compose.prod.yml down
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Support

For technical support, please contact:

- Email: support@auditpulse.example.com
- Internal Documentation: https://docs.auditpulse.example.com
- GitHub Issues: https://github.com/yourusername/auditpulse-mvp/issues 