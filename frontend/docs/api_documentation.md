# AuditPulse API Documentation

## Overview

The AuditPulse API provides programmatic access to the AuditPulse financial transaction monitoring and anomaly detection system. This RESTful API allows developers to integrate AuditPulse's powerful anomaly detection capabilities into their applications, access financial data, and manage user accounts.

**Base URL**: `https://api.auditpulse.example.com/api`

**API Versions**:
- V1: `/api/v1` - Current stable version

## Authentication

All API requests require authentication using JWT (JSON Web Tokens).

### Obtaining Access Tokens

```
POST /api/v1/auth/login
```

**Request Body**:
```json
{
  "email": "user@example.com",
  "password": "your_password"
}
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Using Access Tokens

Include the access token in the Authorization header of all API requests:

```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Refreshing Tokens

```
POST /api/v1/auth/refresh
```

**Request Body**:
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response**: Same as login response.

## Error Handling

The API uses standard HTTP status codes to indicate the success or failure of requests.

Common error codes:
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Invalid or expired authentication credentials
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Input validation failed
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server-side error

Error responses follow this format:
```json
{
  "detail": "Error message describing the issue",
  "status_code": 400,
  "error_code": "INVALID_REQUEST"
}
```

## Rate Limiting

Requests are limited to 100 requests per minute per API key. Rate limit information is included in response headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 99
X-RateLimit-Reset: 1620000000
```

When the rate limit is exceeded, a 429 error is returned.

## Endpoints

### User Management

#### Get Current User

```
GET /api/v1/users/me
```

**Response**:
```json
{
  "id": "user_123456",
  "email": "user@example.com",
  "first_name": "Jane",
  "last_name": "Doe",
  "is_active": true,
  "created_at": "2023-01-15T12:00:00Z",
  "last_login": "2023-05-10T15:30:00Z"
}
```

#### Update User Profile

```
PUT /api/v1/users/me
```

**Request Body**:
```json
{
  "first_name": "Jane",
  "last_name": "Smith",
  "phone_number": "+1234567890"
}
```

**Response**: Updated user object.

### Financial Accounts

#### List Connected Accounts

```
GET /api/v1/accounts
```

**Query Parameters**:
- `status`: Filter by account status (`active`, `disconnected`, `error`)
- `institution_id`: Filter by institution ID
- `limit`: Maximum number of results (default: 50)
- `offset`: Result offset for pagination

**Response**:
```json
{
  "total": 3,
  "accounts": [
    {
      "id": "acc_123456",
      "institution_id": "ins_chase",
      "institution_name": "Chase",
      "name": "Chase Checking",
      "mask": "1234",
      "type": "checking",
      "subtype": "personal",
      "current_balance": 5000.25,
      "available_balance": 4800.00,
      "currency_code": "USD",
      "status": "active",
      "last_updated": "2023-05-10T16:30:00Z"
    },
    // Additional accounts...
  ]
}
```

#### Get Account Details

```
GET /api/v1/accounts/{account_id}
```

**Response**: Detailed account object.

#### Update Account Settings

```
PATCH /api/v1/accounts/{account_id}
```

**Request Body**:
```json
{
  "nickname": "Primary Checking",
  "is_hidden": false,
  "notification_threshold": 5000
}
```

**Response**: Updated account object.

### Transactions

#### List Transactions

```
GET /api/v1/transactions
```

**Query Parameters**:
- `account_id`: Filter by account
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD)
- `category`: Filter by category
- `min_amount`: Minimum transaction amount
- `max_amount`: Maximum transaction amount
- `status`: Transaction status
- `limit`: Maximum number of results (default: 50)
- `offset`: Result offset for pagination

**Response**:
```json
{
  "total": 120,
  "transactions": [
    {
      "id": "trx_123456",
      "account_id": "acc_123456",
      "amount": -75.50,
      "date": "2023-05-09",
      "description": "WHOLE FOODS #123",
      "category": "groceries",
      "location": {
        "address": "123 Main St",
        "city": "San Francisco",
        "region": "CA",
        "postal_code": "94107",
        "country": "US",
        "lat": 37.7749,
        "lon": -122.4194
      },
      "pending": false,
      "transaction_type": "purchase",
      "merchant_name": "Whole Foods",
      "merchant_id": "merch_wholefds",
      "tags": ["food", "organic"],
      "notes": "Weekly grocery shopping"
    },
    // Additional transactions...
  ]
}
```

#### Get Transaction Details

```
GET /api/v1/transactions/{transaction_id}
```

**Response**: Detailed transaction object.

#### Update Transaction Metadata

```
PATCH /api/v1/transactions/{transaction_id}
```

**Request Body**:
```json
{
  "category": "dining",
  "tags": ["business", "client meeting"],
  "notes": "Lunch with client"
}
```

**Response**: Updated transaction object.

### Anomalies

#### List Anomalies

```
GET /api/v1/anomalies
```

**Query Parameters**:
- `status`: Filter by status (`open`, `reviewed`, `dismissed`, `flagged`)
- `severity`: Filter by severity (`low`, `medium`, `high`, `critical`)
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD)
- `account_id`: Filter by account
- `limit`: Maximum number of results (default: 50)
- `offset`: Result offset for pagination

**Response**:
```json
{
  "total": 12,
  "anomalies": [
    {
      "id": "anom_123456",
      "transaction_id": "trx_123456",
      "account_id": "acc_123456",
      "detection_date": "2023-05-10T14:30:00Z",
      "severity": "high",
      "type": "unusual_amount",
      "status": "open",
      "confidence_score": 0.92,
      "description": "Transaction amount 10x higher than typical spending in this category",
      "factors": [
        {
          "name": "amount_deviation",
          "value": 10.35,
          "description": "Amount is 10.35x higher than your average spending for groceries"
        },
        {
          "name": "frequency_unusual",
          "value": true,
          "description": "You typically don't shop at this merchant on Mondays"
        }
      ],
      "similar_anomalies": [
        "anom_789012",
        "anom_345678"
      ]
    },
    // Additional anomalies...
  ]
}
```

#### Get Anomaly Details

```
GET /api/v1/anomalies/{anomaly_id}
```

**Response**: Detailed anomaly object.

#### Update Anomaly Status

```
PATCH /api/v1/anomalies/{anomaly_id}
```

**Request Body**:
```json
{
  "status": "dismissed",
  "resolution_notes": "Authorized large purchase"
}
```

**Response**: Updated anomaly object.

### Models

#### List Model Versions

```
GET /api/v1/models/{model_type}/versions
```

**Response**:
```json
{
  "model_type": "transaction_anomaly",
  "active_version": "v1.2.3",
  "versions": [
    {
      "version": "v1.2.3",
      "created_at": "2023-04-01T12:00:00Z",
      "is_active": true,
      "created_by": "user_admin",
      "description": "Improved detection of unusual amount patterns"
    },
    {
      "version": "v1.1.0",
      "created_at": "2023-03-15T10:30:00Z",
      "is_active": false,
      "created_by": "user_admin",
      "description": "Initial production release"
    }
  ]
}
```

#### Validate Model

```
POST /api/v1/models/{model_type}/validate
```

**Request Body**:
```json
{
  "version": "v1.2.3",
  "validation_data": [
    {
      "input": {
        "amount": 1500,
        "category": "dining",
        "merchant": "EXPENSIVE RESTAURANT",
        "date": "2023-05-09T19:30:00Z",
        "user_id": "user_123456"
      }
    },
    // Additional validation examples...
  ],
  "ground_truth": [
    {
      "is_anomaly": true,
      "anomaly_type": "unusual_amount"
    },
    // Additional ground truth labels...
  ]
}
```

**Response**:
```json
{
  "model_type": "transaction_anomaly",
  "version": "v1.2.3",
  "metrics": {
    "accuracy": 0.92,
    "precision": 0.89,
    "recall": 0.94,
    "f1_score": 0.91
  },
  "validation_size": 100,
  "execution_time_ms": 350,
  "validation_success": true
}
```

### Plaid Integration

#### Create Link Token

```
POST /api/v1/plaid/link/token/create
```

**Request Body**:
```json
{
  "redirect_uri": "https://app.auditpulse.example.com/link-callback"
}
```

**Response**:
```json
{
  "link_token": "link-sandbox-12345...",
  "expiration": "2023-05-10T17:30:00Z"
}
```

#### Exchange Public Token

```
POST /api/v1/plaid/item/public_token/exchange
```

**Request Body**:
```json
{
  "public_token": "public-sandbox-12345..."
}
```

**Response**:
```json
{
  "success": true,
  "accounts_connected": 3
}
```

### Reports

#### Generate Report

```
POST /api/v1/reports
```

**Request Body**:
```json
{
  "report_type": "monthly_summary",
  "parameters": {
    "month": "2023-04",
    "accounts": ["acc_123456", "acc_789012"],
    "include_categories": true,
    "include_anomalies": true
  },
  "format": "pdf"
}
```

**Response**:
```json
{
  "report_id": "rep_123456",
  "status": "processing",
  "estimated_completion_time": "2023-05-10T16:35:00Z"
}
```

#### Get Report Status

```
GET /api/v1/reports/{report_id}
```

**Response**:
```json
{
  "report_id": "rep_123456",
  "status": "completed",
  "created_at": "2023-05-10T16:30:00Z",
  "completed_at": "2023-05-10T16:32:00Z",
  "download_url": "https://api.auditpulse.example.com/api/v1/reports/rep_123456/download",
  "expires_at": "2023-05-17T16:32:00Z"
}
```

### Admin Endpoints

*Note: These endpoints require admin privileges.*

#### List Users

```
GET /api/v1/admin/users
```

**Query Parameters**:
- `status`: Filter by status (`active`, `inactive`, `pending`)
- `created_after`: Filter by creation date
- `created_before`: Filter by creation date
- `search`: Search by name or email
- `limit`: Maximum number of results (default: 50)
- `offset`: Result offset for pagination

**Response**:
```json
{
  "total": 250,
  "users": [
    {
      "id": "user_123456",
      "email": "user@example.com",
      "first_name": "Jane",
      "last_name": "Doe",
      "is_active": true,
      "role": "user",
      "created_at": "2023-01-15T12:00:00Z",
      "last_login": "2023-05-10T15:30:00Z",
      "account_count": 3
    },
    // Additional users...
  ]
}
```

#### System Health Check

```
GET /api/v1/admin/health
```

**Response**:
```json
{
  "status": "healthy",
  "components": {
    "api": {
      "status": "healthy",
      "version": "1.0.0",
      "uptime": 345600
    },
    "database": {
      "status": "healthy",
      "connection_count": 12,
      "latency_ms": 3
    },
    "redis": {
      "status": "healthy",
      "used_memory": "1.2GB",
      "used_memory_peak": "1.5GB"
    },
    "storage": {
      "status": "healthy",
      "available_space": "500GB"
    }
  },
  "timestamp": "2023-05-10T16:30:00Z"
}
```

#### Configure Model Settings

```
PATCH /api/v1/admin/models/{model_type}/settings
```

**Request Body**:
```json
{
  "threshold": 0.85,
  "min_confidence_score": 0.7,
  "max_alerts_per_day": 10,
  "enable_feedback_loop": true
}
```

**Response**:
```json
{
  "model_type": "transaction_anomaly",
  "settings": {
    "threshold": 0.85,
    "min_confidence_score": 0.7,
    "max_alerts_per_day": 10,
    "enable_feedback_loop": true,
    "last_updated": "2023-05-10T16:30:00Z",
    "updated_by": "admin_user"
  }
}
```

## Webhooks

AuditPulse can send webhooks to notify your application of events.

### Webhook Authentication

Webhooks include a signature in the `X-AuditPulse-Signature` header. Verify this signature to ensure the webhook is from AuditPulse.

### Webhook Events

#### Transaction Created

```json
{
  "event_type": "transaction.created",
  "event_id": "evt_123456",
  "created_at": "2023-05-10T16:30:00Z",
  "data": {
    "transaction_id": "trx_123456",
    "account_id": "acc_123456",
    "amount": -75.50,
    "date": "2023-05-09",
    "description": "WHOLE FOODS #123"
  }
}
```

#### Anomaly Detected

```json
{
  "event_type": "anomaly.detected",
  "event_id": "evt_789012",
  "created_at": "2023-05-10T16:30:00Z",
  "data": {
    "anomaly_id": "anom_123456",
    "transaction_id": "trx_123456",
    "account_id": "acc_123456",
    "severity": "high",
    "type": "unusual_amount"
  }
}
```

#### Account Status Changed

```json
{
  "event_type": "account.status_changed",
  "event_id": "evt_345678",
  "created_at": "2023-05-10T16:30:00Z",
  "data": {
    "account_id": "acc_123456",
    "previous_status": "active",
    "current_status": "error",
    "reason": "login_required"
  }
}
```

## SDK Libraries

AuditPulse provides client libraries for easy integration:

- [JavaScript SDK](https://github.com/auditpulse/auditpulse-js)
- [Python SDK](https://github.com/auditpulse/auditpulse-python)
- [Java SDK](https://github.com/auditpulse/auditpulse-java)
- [Ruby SDK](https://github.com/auditpulse/auditpulse-ruby)

## Postman Collection

Download our [Postman Collection](https://api.auditpulse.example.com/postman) to quickly get started with the API.

## Support

For API support, please contact:

- Email: api-support@auditpulse.example.com
- Documentation: https://docs.auditpulse.example.com/api
- Status Page: https://status.auditpulse.example.com 