# AuditPulse AI SDK

The AuditPulse AI SDK provides a Python client library for interacting with the AuditPulse AI API.

## Installation

```bash
pip install auditpulse-ai
```

## Usage

### Basic Usage

```python
from auditpulse_mvp.sdk import AuditPulseClient

# Initialize client
client = AuditPulseClient(
    base_url="https://api.auditpulse.ai",
    api_key="your_api_key"
)

# Login
async with client:
    # Get transactions
    transactions = await client.get_transactions(
        tenant_id="your_tenant_id",
        start_date=datetime.now() - timedelta(days=30)
    )
    
    # Get anomalies
    anomalies = await client.get_anomalies(
        tenant_id="your_tenant_id",
        status="open"
    )
    
    # Update anomaly
    updated_anomaly = await client.update_anomaly(
        anomaly_id="anomaly_id",
        status="resolved",
        resolution="False positive"
    )
    
    # Submit feedback
    feedback = await client.submit_feedback(
        anomaly_id="anomaly_id",
        feedback={
            "type": "false_positive",
            "comment": "This is a false positive",
            "rating": 5
        }
    )
```

### Authentication

The SDK supports two authentication methods:

1. API Key:
```python
client = AuditPulseClient(
    base_url="https://api.auditpulse.ai",
    api_key="your_api_key"
)
```

2. JWT Token:
```python
client = AuditPulseClient(
    base_url="https://api.auditpulse.ai",
    token="your_jwt_token"
)
```

### Error Handling

The SDK raises `AuditPulseError` for API errors:

```python
from auditpulse_mvp.sdk import AuditPulseError

try:
    await client.get_transactions(tenant_id="invalid_id")
except AuditPulseError as e:
    print(f"Error: {e.code} - {e.message}")
    print(f"Details: {e.details}")
```

### Pagination

The SDK supports pagination for list endpoints:

```python
# Get first page
page1 = await client.get_transactions(
    tenant_id="your_tenant_id",
    page=1,
    size=100
)

# Get next page
page2 = await client.get_transactions(
    tenant_id="your_tenant_id",
    page=2,
    size=100
)
```

### Models

The SDK provides Pydantic models for data validation:

```python
from auditpulse_mvp.sdk import Transaction, Anomaly, RiskLevel

# Create transaction
transaction = Transaction(
    id=uuid.uuid4(),
    tenant_id=uuid.uuid4(),
    transaction_id="txn_001",
    amount=1000.0,
    currency="USD",
    description="Office supplies",
    category="Expenses",
    merchant_name="Office Depot",
    transaction_date=datetime.now(),
    source="quickbooks",
    source_account_id="acc_001"
)

# Create anomaly
anomaly = Anomaly(
    id=uuid.uuid4(),
    tenant_id=uuid.uuid4(),
    transaction_id="txn_001",
    type="large_amount",
    risk_score=0.8,
    risk_level=RiskLevel.HIGH,
    amount=1000.0,
    description="Unusually large transaction",
    status="open"
)
```

## Development

### Running Tests

```bash
pytest tests/
```

### Building Documentation

```bash
sphinx-build docs/ docs/_build/
```

## License

MIT License - see LICENSE file for details 