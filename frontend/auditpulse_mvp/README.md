# AuditPulse AI MVP

AuditPulse AI is a financial transaction anomaly detection platform that uses rules-based analysis, machine learning, and AI to detect and explain potentially fraudulent or irregular financial activities.

## Features

- **Data Ingestion**: Seamless integration with QuickBooks and Plaid
- **Anomaly Detection**: Multi-layer approach combining rules-based checks, ML models, and GPT-powered analysis
- **Interactive Dashboard**: View and manage detected anomalies in a user-friendly interface
- **Notifications**: Get alerts for high-risk transactions via email, Slack, or SMS
- **Continuous Learning**: System improves over time based on user feedback
- **Multi-tenancy**: Secure data isolation between different organizations
- **Compliance Ready**: Built with SOC 2 and other regulatory requirements in mind

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL
- Redis

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/auditpulse_mvp.git
cd auditpulse_mvp
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize the database:

```bash
alembic upgrade head
```

6. Run the application:

```bash
uvicorn api.main:app --reload
```

## Development

### Code Standards

This project uses:
- Black for code formatting
- Flake8 for linting
- MyPy for type checking
- Bandit for security scanning
- Pre-commit hooks to enforce standards

### Setting Up Development Environment

1. Install pre-commit hooks:

```bash
pre-commit install
```

2. Run tests:

```bash
pytest
```

### Project Structure

- `api/`: FastAPI application and endpoints
- `database/`: Database models and migrations
- `ingestion/`: Data connectors for QuickBooks, Plaid, etc.
- `rules_engine/`: Rules-based anomaly detection
- `ml_engine/`: Machine learning models for anomaly detection
- `gpt_engine/`: GPT integration for explaining anomalies
- `dashboard/`: Streamlit dashboard
- `alerts/`: Notification services
- `tasks/`: Background task scheduling
- `admin/`: Admin interfaces and tools
- `utils/`: Shared utilities
- `tests/`: Test suite

## License

[MIT License](LICENSE)

## Contact

For questions or support, contact [info@auditpulse.ai](mailto:info@auditpulse.ai) 