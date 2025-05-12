# AuditPulse AI Dashboard

The AuditPulse AI Dashboard is a Streamlit-based interface for monitoring and managing financial anomalies in your transactions.

## Features

- Real-time transaction monitoring
- Anomaly detection and analysis
- Risk scoring and visualization
- User feedback collection
- Notification management
- Multi-tenant support

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Run the dashboard:
```bash
streamlit run app.py
```

## Usage

1. **Dashboard Overview**
   - View key metrics and recent anomalies
   - Monitor transaction trends
   - Track risk levels

2. **Anomaly Management**
   - Review detected anomalies
   - Provide feedback
   - Resolve issues

3. **Settings**
   - Configure risk thresholds
   - Set up notifications
   - Manage user preferences

## Development

### Project Structure

```
dashboard/
├── app.py              # Main application
├── components.py       # UI components
├── requirements.txt    # Dependencies
└── README.md          # Documentation
```

### Testing

Run tests with:
```bash
pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details 