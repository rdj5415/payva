# AuditPulse MVP

AuditPulse is an AI-powered financial transaction monitoring platform that helps businesses detect anomalies, prevent fraud, and gain valuable insights in real-time.

## Features

- 🔍 Real-time transaction monitoring
- 🤖 AI-powered anomaly detection
- 📊 Comprehensive analytics dashboard
- 🔔 Customizable alerts and notifications
- 🔒 Enterprise-grade security
- 📱 Responsive web interface

## Tech Stack

### Backend
- FastAPI (Python)
- PostgreSQL
- SQLAlchemy
- Alembic
- Pydantic
- Celery
- Redis

### Frontend
- React
- TypeScript
- Material-UI
- React Router
- Auth0
- Formik
- Yup

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 18+
- PostgreSQL
- Redis

### Backend Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Initialize the database:
```bash
alembic upgrade head
```

5. Run the development server:
```bash
uvicorn auditpulse_mvp.main:app --reload
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the development server:
```bash
npm start
```

## Deployment

### Backend Deployment
The backend can be deployed on any cloud platform that supports Python applications (AWS, Google Cloud, Azure, etc.).

### Frontend Deployment
The frontend is configured for deployment on Netlify. See `netlify.toml` for configuration details.

## API Documentation

Once the backend server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, email support@auditpulse.com or join our Slack channel. 