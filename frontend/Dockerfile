# Use Python 3.10 as the base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional production dependencies
RUN pip install --no-cache-dir gunicorn uvloop httptools

# Copy the project files
COPY . .

# Initialize the database on container startup
RUN mkdir -p /app/data

# Create a non-root user to run the application
RUN adduser --disabled-password --gecos "" appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose the port the app runs on
EXPOSE $PORT

# Command to run the application with Gunicorn
CMD ["sh", "-c", "python -m auditpulse_mvp.scripts.initialize_db && gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT auditpulse_mvp.main:app"] 