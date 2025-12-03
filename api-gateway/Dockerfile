# Multi-stage build for cost optimization - smaller image size
FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code - need all Python files for imports
# Using explicit /app/ destination since COPY with multiple files requires directory to end with /
COPY main.py cache.py load_balancer.py /app/

# Expose port
EXPOSE 8080

# Set environment variables
ENV PORT=8080

# Health check for Cloud Run
# Cloud Run has its own health checks, but this helps during container startup
# Increased start-period to 60s to allow time for FastAPI to fully start
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Run the application
CMD ["python3", "main.py"]
