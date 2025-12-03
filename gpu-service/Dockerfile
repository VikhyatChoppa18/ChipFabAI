# Multi-stage build for cost optimization - smaller image size
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

# Set working directory
WORKDIR /app

# Install Python 3.11 and minimal dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-venv \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create virtual environment for isolation
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8080
ENV MODEL_CACHE_DIR=/tmp/models
ENV CUDA_VISIBLE_DEVICES=0

# Expose port
EXPOSE 8080

# Health check for Cloud Run
# Cloud Run has its own health checks, but this helps during container startup
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=5)"

# Run the application
CMD ["python3", "main.py"]
