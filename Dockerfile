# Multi-stage Dockerfile for SOM Context Retrieval Evaluation
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

# Development stage
FROM base as development
RUN pip install --no-cache-dir jupyter ipykernel

# Production stage
FROM base as production

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port for any web services
EXPOSE 8000

# Default command
CMD ["python", "-m", "pytest", "tests/", "-v"]

# Evaluation stage - optimized for running evaluations
FROM base as evaluation

# Install additional evaluation dependencies
RUN pip install --no-cache-dir \
    ray[default]==2.8.0 \
    torch==2.1.0 \
    transformers==4.35.0

# Copy application code
COPY . .

# Create directories for outputs
RUN mkdir -p /app/outputs /app/contexts /app/logs

# Set entrypoint for evaluation scripts
ENTRYPOINT ["python"]
CMD ["scale_up_evaluation.py"]
