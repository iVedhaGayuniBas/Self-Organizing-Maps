# Multi-stage Dockerfile for SOM Context Retrieval Evaluation
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 as base

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
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt


# Copy all project files into /app
COPY . /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port for Ray dashboard or web services
EXPOSE 8000

# Entrypoint for running both scripts in parallel and saving results in the results directory
ENTRYPOINT ["bash", "-c"]
CMD ["python3 generate_contexts.py & python3 scale_up_evaluation.py & wait"]
