# Production-Minded Lightweight Dockerfile for Facet Evaluator FastAPI Backend
FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install CPU PyTorch first (180 MB instead of 2.8 GB CUDA wheel), then requirements
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and preprocessed data
COPY src/ ./src/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY .env.example .env

# Expose FastAPI application port
EXPOSE 8000

# Health check instruction
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI application with Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
