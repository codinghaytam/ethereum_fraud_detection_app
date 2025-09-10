# FastAPI ML Service Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables to avoid NumPy issues
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1
ENV NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=0
ENV OMP_NUM_THREADS=1
ENV PYTHONPATH=/app:/app/src

# Declare expected runtime env vars (values provided via docker-compose env_file)
ENV ETHERSCAN_API_KEY=""
ENV PG_PASSWORD=""

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and startup script
COPY . .

# Create non-root user
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Use startup script instead of direct uvicorn
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]