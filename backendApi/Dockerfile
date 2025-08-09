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

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Test critical imports
RUN python -c "import numpy; print(f'NumPy: {numpy.__version__}'); import torch; print(f'PyTorch: {torch.__version__}'); import pandas; print('All imports successful')"

# Copy source code and startup script
COPY . .
COPY startup.sh /app/startup.sh

COPY model/*.pth /app/model/
# Make startup script executable
RUN chmod +x /app/startup.sh

# Create non-root user
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000


# Use startup script instead of direct uvicorn
CMD ["/app/startup.sh"]
