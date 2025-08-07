# FastAPI ML Service Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables to avoid NumPy issues
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1
ENV NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=0
ENV OMP_NUM_THREADS=1

# Copy and install requirements
COPY requirements.txt .

# Install packages in optimized order
RUN pip install --upgrade pip && \
    pip install --no-cache-dir numpy==1.24.4 && \
    pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

# Test critical imports
RUN python -c "import numpy; print(f'NumPy: {numpy.__version__}'); import torch; print(f'PyTorch: {torch.__version__}'); import pandas; print('All imports successful')"

# Copy application code
COPY . .

# Make startup script executable
RUN chmod +x /app/startup.sh

# Create non-root user
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["/app/startup.sh"]
