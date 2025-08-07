#!/bin/bash

# Ethereum Fraud Detection App - Docker Deployment Script

echo "🚀 Starting Ethereum Fraud Detection Application..."
echo "======================================================"

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists in backendApi
if [ ! -f ./backendApi/.env ]; then
    echo "❌ ./backendApi/.env file not found!"
    echo "Please ensure the .env file exists in the backendApi directory."
    echo "It should contain at least: ETHERSCAN_API_KEY=your_api_key_here"
    exit 1
fi

# Validate required environment variables
echo "🔍 Validating environment configuration..."
if [ -f ./backendApi/.env ]; then
    # Check if ETHERSCAN_API_KEY exists and is not empty
    if grep -q "ETHERSCAN_API_KEY=" ./backendApi/.env && ! grep -q "ETHERSCAN_API_KEY=$" ./backendApi/.env; then
        echo "✅ ETHERSCAN_API_KEY found in ./backendApi/.env"
    else
        echo "❌ ETHERSCAN_API_KEY is not configured in ./backendApi/.env file"
        echo "Please add: ETHERSCAN_API_KEY=your_actual_api_key"
        exit 1
    fi
else
    echo "❌ ./backendApi/.env file not found"
    exit 1
fi

echo "✅ Environment configuration looks good!"

# Function to handle cleanup
cleanup() {
    echo "🧹 Cleaning up..."
    docker-compose down
    exit 0
}

# Set up signal handling
trap cleanup SIGINT SIGTERM

# Build and start services
echo "🔨 Building and starting services..."
docker-compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🔍 Checking service health..."

# Check FastAPI
if curl -f http://localhost:8000/ > /dev/null 2>&1; then
    echo "✅ FastAPI service is running (http://localhost:8000)"
else
    echo "❌ FastAPI service is not responding"
fi

# Check Spring Boot Backend
if curl -f http://localhost:8080/actuator/health > /dev/null 2>&1; then
    echo "✅ Spring Boot backend is running (http://localhost:8080)"
else
    echo "⚠️  Spring Boot backend is not responding (might still be starting)"
fi

# Check Frontend
if curl -f http://localhost:3000/ > /dev/null 2>&1; then
    echo "✅ React frontend is running (http://localhost:3000)"
else
    echo "❌ React frontend is not responding"
fi

echo ""
echo "🎉 Application deployment complete!"
echo "======================================"
echo "📱 Frontend:  http://localhost:3000"
echo "🔧 Backend:   http://localhost:8080"
echo "🤖 FastAPI:   http://localhost:8000"
echo ""
echo "📊 View logs: docker-compose logs -f"
echo "🛑 Stop app:  docker-compose down"
echo ""
echo "Press Ctrl+C to stop the application..."

# Follow logs
docker-compose logs -f
