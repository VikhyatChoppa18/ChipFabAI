#!/bin/bash
# Start Services for Demo
# Starts GPU service and API Gateway locally for testing
# Automatically handles port conflicts by killing existing processes

set -e

cd /home/v/PycharmProjects/YadVansh/Competitions/Mgp_Sys

echo "Starting ChipFabAI Services for Demo Video"
echo "Using optimized configuration for local testing"
echo ""

# Checking for and terminating any processes using required ports
echo "Clearing ports..."
sudo killall nginx 2>/dev/null || true
sudo pkill -9 nginx 2>/dev/null || true
lsof -ti:8080,8081 | xargs sudo kill -9 2>/dev/null || true
sudo fuser -k 8080/tcp 2>/dev/null || true
sudo fuser -k 8081/tcp 2>/dev/null || true
sleep 5

# Verify ports are free
if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo " Port 8080 still in use. Trying harder..."
    sudo fuser -k 8080/tcp 2>/dev/null || true
    sleep 2
fi

if lsof -Pi :8081 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo " Port 8081 still in use. Trying harder..."
    sudo fuser -k 8081/tcp 2>/dev/null || true
    sleep 2
fi

# Using port 8082 for GPU service to avoid conflicts with nginx or system services
# Port 8080 is commonly used by nginx and other system services
GPU_PORT=8082
echo "Using port $GPU_PORT for GPU Service (avoids port conflicts)"

# Starting GPU Service in the background
echo "Starting GPU Service on port $GPU_PORT..."
cd gpu-service
source ../venv/bin/activate
# Using smaller model for faster startup during local testing
export MODEL_NAME=microsoft/DialoGPT-small
export PORT=$GPU_PORT
export MODEL_CACHE_DIR=/tmp/models

# Starting service in background and capturing process ID
python3 main.py > /tmp/gpu_demo.log 2>&1 &
GPU_PID=$!
cd ..
echo "GPU Service PID: $GPU_PID"

# Wait for GPU service with timeout
echo "Waiting for GPU Service to start..."
MAX_WAIT=30
for i in $(seq 1 $MAX_WAIT); do
    if curl -s http://localhost:$GPU_PORT/health > /dev/null 2>&1; then
        echo "GPU Service is ready!"
        curl -s http://localhost:$GPU_PORT/health | python3 -m json.tool 2>/dev/null | head -5 || echo "Service responding"
        break
    fi
    if [ $i -eq $MAX_WAIT ]; then
        echo "GPU Service failed to start after ${MAX_WAIT}s"
        echo "Last 20 lines of log:"
        tail -20 /tmp/gpu_demo.log
        exit 1
    fi
    sleep 1
done

# Start API Gateway
echo "Starting API Gateway..."
cd api-gateway
source ../venv/bin/activate
export GPU_SERVICE_URL=http://localhost:$GPU_PORT
export PORT=8081

# Start in background
python3 main.py > /tmp/api_demo.log 2>&1 &
API_PID=$!
cd ..
echo "API Gateway PID: $API_PID"

# Wait for API Gateway with timeout
echo "Waiting for API Gateway to start..."
for i in $(seq 1 $MAX_WAIT); do
    if curl -s http://localhost:8081/health > /dev/null 2>&1; then
        echo "API Gateway is ready!"
        curl -s http://localhost:8081/health | python3 -m json.tool 2>/dev/null | head -5 || echo "Service responding"
        break
    fi
    if [ $i -eq $MAX_WAIT ]; then
        echo "API Gateway failed to start after ${MAX_WAIT}s"
        echo "Last 20 lines of log:"
        tail -20 /tmp/api_demo.log
        kill $GPU_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

sleep 2

echo ""
echo "All services started successfully!"
echo ""
echo "Service URLs:"
echo "  GPU Service:    http://localhost:$GPU_PORT"
echo "  API Gateway:     http://localhost:8081"
echo ""
echo "PIDs (to stop services):"
echo "  GPU Service:    $GPU_PID"
echo "  API Gateway:     $API_PID"
echo ""
echo "To stop services: kill $GPU_PID $API_PID"
echo ""

# Save PIDs for cleanup
echo "$GPU_PID" > /tmp/gpu_demo.pid
echo "$API_PID" > /tmp/api_demo.pid

echo "Ready for testing! Run: python3 test_running_services.py"

