#!/bin/bash
# ChipFabAI Local Development Startup Script
# This script helps start all services for local testing

set -e

# Configuration
FRONTEND_PORT=${FRONTEND_PORT:-3333}

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ChipFabAI Local Development Setup${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip, setuptools, and wheel first (fixes Python 3.12 compatibility)
echo -e "${YELLOW}Upgrading pip, setuptools, and wheel...${NC}"
pip install --upgrade pip setuptools wheel

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -q -r api-gateway/requirements.txt
pip install -q -r gpu-service/requirements.txt
pip install -q -r data-processor/requirements.txt

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js is not installed. Please install Node.js 18+${NC}"
    exit 1
fi

# Install frontend dependencies
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    cd frontend
    npm install
    cd ..
fi

echo -e "\n${GREEN}Dependencies installed successfully!${NC}\n"

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Check ports
echo -e "${YELLOW}Checking if ports are available...${NC}"
if check_port 8080; then
    echo -e "${RED}Port 8080 is already in use. Please stop the service using that port.${NC}"
    exit 1
fi

if check_port 8081; then
    echo -e "${RED}Port 8081 is already in use. Please stop the service using that port.${NC}"
    exit 1
fi

if check_port $FRONTEND_PORT; then
    echo -e "${RED}Port ${FRONTEND_PORT} is already in use. Please stop the service using that port.${NC}"
    exit 1
fi

echo -e "${GREEN}All ports are available!${NC}\n"

# Create log directory
mkdir -p logs

echo -e "${BLUE}Starting services...${NC}\n"
echo -e "${YELLOW}Note: Services will run in the background.${NC}"
echo -e "${YELLOW}Logs will be written to the 'logs' directory.${NC}\n"

# Start GPU Service
echo -e "${BLUE}[1/3] Starting GPU Service on port 8080...${NC}"
cd gpu-service
python main.py > ../logs/gpu-service.log 2>&1 &
GPU_PID=$!
cd ..
echo -e "${GREEN}GPU Service started (PID: $GPU_PID)${NC}"

# Wait a bit for GPU service to start
sleep 3

# Start API Gateway
echo -e "${BLUE}[2/3] Starting API Gateway on port 8081...${NC}"
cd api-gateway
export GPU_SERVICE_URL=http://localhost:8080
python main.py > ../logs/api-gateway.log 2>&1 &
API_PID=$!
cd ..
echo -e "${GREEN}API Gateway started (PID: $API_PID)${NC}"

# Wait a bit for API Gateway to start
sleep 2

# Start Frontend
echo -e "${BLUE}[3/3] Starting Frontend on port ${FRONTEND_PORT}...${NC}"
cd frontend
export REACT_APP_API_URL=http://localhost:8081
npm start -- --port ${FRONTEND_PORT} > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo -e "${GREEN}Frontend started (PID: $FRONTEND_PID)${NC}"

# Save PIDs to file for easy cleanup
echo "$GPU_PID" > logs/gpu-service.pid
echo "$API_PID" > logs/api-gateway.pid
echo "$FRONTEND_PID" > logs/frontend.pid

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}All services started successfully!${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo -e "${BLUE}Service URLs:${NC}"
echo -e "  GPU Service:    http://localhost:8080"
echo -e "  API Gateway:     http://localhost:8081"
echo -e "  Frontend:        http://localhost:${FRONTEND_PORT}"
echo -e "\n${BLUE}Logs:${NC}"
echo -e "  GPU Service:    logs/gpu-service.log"
echo -e "  API Gateway:     logs/api-gateway.log"
echo -e "  Frontend:        logs/frontend.log"
echo -e "\n${YELLOW}To stop all services, run:${NC}"
echo -e "  ./stop_local.sh"
echo -e "\n${YELLOW}To view logs:${NC}"
echo -e "  tail -f logs/gpu-service.log"
echo -e "  tail -f logs/api-gateway.log"
echo -e "  tail -f logs/frontend.log"
echo -e "\n${GREEN}Waiting for services to be ready...${NC}\n"

# Wait for services to be ready
sleep 5

# Check if services are responding
echo -e "${YELLOW}Checking service health...${NC}"

# Check GPU Service
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ GPU Service is healthy${NC}"
else
    echo -e "${RED}✗ GPU Service is not responding. Check logs/gpu-service.log${NC}"
fi

# Check API Gateway
if curl -s http://localhost:8081/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ API Gateway is healthy${NC}"
else
    echo -e "${RED}✗ API Gateway is not responding. Check logs/api-gateway.log${NC}"
fi

echo -e "\n${GREEN}Setup complete! Open http://localhost:${FRONTEND_PORT} in your browser.${NC}\n"

