#!/bin/bash
# ChipFabAI Local Development Stop Script
# This script stops all running services

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
echo -e "${BLUE}Stopping ChipFabAI Services${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Function to stop a service by PID file
stop_service() {
    local service_name=$1
    local pid_file="logs/${service_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo -e "${YELLOW}Stopping ${service_name} (PID: $pid)...${NC}"
            kill $pid 2>/dev/null || true
            sleep 1
            # Force kill if still running
            if ps -p $pid > /dev/null 2>&1; then
                kill -9 $pid 2>/dev/null || true
            fi
            echo -e "${GREEN}✓ ${service_name} stopped${NC}"
        else
            echo -e "${YELLOW}${service_name} was not running${NC}"
        fi
        rm -f "$pid_file"
    else
        echo -e "${YELLOW}No PID file found for ${service_name}${NC}"
    fi
}

# Stop services
stop_service "gpu-service"
stop_service "api-gateway"
stop_service "frontend"

# Also kill any processes on the ports (fallback)
echo -e "\n${YELLOW}Checking for processes on service ports...${NC}"

# Kill processes on port 8080 (GPU Service)
if lsof -ti:8080 > /dev/null 2>&1; then
    echo -e "${YELLOW}Killing process on port 8080...${NC}"
    lsof -ti:8080 | xargs kill -9 2>/dev/null || true
fi

# Kill processes on port 8081 (API Gateway)
if lsof -ti:8081 > /dev/null 2>&1; then
    echo -e "${YELLOW}Killing process on port 8081...${NC}"
    lsof -ti:8081 | xargs kill -9 2>/dev/null || true
fi

# Kill processes on frontend port
if lsof -ti:${FRONTEND_PORT} > /dev/null 2>&1; then
    echo -e "${YELLOW}Killing process on port ${FRONTEND_PORT}...${NC}"
    lsof -ti:${FRONTEND_PORT} | xargs kill -9 2>/dev/null || true
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}All services stopped!${NC}"
echo -e "${GREEN}========================================${NC}\n"

