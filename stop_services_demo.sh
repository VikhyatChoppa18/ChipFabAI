!/bin/bash
# Stop Demo Services

cd /home/v/PycharmProjects/YadVansh/Competitions/Mgp_Sys

echo " Stopping ChipFabAI Demo Services..."

# Read PIDs from files
GPU_PID=$(cat /tmp/gpu_demo.pid 2>/dev/null || echo "")
API_PID=$(cat /tmp/api_demo.pid 2>/dev/null || echo "")

# Kill by PID
if [ -n "$GPU_PID" ]; then
    kill $GPU_PID 2>/dev/null || true
    echo " Stopped GPU Service (PID: $GPU_PID)"
fi

if [ -n "$API_PID" ]; then
    kill $API_PID 2>/dev/null || true
    echo " Stopped API Gateway (PID: $API_PID)"
fi

# Kill by port (fallback)
lsof -ti:8080 | xargs kill -9 2>/dev/null || true
lsof -ti:8081 | xargs kill -9 2>/dev/null || true

# Clean up PID files
rm -f /tmp/gpu_demo.pid /tmp/api_demo.pid

echo " All services stopped"

