#!/bin/bash

# Comprehensive Performance Test Script
# Tests caching, load balancing, and performance optimizations
# Validates production-ready performance characteristics

set -e

echo "ChipFabAI Performance Features Test Suite"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
API_GATEWAY_URL=${API_GATEWAY_URL:-"http://localhost:8081"}
GPU_SERVICE_URL=${GPU_SERVICE_URL:-"http://localhost:8080"}

echo "Configuration:"
echo "  API Gateway: $API_GATEWAY_URL"
echo "  GPU Service: $GPU_SERVICE_URL"
echo ""

# Check if services are running
echo "Checking if services are running..."
if ! curl -s "${API_GATEWAY_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}ERROR: API Gateway is not running at ${API_GATEWAY_URL}${NC}"
    echo "Please start the services first:"
    echo "  ./start_services_demo.sh"
    exit 1
fi

if ! curl -s "${GPU_SERVICE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}ERROR: GPU Service is not running at ${GPU_SERVICE_URL}${NC}"
    echo "Please start the services first:"
    echo "  ./start_services_demo.sh"
    exit 1
fi

echo -e "${GREEN}Services are running${NC}"
echo ""

# Run performance tests
echo "Running performance tests..."
echo ""

python3 test_performance_features.py "${API_GATEWAY_URL}"

TEST_EXIT_CODE=$?

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}All performance tests passed!${NC}"
else
    echo -e "${RED}Some performance tests failed${NC}"
fi

echo ""
echo "Performance Test Summary:"
echo "  - Cache functionality: Tested"
echo "  - Cache statistics: Tested"
echo "  - Metrics endpoint: Tested"
echo "  - Concurrent requests: Tested"
echo "  - Load balancer health: Tested"
echo ""

exit $TEST_EXIT_CODE

