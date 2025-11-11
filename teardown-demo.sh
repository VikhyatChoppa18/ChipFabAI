#!/bin/bash

# ChipFabAI Demo Teardown Script
# Removes demo services to stop costs

set -e

# Configuration
REGION=${REGION:-"europe-west4"}
GPU_SERVICE_NAME="chipfabai-gpu-demo"
API_SERVICE_NAME="chipfabai-api-demo"
FRONTEND_SERVICE_NAME="chipfabai-frontend-demo"

echo "Tearing down ChipFabAI Demo Services..."
echo "Region: $REGION"
echo ""

# Delete services
echo "Deleting GPU Service..."
gcloud run services delete $GPU_SERVICE_NAME --region=$REGION --quiet || true

echo "Deleting API Gateway..."
gcloud run services delete $API_SERVICE_NAME --region=$REGION --quiet || true

echo "Deleting Frontend..."
gcloud run services delete $FRONTEND_SERVICE_NAME --region=$REGION --quiet || true

echo ""
echo "Demo services deleted successfully!"
echo "No more charges for these services"
echo ""
echo "To redeploy, run: ./deploy-demo.sh"

