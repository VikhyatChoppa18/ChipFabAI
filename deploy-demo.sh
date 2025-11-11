#!/bin/bash

# ChipFabAI Demo Deployment Script
# Deploys all services to Google Cloud Run with configuration optimized for demo usage
# Services are configured to scale to zero when idle to minimize costs

set -e

# Configuration variables with default values
PROJECT_ID=${PROJECT_ID:-"your-project-id"}
REGION=${REGION:-"europe-west4"}
GPU_SERVICE_NAME="chipfabai-gpu-demo"
API_SERVICE_NAME="chipfabai-api-demo"
FRONTEND_SERVICE_NAME="chipfabai-frontend-demo"

echo "ChipFabAI Demo Deployment"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo ""
echo "Services are configured to scale to zero when idle"
echo "Estimated cost for 3-hour demo session: ~$2-4"
echo ""

# Setting the active GCP project for deployment
gcloud config set project $PROJECT_ID

# Enabling Google Cloud APIs needed for Cloud Run deployment
echo "Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable aiplatform.googleapis.com

# Deploying GPU Service with NVIDIA L4 GPU
# Configuration includes GPU allocation, memory settings, and auto-scaling limits
echo "Deploying GPU Service..."
cd gpu-service
gcloud run deploy $GPU_SERVICE_NAME \
  --source . \
  --region=$REGION \
  --platform=managed \
  --allow-unauthenticated \
  --set-env-vars="MODEL_CACHE_DIR=/tmp/models,MODEL_NAME=microsoft/DialoGPT-small" \
  --memory=16Gi \
  --cpu=4 \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --no-gpu-zonal-redundancy \
  --timeout=300 \
  --max-instances=2 \
  --min-instances=0 \
  --concurrency=10

GPU_SERVICE_URL=$(gcloud run services describe $GPU_SERVICE_NAME --region=$REGION --format='value(status.url)')
echo "GPU Service deployed: $GPU_SERVICE_URL"
cd ..

# Deploying API Gateway service
# API Gateway acts as an intermediary between frontend and GPU service
# Configured with reduced resource allocation and auto-scaling
echo "Deploying API Gateway..."
cd api-gateway
gcloud run deploy $API_SERVICE_NAME \
  --source . \
  --region=$REGION \
  --platform=managed \
  --allow-unauthenticated \
  --set-env-vars="GPU_SERVICE_URL=$GPU_SERVICE_URL" \
  --memory=1Gi \
  --cpu=1 \
  --timeout=300 \
  --max-instances=2 \
  --min-instances=0 \
  --concurrency=80 \
  --cpu-throttling

API_SERVICE_URL=$(gcloud run services describe $API_SERVICE_NAME --region=$REGION --format='value(status.url)')
echo "API Gateway deployed: $API_SERVICE_URL"
cd ..

# Deploying Frontend service
# Frontend is built using Cloud Build and then deployed to Cloud Run
echo "Deploying Frontend..."
cd frontend

# Building container image using Cloud Build with environment-specific substitutions
IMAGE_NAME="gcr.io/$PROJECT_ID/$FRONTEND_SERVICE_NAME:latest"
gcloud builds submit \
  --config=cloudbuild.yaml \
  --substitutions=_REACT_APP_API_URL="$API_SERVICE_URL",_IMAGE_URL="$IMAGE_NAME" \
  --region=$REGION

# Deploying built image to Cloud Run
gcloud run deploy $FRONTEND_SERVICE_NAME \
  --image $IMAGE_NAME \
  --region=$REGION \
  --platform=managed \
  --allow-unauthenticated \
  --memory=256Mi \
  --cpu=1 \
  --timeout=300 \
  --max-instances=2 \
  --min-instances=0 \
  --concurrency=80 \
  --cpu-throttling

FRONTEND_SERVICE_URL=$(gcloud run services describe $FRONTEND_SERVICE_NAME --region=$REGION --format='value(status.url)')
echo "Frontend deployed: $FRONTEND_SERVICE_URL"
cd ..

echo ""
echo "Demo Deployment Complete!"
echo ""
echo "Services:"
echo "  GPU Service:    $GPU_SERVICE_URL"
echo "  API Gateway:    $API_SERVICE_URL"
echo "  Frontend:       $FRONTEND_SERVICE_URL"
echo ""
echo "Cost Optimization Features:"
echo "  min-instances=0 (scales to zero when idle)"
echo "  Reduced memory/CPU for non-GPU services"
echo "  CPU throttling enabled (API Gateway & Frontend)"
echo "  Limited max-instances (2 instead of 10)"
echo ""
echo "Estimated Cost:"
echo "  - Idle (no requests): $0.00/hour"
echo "  - Active (handling requests): ~$0.50-1.00/hour"
echo "  - 3-hour demo session: ~$2-4"
echo ""
echo "Remember to tear down after demo: ./teardown-demo.sh"
echo ""
echo "Access the application at: $FRONTEND_SERVICE_URL"

