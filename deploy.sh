#!/bin/bash

# ChipFabAI Deployment Script
# Deploys all services to Google Cloud Run

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"your-project-id"}
REGION=${REGION:-"europe-west4"}
GPU_SERVICE_NAME="chipfabai-gpu"
API_SERVICE_NAME="chipfabai-api"
FRONTEND_SERVICE_NAME="chipfabai-frontend"

echo " Starting ChipFabAI Deployment..."
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo " Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable aiplatform.googleapis.com

# Deploy GPU Service
echo " Deploying GPU Service..."
cd gpu-service
gcloud run deploy $GPU_SERVICE_NAME \
  --source . \
  --region=$REGION \
  --platform=managed \
  --allow-unauthenticated \
  --set-env-vars="MODEL_CACHE_DIR=/tmp/models" \
  --memory=16Gi \
  --cpu=4 \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --timeout=300 \
  --max-instances=10 \
  --min-instances=0

GPU_SERVICE_URL=$(gcloud run services describe $GPU_SERVICE_NAME --region=$REGION --format='value(status.url)')
echo " GPU Service deployed: $GPU_SERVICE_URL"
cd ..

# Deploy API Gateway
echo " Deploying API Gateway..."
cd api-gateway
gcloud run deploy $API_SERVICE_NAME \
  --source . \
  --region=$REGION \
  --platform=managed \
  --allow-unauthenticated \
  --set-env-vars="GPU_SERVICE_URL=$GPU_SERVICE_URL" \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300 \
  --max-instances=10 \
  --min-instances=0

API_SERVICE_URL=$(gcloud run services describe $API_SERVICE_NAME --region=$REGION --format='value(status.url)')
echo " API Gateway deployed: $API_SERVICE_URL"
cd ..

# Deploy Frontend
echo " Deploying Frontend..."
cd frontend
gcloud run deploy $FRONTEND_SERVICE_NAME \
  --source . \
  --region=$REGION \
  --platform=managed \
  --allow-unauthenticated \
  --set-env-vars="REACT_APP_API_URL=$API_SERVICE_URL" \
  --memory=512Mi \
  --cpu=1 \
  --timeout=300 \
  --max-instances=10 \
  --min-instances=0

FRONTEND_SERVICE_URL=$(gcloud run services describe $FRONTEND_SERVICE_NAME --region=$REGION --format='value(status.url)')
echo " Frontend deployed: $FRONTEND_SERVICE_URL"
cd ..

echo ""
echo " Deployment Complete!"
echo ""
echo "Services:"
echo "  GPU Service:    $GPU_SERVICE_URL"
echo "  API Gateway:    $API_SERVICE_URL"
echo "  Frontend:       $FRONTEND_SERVICE_URL"
echo ""
echo "Access the application at: $FRONTEND_SERVICE_URL"

