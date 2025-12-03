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
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable pubsub.googleapis.com
gcloud services enable notebooks.googleapis.com

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

# Create Pub/Sub topics for event streaming
echo ""
echo "Setting up Pub/Sub topics..."
gcloud pubsub topics create chipfabai-predictions --project=$PROJECT_ID 2>/dev/null || echo "Topic chipfabai-predictions already exists"
gcloud pubsub topics create chipfabai-alerts --project=$PROJECT_ID 2>/dev/null || echo "Topic chipfabai-alerts already exists"

# Deploy Cloud Functions
echo ""
echo "Deploying Cloud Functions..."

# Anomaly Detector Function
echo "Deploying Anomaly Detector Function..."
cd cloud-functions/anomaly-detector
gcloud functions deploy anomaly-detector \
  --gen2 \
  --runtime=python311 \
  --region=$REGION \
  --source=. \
  --entry-point=anomaly_detector \
  --trigger-topic=chipfabai-predictions \
  --set-env-vars="GCP_PROJECT=$PROJECT_ID,ALERTS_TOPIC=chipfabai-alerts,ENABLE_ALERTS=true" \
  --timeout=60s \
  --memory=256MB \
  --max-instances=10 \
  --allow-unauthenticated 2>/dev/null || echo "Anomaly detector function deployment skipped (may need manual setup)"
cd ../..

# Create Cloud Storage bucket for training data if it doesn't exist
BUCKET_NAME="${PROJECT_ID}-training-data"
gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME 2>/dev/null || echo "Bucket $BUCKET_NAME already exists or creation skipped"

# Model Retrainer Function
echo "Deploying Model Retrainer Function..."
cd cloud-functions/model-retrainer
gcloud functions deploy model-retrainer \
  --gen2 \
  --runtime=python311 \
  --region=$REGION \
  --source=. \
  --entry-point=model_retrainer \
  --trigger-bucket=$BUCKET_NAME \
  --set-env-vars="GCP_PROJECT=$PROJECT_ID,REGION=$REGION,TRAINING_DATA_BUCKET=$BUCKET_NAME,MODEL_NAME=chipfabai-yield-predictor,MIN_SAMPLES_FOR_RETRAIN=1000" \
  --timeout=540s \
  --memory=512MB \
  --max-instances=5 \
  --allow-unauthenticated 2>/dev/null || echo "Model retrainer function deployment skipped (may need manual setup)"
cd ../..

echo ""
echo "Demo Deployment Complete!"
echo ""
echo "Services:"
echo "  GPU Service:    $GPU_SERVICE_URL"
echo "  API Gateway:    $API_SERVICE_URL"
echo "  Frontend:       $FRONTEND_SERVICE_URL"
echo ""
echo "Event-Driven Services:"
echo "  Pub/Sub Topics: chipfabai-predictions, chipfabai-alerts"
echo "  Cloud Functions: anomaly-detector, model-retrainer"
echo "  Vertex AI Workbench: notebooks/model_experimentation.py"
echo ""
echo "Cost Optimization Features:"
echo "  min-instances=0 (scales to zero when idle)"
echo "  Reduced memory/CPU for non-GPU services"
echo "  CPU throttling enabled (API Gateway & Frontend)"
echo "  Limited max-instances (2 instead of 10)"
echo ""
echo "Advanced Features:"
echo "  ✓ Pub/Sub event streaming for real-time processing"
echo "  ✓ Cloud Functions for anomaly detection"
echo "  ✓ Automated model retraining triggers"
echo "  ✓ Vertex AI Workbench for ML experimentation"
echo ""
echo "Estimated Cost:"
echo "  - Idle (no requests): $0.00/hour"
echo "  - Active (handling requests): ~$0.50-1.00/hour"
echo "  - 3-hour demo session: ~$2-4"
echo ""
echo "Remember to tear down after demo: ./teardown-demo.sh"
echo ""
echo "Access the application at: $FRONTEND_SERVICE_URL"

