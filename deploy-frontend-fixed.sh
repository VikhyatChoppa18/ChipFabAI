#!/bin/bash

# Deploy Frontend with Fixed API URL
# This script builds and deploys the frontend with the correct API URL

set -e

PROJECT_ID=${PROJECT_ID:-"mgpsys"}
REGION=${REGION:-"europe-west4"}
FRONTEND_SERVICE_NAME="chipfabai-frontend-demo"

echo "Deploying Frontend with Fixed API URL..."
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# Get API Gateway URL
echo "Getting API Gateway URL..."
API_SERVICE_URL=$(gcloud run services describe chipfabai-api-demo --project=$PROJECT_ID --region=$REGION --format='value(status.url)')
echo "API URL: $API_SERVICE_URL"
echo ""

# Build and deploy frontend
cd frontend

IMAGE_NAME="gcr.io/$PROJECT_ID/$FRONTEND_SERVICE_NAME:latest"

echo "Building frontend with API URL: $API_SERVICE_URL"
echo "This will take 3-5 minutes..."
echo ""

# Build the image
gcloud builds submit \
  --config=cloudbuild.yaml \
  --substitutions=_REACT_APP_API_URL="$API_SERVICE_URL",_IMAGE_URL="$IMAGE_NAME" \
  --region=$REGION \
  --project=$PROJECT_ID

echo ""
echo "Deploying to Cloud Run..."

# Deploy to Cloud Run
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
  --cpu-throttling \
  --project=$PROJECT_ID

FRONTEND_SERVICE_URL=$(gcloud run services describe $FRONTEND_SERVICE_NAME --region=$REGION --format='value(status.url)')

echo ""
echo "Frontend deployed successfully!"
echo ""
echo "Frontend URL: $FRONTEND_SERVICE_URL"
echo "API URL configured: $API_SERVICE_URL"
echo ""
echo "The frontend should now work correctly!"
echo "Test it at: $FRONTEND_SERVICE_URL"

cd ..

