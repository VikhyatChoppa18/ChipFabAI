# ChipFabAI Deployment Guide


1. **Google Cloud Account** with billing enabled
2. **Google Cloud SDK** installed and configured
3. **Docker** installed (for local testing)
4. **Node.js 18+** (for frontend development)



```bash
# Set your project ID
export PROJECT_ID=your-project-id
export REGION=europe-west4

# Authenticate
gcloud auth login
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable aiplatform.googleapis.com
```


```bash
# Make script executable
chmod +x deploy.sh

# Run deployment
export PROJECT_ID=your-project-id
export REGION=europe-west4
./deploy.sh
```



```bash
cd gpu-service

gcloud run deploy chipfabai-gpu \
  --source . \
  --region=europe-west4 \
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

# Get service URL
GPU_SERVICE_URL=$(gcloud run services describe chipfabai-gpu --region=europe-west4 --format='value(status.url)')
echo "GPU Service URL: $GPU_SERVICE_URL"
```


```bash
cd ../api-gateway

gcloud run deploy chipfabai-api \
  --source . \
  --region=europe-west4 \
  --platform=managed \
  --allow-unauthenticated \
  --set-env-vars="GPU_SERVICE_URL=$GPU_SERVICE_URL" \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300 \
  --max-instances=10 \
  --min-instances=0

# Get service URL
API_SERVICE_URL=$(gcloud run services describe chipfabai-api --region=europe-west4 --format='value(status.url)')
echo "API Service URL: $API_SERVICE_URL"
```


```bash
cd ../frontend

gcloud run deploy chipfabai-frontend \
  --source . \
  --region=europe-west4 \
  --platform=managed \
  --allow-unauthenticated \
  --set-env-vars="REACT_APP_API_URL=$API_SERVICE_URL" \
  --memory=512Mi \
  --cpu=1 \
  --timeout=300 \
  --max-instances=10 \
  --min-instances=0

# Get service URL
FRONTEND_SERVICE_URL=$(gcloud run services describe chipfabai-frontend --region=europe-west4 --format='value(status.url)')
echo "Frontend URL: $FRONTEND_SERVICE_URL"
```



```bash
cd gpu-service

# Install dependencies
pip install -r requirements.txt

# Run locally (requires GPU)
python main.py
```


```bash
cd api-gateway

# Install dependencies
pip install -r requirements.txt

# Set environment variable
export GPU_SERVICE_URL=http://localhost:8080

# Run locally
python main.py
```


```bash
cd frontend

# Install dependencies
npm install

# Set environment variable
export REACT_APP_API_URL=http://localhost:8080

# Run development server
npm start
```



```bash
curl -X POST https://chipfabai-gpu-xxxxx.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 200.0,
    "pressure": 1.5,
    "etch_time": 60.0,
    "gas_flow": 100.0,
    "chamber_pressure": 5.0,
    "wafer_size": 300,
    "process_type": "etching"
  }'
```


```bash
curl -X POST https://chipfabai-api-xxxxx.run.app/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 200.0,
    "pressure": 1.5,
    "etch_time": 60.0,
    "gas_flow": 100.0,
    "chamber_pressure": 5.0,
    "wafer_size": 300,
    "process_type": "etching"
  }'
```



```bash
# GPU Service logs
gcloud run services logs read chipfabai-gpu --region=europe-west4

# API Gateway logs
gcloud run services logs read chipfabai-api --region=europe-west4

# Frontend logs
gcloud run services logs read chipfabai-frontend --region=europe-west4
```


```bash
# Open Cloud Console
gcloud run services describe chipfabai-gpu --region=europe-west4
```



- Ensure you're using `europe-west4` or `europe-west1` region
- Check GPU quota in Cloud Console
- Verify GPU type is `nvidia-l4`


- Check GPU service logs for errors
- Verify model cache directory permissions
- Ensure sufficient memory (16Gi)


- Verify GPU service URL is correct
- Check CORS configuration
- Verify service is running and healthy



- **GPU Service**: ~$500-1000/month (with auto-scaling)
- **API Gateway**: ~$50-100/month
- **Frontend**: ~$20-50/month
- **Storage**: ~$10-20/month
- **Total**: ~$600-1200/month


1. Use auto-scaling (min-instances=0)
2. Cache model responses
3. Use quantized models
4. Monitor usage and adjust limits


1. **Authentication**: Add IAM authentication for production
2. **HTTPS**: All services use HTTPS by default
3. **CORS**: Configure CORS for specific origins
4. **Rate Limiting**: Implement rate limiting in API Gateway
5. **Secrets**: Use Secret Manager for sensitive data



- Auto-scaling enabled (0-10 instances)
- Cloud Run handles load balancing
- Each instance can handle 80 concurrent requests


- GPU Service: Increase memory/CPU if needed
- API Gateway: Increase memory/CPU for higher load
- Frontend: Increase memory/CPU for larger deployments


1. **Code**: Store in version control (Git)
2. **Data**: Use Cloud Storage for data persistence
3. **Models**: Cache models in Cloud Storage
4. **Config**: Store configuration in environment variables


1. Set up monitoring and alerting
2. Implement authentication
3. Add rate limiting
4. Set up CI/CD pipeline
5. Configure custom domains
6. Add SSL certificates

