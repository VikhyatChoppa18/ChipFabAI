# ChipFabAI GCP Services Integration Guide

## New Services Added

### 1. Cloud Pub/Sub - Event Streaming
**Location**: Integrated in `api-gateway/main.py`

**What it does**:
- Publishes prediction events to `chipfabai-predictions` topic after each prediction
- Enables real-time event processing without blocking requests
- Events include parameters, predictions, and metadata

**Configuration**:
- Set `ENABLE_PUBSUB=true` (default: true)
- Set `PUBSUB_TOPIC_PREDICTIONS=chipfabai-predictions` (default)
- Requires `GCP_PROJECT` or `PROJECT_ID` environment variable

**How it works**:
When a prediction is made, the API Gateway publishes an event to Pub/Sub. This event is then consumed by Cloud Functions for anomaly detection and data collection.

---

### 2. Cloud Functions - Anomaly Detector
**Location**: `cloud-functions/anomaly-detector/`

**What it does**:
- Triggered by Pub/Sub messages from prediction events
- Detects anomalies in process parameters (temperature, pressure, yield)
- Sends alerts to `chipfabai-alerts` topic when anomalies are detected

**Deployment**:
```bash
cd cloud-functions/anomaly-detector
gcloud functions deploy anomaly-detector \
  --gen2 \
  --runtime=python311 \
  --region=europe-west4 \
  --source=. \
  --entry-point=anomaly_detector \
  --trigger-topic=chipfabai-predictions \
  --set-env-vars="GCP_PROJECT=chipfab-ai,ALERTS_TOPIC=chipfabai-alerts"
```

**Anomaly Detection Rules**:
- Temperature outside 150-250°C range
- Pressure outside 0.5-2.5 Torr range
- Yield prediction below 70%
- More than 2 risk factors detected

---

### 3. Cloud Functions - Model Retrainer
**Location**: `cloud-functions/model-retrainer/`

**What it does**:
- Triggered by Cloud Storage events (new training data uploaded)
- Can also be triggered by Pub/Sub (scheduled checks)
- Checks if enough new data is available for retraining
- Triggers Vertex AI training job when threshold is met

**Deployment**:
```bash
cd cloud-functions/model-retrainer
gcloud functions deploy model-retrainer \
  --gen2 \
  --runtime=python311 \
  --region=europe-west4 \
  --source=. \
  --entry-point=model_retrainer \
  --trigger-bucket=chipfab-ai-training-data \
  --set-env-vars="GCP_PROJECT=chipfab-ai,MIN_SAMPLES_FOR_RETRAIN=1000"
```

**Configuration**:
- `MIN_SAMPLES_FOR_RETRAIN`: Minimum samples needed before retraining (default: 1000)
- `TRAINING_DATA_BUCKET`: Cloud Storage bucket with training data
- `MODEL_NAME`: Name of the model to retrain

---

### 4. Vertex AI Workbench - ML Experimentation
**Location**: `notebooks/model_experimentation.py`

**What it does**:
- Python script for Vertex AI Workbench notebooks
- Model comparison (Random Forest, Gradient Boosting)
- Feature engineering experiments
- Hyperparameter tuning preparation
- Model evaluation and validation

**Usage**:
1. Open Vertex AI Workbench in GCP Console
2. Create a new notebook instance
3. Upload `notebooks/model_experimentation.py`
4. Run the script to experiment with different models

**Features**:
- Loads data from Cloud Storage
- Creates derived features
- Trains and compares multiple models
- Evaluates model performance
- Saves experiment results

---

## Architecture Flow

```
User Request → API Gateway → GPU Service
                    ↓
              Pub/Sub Topic (chipfabai-predictions)
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
Anomaly Detector        Data Collection
        ↓                       ↓
  Alerts Topic          Cloud Storage
  (chipfabai-alerts)    (training data)
                            ↓
                    Model Retrainer
                    (Cloud Function)
                            ↓
                    Vertex AI Training
```

---

## Deployment Steps

1. **Deploy Core Services** (as before):
   ```bash
   export PROJECT_ID=chipfab-ai
   export REGION=europe-west4
   ./deploy-demo.sh
   ```

2. **The deployment script now automatically**:
   - Enables required APIs (Pub/Sub, Cloud Functions, Notebooks)
   - Creates Pub/Sub topics
   - Deploys Cloud Functions
   - Creates training data bucket

3. **Manual Steps** (if automatic deployment fails):
   ```bash
   # Create Pub/Sub topics
   gcloud pubsub topics create chipfabai-predictions --project=chipfab-ai
   gcloud pubsub topics create chipfabai-alerts --project=chipfab-ai
   
   # Deploy Cloud Functions manually (see commands above)
   ```

---

## Testing the Integration

1. **Test Pub/Sub Event Publishing**:
   - Make a prediction through API Gateway
   - Check Pub/Sub topic for messages:
     ```bash
     gcloud pubsub subscriptions create test-sub --topic=chipfabai-predictions
     gcloud pubsub subscriptions pull test-sub --limit=1
     ```

2. **Test Anomaly Detection**:
   - Make a prediction with anomalous parameters (e.g., temperature=300°C)
   - Check Cloud Functions logs:
     ```bash
     gcloud functions logs read anomaly-detector --region=europe-west4
     ```

3. **Test Model Retrainer**:
   - Upload training data to Cloud Storage:
     ```bash
     gsutil cp sample_data.csv gs://chipfab-ai-training-data/predictions/
     ```
   - Check Cloud Functions logs for retraining trigger

---

## Environment Variables

### API Gateway
- `ENABLE_PUBSUB`: Enable/disable Pub/Sub (default: true)
- `PUBSUB_TOPIC_PREDICTIONS`: Topic name for predictions (default: chipfabai-predictions)
- `GCP_PROJECT` or `PROJECT_ID`: GCP project ID

### Anomaly Detector Function
- `GCP_PROJECT`: GCP project ID
- `ALERTS_TOPIC`: Topic for alerts (default: chipfabai-alerts)
- `ENABLE_ALERTS`: Enable alerting (default: true)

### Model Retrainer Function
- `GCP_PROJECT`: GCP project ID
- `REGION`: GCP region (default: europe-west4)
- `TRAINING_DATA_BUCKET`: Cloud Storage bucket name
- `MODEL_NAME`: Model name for retraining
- `MIN_SAMPLES_FOR_RETRAIN`: Minimum samples needed (default: 1000)

---

## Benefits for Hackathon

1. **Event-Driven Architecture**: Shows modern, scalable design
2. **Real-Time Processing**: Anomaly detection happens automatically
3. **Automated ML Pipeline**: Model retraining triggered by data
4. **ML Experimentation**: Vertex AI Workbench for advanced ML work
5. **Multiple GCP Services**: Demonstrates platform integration

---

## Cost Considerations

- **Pub/Sub**: Free tier includes 10GB/month
- **Cloud Functions**: Pay per invocation (very cheap)
- **Vertex AI Workbench**: Pay for compute time (can be expensive)
- **Cloud Storage**: Pay for storage and operations

For demo: All services scale to zero when idle, keeping costs minimal.

