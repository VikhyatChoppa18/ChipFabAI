"""
Cloud Function for Triggering Model Retraining
Triggered by Cloud Storage events when new training data is uploaded
Or triggered by Pub/Sub when enough new predictions have been collected
"""

import json
import os
import base64
import logging
from google.cloud import storage
from google.cloud import aiplatform
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ID = os.getenv("GCP_PROJECT")
REGION = os.getenv("REGION", "europe-west4")
BUCKET_NAME = os.getenv("TRAINING_DATA_BUCKET", "chipfabai-training-data")
MODEL_NAME = os.getenv("MODEL_NAME", "chipfabai-yield-predictor")
MIN_SAMPLES_FOR_RETRAIN = int(os.getenv("MIN_SAMPLES_FOR_RETRAIN", "1000"))


def check_retraining_conditions():
    """
    Check if we have enough new data to trigger retraining
    I count new prediction samples in Cloud Storage
    """
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # Count prediction files in the bucket
        blobs = bucket.list_blobs(prefix="predictions/")
        count = sum(1 for _ in blobs)
        
        logger.info(f"Found {count} prediction samples in storage")
        
        if count >= MIN_SAMPLES_FOR_RETRAIN:
            return True, count
        else:
            return False, count
            
    except Exception as e:
        logger.error(f"Error checking retraining conditions: {e}")
        return False, 0


def trigger_vertex_ai_training():
    """
    Trigger a Vertex AI training job
    In a real implementation, this would start a custom training job
    For now, I log that retraining should be triggered
    """
    try:
        logger.info(f"Triggering model retraining for {MODEL_NAME}")
        logger.info("In production, this would:")
        logger.info("  1. Create a Vertex AI Custom Training Job")
        logger.info("  2. Use the new training data from Cloud Storage")
        logger.info("  3. Train a new model version")
        logger.info("  4. Deploy the new model to Cloud Run")
        
        # Example: This would trigger a Vertex AI Pipeline
        # pipeline = aiplatform.PipelineJob(
        #     display_name="chipfabai-retraining",
        #     template_path="gs://.../pipeline.json",
        #     parameter_values={"training_data": f"gs://{BUCKET_NAME}/predictions/"}
        # )
        # pipeline.run()
        
        return {
            "status": "retraining_triggered",
            "model_name": MODEL_NAME,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error triggering retraining: {e}")
        raise


def model_retrainer(event, context):
    """
    Cloud Function entry point
    Can be triggered by:
    1. Cloud Storage event (new training data uploaded)
    2. Pub/Sub message (scheduled retraining check)
    """
    try:
        # Handle Cloud Storage trigger
        if 'bucket' in event:
            bucket_name = event['bucket']
            file_name = event['name']
            
            logger.info(f"New file uploaded: gs://{bucket_name}/{file_name}")
            
            # Check if we should retrain
            should_retrain, sample_count = check_retraining_conditions()
            
            if should_retrain:
                logger.info(f"Enough samples ({sample_count}) - triggering retraining")
                return trigger_vertex_ai_training()
            else:
                logger.info(f"Not enough samples ({sample_count}/{MIN_SAMPLES_FOR_RETRAIN})")
                return {
                    "status": "waiting_for_more_data",
                    "current_samples": sample_count,
                    "required_samples": MIN_SAMPLES_FOR_RETRAIN
                }
        
        # Handle Pub/Sub trigger (scheduled check)
        elif 'data' in event:
            message_data = json.loads(base64.b64decode(event['data']).decode('utf-8'))
            logger.info("Scheduled retraining check triggered")
            
            should_retrain, sample_count = check_retraining_conditions()
            
            if should_retrain:
                return trigger_vertex_ai_training()
            else:
                return {
                    "status": "no_retraining_needed",
                    "current_samples": sample_count
                }
        
        else:
            # Manual trigger
            logger.info("Manual retraining check")
            should_retrain, sample_count = check_retraining_conditions()
            
            if should_retrain:
                return trigger_vertex_ai_training()
            else:
                return {
                    "status": "no_retraining_needed",
                    "current_samples": sample_count
                }
                
    except Exception as e:
        logger.error(f"Error in model retrainer: {e}", exc_info=True)
        raise

