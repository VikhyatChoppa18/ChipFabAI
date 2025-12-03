"""
Cloud Function for Real-Time Anomaly Detection
Triggered by Pub/Sub messages from API Gateway when predictions are made
Detects anomalies in process parameters and sends alerts
"""

import json
import os
import base64
import logging
from google.cloud import pubsub_v1
from google.cloud import monitoring_v3
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ID = os.getenv("GCP_PROJECT")
TOPIC_ALERTS = os.getenv("ALERTS_TOPIC", "chipfabai-alerts")
ENABLE_ALERTS = os.getenv("ENABLE_ALERTS", "true").lower() == "true"

# Publisher for sending alerts
publisher = pubsub_v1.PublisherClient() if ENABLE_ALERTS else None


def detect_anomaly(parameters: dict, prediction: dict) -> bool:
    """
    Detect anomalies in process parameters or predictions
    I use simple threshold-based detection - in production you'd use ML models
    """
    # Check for parameter anomalies
    temp = parameters.get("temperature", 0)
    pressure = parameters.get("pressure", 0)
    yield_pred = prediction.get("predicted_yield", 0)
    
    # Anomaly conditions
    anomalies = []
    
    # Temperature out of safe range
    if temp < 150 or temp > 250:
        anomalies.append(f"Temperature anomaly: {temp}°C (safe range: 150-250°C)")
    
    # Pressure out of safe range
    if pressure < 0.5 or pressure > 2.5:
        anomalies.append(f"Pressure anomaly: {pressure} Torr (safe range: 0.5-2.5 Torr)")
    
    # Yield prediction too low
    if yield_pred < 70:
        anomalies.append(f"Low yield prediction: {yield_pred}% (threshold: 70%)")
    
    # High risk factors
    risk_factors = prediction.get("risk_factors", [])
    if len(risk_factors) > 2:
        anomalies.append(f"Multiple risk factors detected: {len(risk_factors)}")
    
    return len(anomalies) > 0, anomalies


def send_alert(anomalies: list, parameters: dict, prediction: dict):
    """Send alert via Pub/Sub to alerting topic"""
    if not publisher or not ENABLE_ALERTS:
        return
    
    try:
        alert_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "anomaly_detected",
            "anomalies": anomalies,
            "parameters": parameters,
            "prediction": {
                "predicted_yield": prediction.get("predicted_yield"),
                "confidence": prediction.get("confidence")
            },
            "severity": "high" if len(anomalies) > 2 else "medium"
        }
        
        topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ALERTS)
        message_data = json.dumps(alert_data).encode("utf-8")
        
        future = publisher.publish(topic_path, message_data)
        message_id = future.result()
        logger.info(f"Alert sent: {message_id}")
        
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")


def anomaly_detector(event, context):
    """
    Cloud Function entry point - triggered by Pub/Sub
    Receives prediction events and detects anomalies
    """
    try:
        # Decode Pub/Sub message
        if 'data' in event:
            message_data = json.loads(base64.b64decode(event['data']).decode('utf-8'))
        else:
            message_data = event
        
        parameters = message_data.get("parameters", {})
        prediction = message_data.get("prediction", {})
        
        logger.info(f"Processing prediction event for parameters: {parameters.get('temperature')}°C")
        
        # Detect anomalies
        has_anomaly, anomalies = detect_anomaly(parameters, prediction)
        
        if has_anomaly:
            logger.warning(f"Anomaly detected: {anomalies}")
            send_alert(anomalies, parameters, prediction)
            
            return {
                "status": "anomaly_detected",
                "anomalies": anomalies,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            logger.info("No anomalies detected")
            return {
                "status": "normal",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}", exc_info=True)
        raise

