"""
API Gateway for ChipFabAI - the entry point that handles all client requests
This sits in front of the GPU service and adds caching, load balancing, and retry logic
I built this to handle production traffic reliably without hammering the GPU service
"""

import os
import logging
import time
import asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import httpx
from datetime import datetime, timezone
import json
from contextlib import asynccontextmanager

# Pub/Sub integration for event streaming
# This enables real-time event processing for anomaly detection and data collection
try:
    from google.cloud import pubsub_v1
    PUBSUB_ENABLED = os.getenv("ENABLE_PUBSUB", "true").lower() == "true"
except ImportError:
    PUBSUB_ENABLED = False
    logger.warning("Pub/Sub not available - install google-cloud-pubsub")

# Import the caching and load balancing modules I built
from cache import prediction_cache, get_cache_key, cached_predict
from load_balancer import LoadBalancer, LoadBalancingStrategy

# Logging setup - helps me debug issues in production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration - all of these can be overridden via environment variables
# This makes it easy to deploy in different environments without code changes
GPU_SERVICE_URL = os.getenv("GPU_SERVICE_URL", "http://localhost:8080")
STORAGE_BUCKET = os.getenv("STORAGE_BUCKET", "chipfabai-data")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60.0"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))

# Load balancing configuration - when you have multiple GPU service instances
# This distributes requests across them intelligently based on health and load
ENABLE_LOAD_BALANCING = os.getenv("ENABLE_LOAD_BALANCING", "false").lower() == "true"
GPU_SERVICE_URLS = os.getenv("GPU_SERVICE_URLS", GPU_SERVICE_URL).split(",")
LOAD_BALANCING_STRATEGY = os.getenv("LOAD_BALANCING_STRATEGY", "health_based")

# Caching configuration - this is crucial for performance
# When someone requests the same parameters twice, we return cached results instantly
# 5 minutes TTL is a good balance - not too stale, but long enough to help
ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))

# Pub/Sub configuration for event streaming
# I use Pub/Sub to publish prediction events for real-time anomaly detection
# This enables event-driven architecture without blocking the main request flow
PUBSUB_PROJECT_ID = os.getenv("GCP_PROJECT", os.getenv("PROJECT_ID", ""))
PUBSUB_TOPIC_PREDICTIONS = os.getenv("PUBSUB_TOPIC_PREDICTIONS", "chipfabai-predictions")
pubsub_publisher: Optional[pubsub_v1.PublisherClient] = None

# Global state that persists across requests
request_count = 0
load_balancer: Optional[LoadBalancer] = None
http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown lifecycle of the API Gateway
    I use this to initialize resources once at startup instead of per-request
    """
    global load_balancer, http_client
    
    # Startup sequence
    logger.info("Starting ChipFabAI API Gateway...")
    logger.info("Features: Caching, Load Balancing, Connection Pooling")
    
    # Create the HTTP client with connection pooling
    # Connection pooling is huge for performance - reusing connections instead of
    # creating new ones for every request saves a ton of overhead
    # I tuned these numbers based on real traffic patterns
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0),
        limits=httpx.Limits(
            max_keepalive_connections=50,  # Keep 50 connections warm
            max_connections=100,  # Allow up to 100 concurrent connections
            keepalive_expiry=30.0  # Keep idle connections alive for 30s
        )
    )
    
    # Set up load balancing if we have multiple GPU service instances
    # This is useful when you scale horizontally - multiple GPU services behind the gateway
    if ENABLE_LOAD_BALANCING and len(GPU_SERVICE_URLS) > 1:
        strategy_map = {
            "round_robin": LoadBalancingStrategy.ROUND_ROBIN,
            "least_connections": LoadBalancingStrategy.LEAST_CONNECTIONS,
            "health_based": LoadBalancingStrategy.HEALTH_BASED,
            "random": LoadBalancingStrategy.RANDOM
        }
        strategy = strategy_map.get(LOAD_BALANCING_STRATEGY, LoadBalancingStrategy.HEALTH_BASED)
        
        load_balancer = LoadBalancer(
            backend_urls=GPU_SERVICE_URLS,
            strategy=strategy,
            health_check_interval=30,
            health_check_timeout=5.0,
            max_failures=3
        )
        await load_balancer.start_health_checks()
        logger.info(f"Load balancer initialized with {len(GPU_SERVICE_URLS)} backends (strategy: {strategy.value})")
    else:
        logger.info(f"Single GPU service mode: {GPU_SERVICE_URL}")
    
    # Start the background task that cleans up expired cache entries
    # Without this, the cache would grow indefinitely with stale entries
    if ENABLE_CACHING:
        cleanup_task = asyncio.create_task(cache_cleanup_task())
        logger.info(f"Caching enabled (TTL: {CACHE_TTL}s)")
    
    # Initialize Pub/Sub publisher for event streaming
    # I do this asynchronously to avoid blocking startup if Pub/Sub is slow
    global pubsub_publisher
    if PUBSUB_ENABLED and PUBSUB_PROJECT_ID:
        try:
            pubsub_publisher = pubsub_v1.PublisherClient()
            logger.info(f"Pub/Sub enabled - publishing to topic: {PUBSUB_TOPIC_PREDICTIONS}")
        except Exception as e:
            logger.warning(f"Failed to initialize Pub/Sub: {e} - continuing without it")
            pubsub_publisher = None
    else:
        logger.info("Pub/Sub disabled or PROJECT_ID not set - continuing without event streaming")
    
    # Log the port we're listening on - Cloud Run sets PORT env var
    port = int(os.getenv("PORT", 8080))
    logger.info(f"API Gateway ready and accepting requests on port {port}")
    
    yield
    
    # Clean shutdown - close connections gracefully
    logger.info("Shutting down API Gateway...")
    if load_balancer:
        await load_balancer.close()
    if http_client:
        await http_client.aclose()
    logger.info("API Gateway shutdown complete")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="ChipFabAI API Gateway",
    version="2.0.0",
    description="API Gateway for ChipFabAI with caching, load balancing, and performance optimizations",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def cache_cleanup_task():
    """
    Background task that runs every minute to remove expired cache entries
    I run this in the background so it doesn't block request handling
    """
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
            prediction_cache.cleanup_expired()
        except Exception as e:
            logger.error(f"Error in cache cleanup task: {e}")


class ProcessParameters(BaseModel):
    """
    Input parameters for semiconductor manufacturing process optimization
    I use Pydantic here because it gives me automatic validation and nice error messages
    The Field constraints ensure we reject invalid inputs before they hit the GPU service
    """
    temperature: float = Field(..., ge=0, le=500, description="Temperature in Celsius")
    pressure: float = Field(..., ge=0, le=10, description="Pressure in Torr")
    etch_time: float = Field(..., ge=0, le=300, description="Etch time in seconds")
    gas_flow: float = Field(..., ge=0, le=1000, description="Gas flow rate in sccm")
    chamber_pressure: float = Field(..., ge=0, le=100, description="Chamber pressure in mTorr")
    wafer_size: int = Field(default=300, ge=100, le=450, description="Wafer size in mm")
    process_type: str = Field(default="etching", description="Process type")
    batch_id: Optional[str] = Field(default=None, description="Batch identifier for tracking")

    @validator('temperature', 'pressure', 'etch_time', 'gas_flow', 'chamber_pressure')
    def validate_positive(cls, v):
        # Double-check that values are positive - can't have negative pressure or time
        if v < 0:
            raise ValueError('Value must be positive')
        return v


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_yield: float = Field(..., ge=0, le=100, description="Predicted yield percentage")
    optimal_temperature: dict = Field(..., description="Optimal temperature range")
    optimal_pressure: dict = Field(..., description="Optimal pressure range")
    risk_factors: list = Field(..., description="Identified risk factors")
    recommendations: list = Field(..., description="Optimization recommendations")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence score")
    processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")
    batch_id: Optional[str] = Field(default=None, description="Batch identifier")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    parameters_list: List[ProcessParameters] = Field(..., min_items=1, max_items=100)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    count: int
    successful: int
    failed: int
    total_processing_time_ms: float
    average_time_per_prediction_ms: float


async def check_gpu_service_health() -> dict:
    """
    Check if the GPU service is healthy and responding
    I use this in the /health endpoint to report system status
    If load balancing is enabled, I check all backends; otherwise just the single service
    """
    if http_client is None:
        return {"status": "uninitialized", "error": "HTTP client not initialized"}
    
    # If we have load balancing enabled, use the load balancer's health info
    # It already tracks which backends are healthy, so no need to duplicate that logic
    if load_balancer and ENABLE_LOAD_BALANCING:
        stats = load_balancer.get_stats()
        healthy_backends = [b for b in stats.get("backends", []) if b.get("healthy", False)]
        if healthy_backends:
            return {
                "status": "healthy",
                "healthy_backends": len(healthy_backends),
                "total_backends": stats.get("total_backends", 0)
            }
        else:
            return {"status": "unhealthy", "error": "No healthy backends available"}
    
    # Single service mode - just hit the health endpoint directly
    # I retry a few times because network hiccups happen
    for attempt in range(MAX_RETRIES):
        try:
            response = await http_client.get(f"{GPU_SERVICE_URL}/health", timeout=5.0)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "unhealthy", "status_code": response.status_code}
        except httpx.TimeoutException:
            logger.warning(f"GPU service health check timeout (attempt {attempt + 1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
        except Exception as e:
            logger.error(f"GPU service health check failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
    
    return {"status": "unreachable", "error": "Service unavailable after retries"}


async def call_gpu_service(endpoint: str, data: dict, timeout: float = None) -> dict:
    """
    Make a request to the GPU service with proper error handling and retries
    This is where all the resilience logic lives - retries, timeouts, load balancing
    I use exponential backoff on retries to avoid hammering a struggling service
    """
    timeout = timeout or REQUEST_TIMEOUT
    
    # If load balancing is enabled, let the load balancer handle request routing
    # It knows which backends are healthy and will pick the best one
    if load_balancer and ENABLE_LOAD_BALANCING:
        for attempt in range(MAX_RETRIES):
            try:
                response_data, backend = await load_balancer.execute_request(
                    method="POST",
                    endpoint=endpoint,
                    data=data,
                    timeout=timeout
                )
                if response_data:
                    return response_data
                elif attempt < MAX_RETRIES - 1:
                    # Exponential backoff - wait longer each retry
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                else:
                    raise HTTPException(
                        status_code=503,
                        detail="All GPU service backends unavailable after retries"
                    )
            except Exception as e:
                logger.error(f"Load balanced request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    # Single service mode - direct call to the GPU service
    url = f"{GPU_SERVICE_URL}/{endpoint.lstrip('/')}"
    
    for attempt in range(MAX_RETRIES):
        try:
            response = await http_client.post(
                url,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            logger.warning(f"GPU service timeout (attempt {attempt + 1}/{MAX_RETRIES}): {endpoint}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
            else:
                raise HTTPException(
                    status_code=504,
                    detail=f"GPU service timeout after {MAX_RETRIES} attempts"
                )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                # Service unavailable - might be temporary, so retry
                logger.warning(f"GPU service unavailable (attempt {attempt + 1}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise HTTPException(
                        status_code=503,
                        detail="GPU service unavailable after retries"
                    )
            else:
                # Other HTTP errors - probably not retryable
                logger.error(f"GPU service error: {e.response.status_code} - {e.response.text}")
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=f"GPU service error: {e.response.text[:200]}"
                )
        except Exception as e:
            logger.error(f"Unexpected error calling GPU service: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    raise HTTPException(status_code=500, detail="Failed to call GPU service")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for better error responses"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


@app.get("/")
async def root():
    """Health check endpoint"""
    global request_count
    request_count += 1
    
    return {
        "service": "ChipFabAI API Gateway",
        "status": "healthy",
        "gpu_service_url": GPU_SERVICE_URL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "requests_processed": request_count
    }


@app.get("/health")
async def health():
    """Detailed health check with GPU service connectivity and system metrics"""
    gpu_service_health = await check_gpu_service_health()
    
    gateway_status = "healthy"
    if gpu_service_health.get("status") not in ["healthy", "operational"]:
        gateway_status = "degraded"
    
    # Get cache statistics
    cache_stats = prediction_cache.get_stats() if ENABLE_CACHING else None
    
    # Get load balancer statistics
    lb_stats = load_balancer.get_stats() if load_balancer else None
    
    return {
        "status": gateway_status,
        "gateway": "operational",
        "gpu_service": gpu_service_health,
        "cache": cache_stats,
        "load_balancer": lb_stats,
        "features": {
            "caching_enabled": ENABLE_CACHING,
            "load_balancing_enabled": ENABLE_LOAD_BALANCING and load_balancer is not None,
            "connection_pooling": True,
            "pubsub_enabled": PUBSUB_ENABLED and pubsub_publisher is not None,
            "event_streaming": PUBSUB_ENABLED
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "requests_processed": request_count
    }


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(parameters: ProcessParameters):
    """
    Main prediction endpoint - this is where clients send their process parameters
    The caching here is key - if someone requests the same parameters twice,
    we return the cached result instantly instead of hitting the GPU service again
    This makes a huge difference for repeated queries during optimization workflows
    """
    global request_count
    request_count += 1
    
    start_time = time.time()
    params_dict = parameters.model_dump()
    
    try:
        # First, check if we have this exact request cached
        # The cache key is a hash of the parameters, so identical requests hit the cache
        if ENABLE_CACHING:
            cache_key = get_cache_key(params_dict)
            cached_result = prediction_cache.get(cache_key)
            
            if cached_result is not None:
                logger.info(f"Cache HIT for prediction (key: {cache_key[:8]}...)")
                processing_time = (time.time() - start_time) * 1000
                cached_result["gateway_processing_time_ms"] = round(processing_time, 2)
                cached_result["_cached"] = True
                # Remove the internal cache flag before returning to client
                cached_result.pop("_cached", None)
                return PredictionResponse(**cached_result)
        
        # Cache miss - need to actually call the GPU service
        logger.debug(f"Cache MISS or caching disabled - calling GPU service")
        result = await call_gpu_service("predict", params_dict)
        processing_time = (time.time() - start_time) * 1000
        
        # Track how long the gateway took (separate from GPU processing time)
        result["gateway_processing_time_ms"] = round(processing_time, 2)
        
        # Store the result in cache for next time
        if ENABLE_CACHING:
            cache_key = get_cache_key(params_dict)
            prediction_cache.set(cache_key, result, ttl=CACHE_TTL)
            logger.debug(f"Cached prediction result (key: {cache_key[:8]}...)")
        
        # Publish prediction event to Pub/Sub for real-time processing
        # This enables anomaly detection and data collection for retraining
        if pubsub_publisher and PUBSUB_ENABLED:
            try:
                event_data = {
                    "parameters": params_dict,
                    "prediction": result,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "request_id": f"req-{request_count}"
                }
                topic_path = pubsub_publisher.topic_path(PUBSUB_PROJECT_ID, PUBSUB_TOPIC_PREDICTIONS)
                message_data = json.dumps(event_data).encode("utf-8")
                # Publish asynchronously - don't block the response
                pubsub_publisher.publish(topic_path, message_data)
                logger.debug("Published prediction event to Pub/Sub")
            except Exception as e:
                # Don't fail the request if Pub/Sub fails
                logger.warning(f"Failed to publish to Pub/Sub: {e}")
        
        return PredictionResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Batch prediction for multiple process configurations
    
    Args:
        request: Batch prediction request with parameters list
    
    Returns:
        Batch prediction response
    """
    global request_count
    request_count += 1
    
    start_time = time.time()
    
    try:
        result = await call_gpu_service(
            "batch-predict",
            {"parameters_list": [p.model_dump() for p in request.parameters_list]},
            timeout=300.0  # Longer timeout for batch processing
        )
        
        total_time = (time.time() - start_time) * 1000
        
        # Convert results to PredictionResponse objects
        predictions = []
        successful = 0
        failed = 0
        
        for pred_data in result.get("predictions", []):
            if pred_data is not None:
                try:
                    predictions.append(PredictionResponse(**pred_data))
                    successful += 1
                except Exception as e:
                    logger.error(f"Failed to parse prediction: {e}")
                    failed += 1
            else:
                failed += 1
        
        return BatchPredictionResponse(
            predictions=predictions,
            count=result.get("count", len(predictions)),
            successful=successful,
            failed=failed,
            total_processing_time_ms=round(total_time, 2),
            average_time_per_prediction_ms=round(total_time / len(request.parameters_list), 2) if request.parameters_list else 0
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/sample-data")
async def get_sample_data():
    """Get sample process data for testing"""
    return {
        "sample_parameters": {
            "temperature": 200.0,
            "pressure": 1.5,
            "etch_time": 60.0,
            "gas_flow": 100.0,
            "chamber_pressure": 5.0,
            "wafer_size": 300,
            "process_type": "etching"
        },
        "description": "Sample semiconductor manufacturing parameters for testing",
        "usage": "Use these parameters to test the prediction endpoint"
    }


@app.post("/api/v1/optimize")
async def optimize_process(parameters: ProcessParameters):
    """
    Optimize process with enhanced analysis
    
    Args:
        parameters: Process parameters
    
    Returns:
        Enhanced optimization results with cost savings estimates
    """
    global request_count
    request_count += 1
    
    try:
        # Get prediction
        prediction = await predict(parameters)
        
        # Calculate optimization score
        optimization_score = calculate_optimization_score(prediction)
        
        # Enhanced response
        return {
            **prediction.model_dump(),
            "optimization_score": optimization_score,
            "improvement_potential": calculate_improvement_potential(prediction),
            "cost_savings_estimate": calculate_cost_savings(prediction),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def calculate_optimization_score(prediction: PredictionResponse) -> float:
    """
    Calculate an overall optimization score from 0-100
    I weight yield at 60% and confidence at 40% - yield is more important,
    but confidence tells us how reliable the prediction is
    """
    yield_score = prediction.predicted_yield * 0.6
    confidence_score = prediction.confidence * 100 * 0.4
    return round(yield_score + confidence_score, 2)


def calculate_improvement_potential(prediction: PredictionResponse) -> dict:
    """Calculate improvement potential based on current yield"""
    current_yield = prediction.predicted_yield
    optimal_yield = min(99.0, current_yield + 10.0)  # Assume 10% improvement potential
    
    return {
        "current_yield": current_yield,
        "optimal_yield": optimal_yield,
        "improvement_percentage": round((optimal_yield - current_yield), 2)
    }


def calculate_cost_savings(prediction: PredictionResponse) -> dict:
    """
    Estimate the cost savings from optimizing the process
    This is a simplified model - in reality, wafer costs vary a lot by process node
    But it gives customers a sense of the financial impact of optimization
    The math: lower yield means more wafers needed per good die, which costs more
    """
    # Typical cost per wafer for a medium-sized fab (this is a rough estimate)
    base_cost_per_wafer = 1000
    current_yield = prediction.predicted_yield
    optimal_yield = min(99.0, current_yield + 10.0)  # Assume we can improve by 10%
    
    # Cost per good die = cost per wafer / yield percentage
    # Lower yield means you need more wafers to get the same number of good dies
    current_cost = base_cost_per_wafer / (current_yield / 100)
    optimal_cost = base_cost_per_wafer / (optimal_yield / 100)
    
    savings_per_wafer = current_cost - optimal_cost
    savings_per_million = savings_per_wafer * 1000000
    
    return {
        "savings_per_wafer_usd": round(savings_per_wafer, 2),
        "savings_per_million_wafers_usd": round(savings_per_million, 2),
        "roi_percentage": round((savings_per_wafer / base_cost_per_wafer) * 100, 2)
    }


@app.get("/api/v1/metrics")
async def get_metrics():
    """Get performance metrics and statistics"""
    cache_stats = prediction_cache.get_stats() if ENABLE_CACHING else None
    lb_stats = load_balancer.get_stats() if load_balancer else None
    
    return {
        "requests_processed": request_count,
        "cache": cache_stats,
        "load_balancer": lb_stats,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    
    # Cloud Run provides PORT environment variable - must use it
    # Default to 8080 for Cloud Run compatibility
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Starting API Gateway on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
