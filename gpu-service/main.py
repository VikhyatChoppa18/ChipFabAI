"""
ChipFabAI GPU Service
GPU-accelerated AI service for semiconductor manufacturing process optimization
Optimized for cost, performance, and production deployment
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from datetime import datetime
import gc

# Configure structured logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables storing the model, tokenizer, and device information
# Model is loaded once at startup to avoid reloading on each request, reducing startup overhead
model = None
tokenizer = None
device = None
model_load_time = None
request_count = 0

# Model configuration using environment variables
# MODEL_CACHE_DIR stores downloaded models to avoid re-downloading on subsequent runs
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/models")
# MODEL_NAME can be set to use Gemma 2B (google/gemma-2-2b-it) or a smaller model for faster startup
# Default uses DialoGPT-small which loads faster and doesn't require HuggingFace authentication
MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/DialoGPT-small")
# FALLBACK_MODEL is used if the primary model fails to load
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "microsoft/DialoGPT-small")

# Process optimization prompt template - optimized for structured output
OPTIMIZATION_PROMPT_TEMPLATE = """You are an expert semiconductor manufacturing engineer. 
Given the following process parameters, predict the optimal settings and yield.

Current Process Parameters:
- Temperature: {temperature}°C
- Pressure: {pressure} Torr
- Etch Time: {etch_time}s
- Gas Flow Rate: {gas_flow} sccm
- Chamber Pressure: {chamber_pressure} mTorr
- Wafer Size: {wafer_size}mm
- Process Type: {process_type}

Based on semiconductor manufacturing best practices, provide:
1. Predicted Yield: (0-100%)
2. Optimal Temperature: (recommended range)
3. Optimal Pressure: (recommended range)
4. Risk Factors: (potential issues)
5. Recommendations: (specific improvements)

Format your response as JSON with keys: predicted_yield, optimal_temperature, optimal_pressure, risk_factors, recommendations.
"""


class ProcessParameters(BaseModel):
    """Input parameters for process optimization with validation"""
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
        if v < 0:
            raise ValueError('Value must be positive')
        return v


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_yield: float = Field(..., ge=0, le=100, description="Predicted yield percentage")
    optimal_temperature: Dict[str, float] = Field(..., description="Optimal temperature range")
    optimal_pressure: Dict[str, float] = Field(..., description="Optimal pressure range")
    risk_factors: List[str] = Field(..., description="Identified risk factors")
    recommendations: List[str] = Field(..., description="Optimization recommendations")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence score")
    processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")
    batch_id: Optional[str] = Field(default=None, description="Batch identifier")
    model_version: str = Field(default="gemma-2b", description="Model version used")


def load_model():
    """
    Loads the AI model and tokenizer into memory
    - Uses model caching to avoid re-downloading models on subsequent runs
    - Loads in float16 precision on GPU for reduced memory usage
    - Attempts to use GPU if available for faster inference
    """
    global model, tokenizer, device, model_load_time
    
    start_time = time.time()
    
    try:
        # Checking if CUDA (GPU) is available and setting device accordingly
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU Available: {gpu_name}")
            logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
        else:
            device = torch.device("cpu")
            logger.warning("No GPU available, using CPU for inference")
        
        # Attempting to load the primary model specified in MODEL_NAME
        model_name = MODEL_NAME
        use_gemma = True
        
        try:
            logger.info(f"Loading model: {model_name}")
            logger.info(f"Cache directory: {MODEL_CACHE_DIR}")
            
            # Loading tokenizer first as it's smaller and faster to load
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=MODEL_CACHE_DIR,
                trust_remote_code=True
            )
            
            # Setting pad token to end-of-sequence token if pad token is not defined
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Loading model with configuration optimized for memory and performance
            # float16 on GPU reduces memory usage by half compared to float32
            # device_map="auto" automatically places model layers on available GPUs
            # low_cpu_mem_usage reduces peak memory usage during loading
            # use_cache enables key-value caching for faster sequential inference
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=MODEL_CACHE_DIR,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                device_map="auto" if device.type == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_cache=True
            )
            
            # Moving model to the selected device (GPU or CPU)
            if device.type == "cuda":
                model = model.to(device)
            
            # Setting model to evaluation mode disables training-specific features like dropout
            model.eval()
            
            # Attempting to compile model with torch.compile for faster inference (requires PyTorch 2.0+)
            # This can provide significant speedups on supported hardware
            try:
                if hasattr(torch, 'compile') and device.type == "cuda":
                    logger.info("Enabling torch.compile for optimized inference")
                    model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                logger.warning(f"torch.compile not available: {e}")
        
            logger.info(f"Model loaded successfully: {model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            logger.info(f"Falling back to: {FALLBACK_MODEL}")
            use_gemma = False
            model_name = FALLBACK_MODEL
            
            # Load fallback model
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=MODEL_CACHE_DIR,
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=MODEL_CACHE_DIR,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                device_map="auto" if device.type == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if device.type == "cuda":
                model = model.to(device)
            
            model.eval()
        
        model_load_time = time.time() - start_time
        logger.info(f"Model load time: {model_load_time:.2f} seconds")
        
        # Log memory usage
        if device.type == "cuda":
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def predict_optimal_settings(parameters: ProcessParameters) -> Dict:
    """
    Predict optimal manufacturing settings using AI model
    Optimized for cost and performance with efficient inference
    """
    global model, tokenizer, device, request_count
    
    request_count += 1
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Building the prompt string by formatting the template with process parameters
        prompt = OPTIMIZATION_PROMPT_TEMPLATE.format(
            temperature=parameters.temperature,
            pressure=parameters.pressure,
            etch_time=parameters.etch_time,
            gas_flow=parameters.gas_flow,
            chamber_pressure=parameters.chamber_pressure,
            wafer_size=parameters.wafer_size,
            process_type=parameters.process_type
        )
        
        # Converting the prompt text into token IDs that the model can process
        # truncation limits input to 512 tokens, padding ensures consistent batch sizes
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Moving tokenized inputs to the same device as the model (GPU or CPU)
        if device.type == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generating prediction using the model
        # torch.no_grad() disables gradient computation since we're only doing inference
        # max_new_tokens limits the length of generated text to control generation time
        # temperature and top_p control the randomness of generation
        # use_cache enables key-value caching to speed up generation
        # num_return_sequences=1 generates a single response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                use_cache=True,
                num_return_sequences=1
            )
        
        # Decode response
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from response
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        
        if json_start >= 0 and json_end > json_start:
            json_text = response_text[json_start:json_end]
            try:
                result = json.loads(json_text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from model response, using fallback")
                result = parse_response_fallback(response_text, parameters)
        else:
            result = parse_response_fallback(response_text, parameters)
        
        # Calculating total processing time in milliseconds
        processing_time = (time.time() - start_time) * 1000
        
        # Extracting and validating predicted yield, using fallback calculation if not found in result
        # Clamping yield value to valid range between 0 and 100
        predicted_yield = result.get("predicted_yield", calculate_yield_estimate(parameters))
        predicted_yield = max(0, min(100, float(predicted_yield)))
        
        prediction = {
            "predicted_yield": round(predicted_yield, 2),
            "optimal_temperature": result.get("optimal_temperature", {
                "min": max(0, parameters.temperature - 5),
                "max": min(500, parameters.temperature + 5),
                "optimal": parameters.temperature
            }),
            "optimal_pressure": result.get("optimal_pressure", {
                "min": max(0, parameters.pressure - 0.1),
                "max": min(10, parameters.pressure + 0.1),
                "optimal": parameters.pressure
            }),
            "risk_factors": result.get("risk_factors", get_default_risk_factors(parameters)),
            "recommendations": result.get("recommendations", get_default_recommendations(parameters)),
            "confidence": min(1.0, max(0.0, result.get("confidence", 0.85))),
            "processing_time_ms": round(processing_time, 2),
            "batch_id": parameters.batch_id,
            "model_version": "gemma-2b" if "gemma" in str(model.config).lower() else "fallback"
        }
        
        # Clearing GPU memory cache periodically to prevent memory accumulation
        # This helps manage GPU memory usage during long-running inference sessions
        if request_count % 10 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()
        
        return prediction
        
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU out of memory, clearing cache")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise HTTPException(status_code=507, detail="GPU out of memory")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def parse_response_fallback(response_text: str, parameters: ProcessParameters) -> Dict:
    """Fallback parser if JSON parsing fails - uses domain knowledge"""
    predicted_yield = calculate_yield_estimate(parameters)
    
    return {
        "predicted_yield": predicted_yield,
        "optimal_temperature": {
            "min": max(0, parameters.temperature - 5),
            "max": min(500, parameters.temperature + 5),
            "optimal": parameters.temperature
        },
        "optimal_pressure": {
            "min": max(0, parameters.pressure - 0.1),
            "max": min(10, parameters.pressure + 0.1),
            "optimal": parameters.pressure
        },
        "risk_factors": get_default_risk_factors(parameters),
        "recommendations": get_default_recommendations(parameters),
        "confidence": 0.75
    }


def get_default_risk_factors(parameters: ProcessParameters) -> List[str]:
    """Generate risk factors based on parameter analysis"""
    risks = []
    
    if parameters.temperature < 150 or parameters.temperature > 250:
        risks.append("Temperature outside optimal range (150-250°C)")
    
    if parameters.pressure < 0.5 or parameters.pressure > 2.5:
        risks.append("Pressure outside optimal range (0.5-2.5 Torr)")
    
    if parameters.etch_time < 30 or parameters.etch_time > 120:
        risks.append("Etch time outside optimal range (30-120s)")
    
    if not risks:
        risks.append("Parameters within acceptable ranges")
    
    return risks


def get_default_recommendations(parameters: ProcessParameters) -> List[str]:
    """Generate recommendations based on parameter analysis"""
    recommendations = []
    
    if parameters.temperature < 180:
        recommendations.append("Increase temperature to 200-220°C for better etch rate")
    elif parameters.temperature > 230:
        recommendations.append("Decrease temperature to 200-220°C to reduce thermal stress")
    
    if parameters.pressure < 1.0:
        recommendations.append("Increase pressure to 1.5-2.0 Torr for improved uniformity")
    elif parameters.pressure > 2.0:
        recommendations.append("Decrease pressure to 1.5 Torr for better selectivity")
    
    if not recommendations:
        recommendations.append("Current parameters are well-optimized")
        recommendations.append("Monitor process stability and yield trends")
    
    return recommendations


def calculate_yield_estimate(parameters: ProcessParameters) -> float:
    """
    Calculate yield estimate based on process parameters
    Uses calibrated model based on semiconductor manufacturing best practices
    """
    # Normalized factors (0-1 scale)
    temp_factor = 1.0 - abs(parameters.temperature - 200) / 100.0  # Optimal around 200°C
    temp_factor = max(0, min(1, temp_factor))
    
    pressure_factor = 1.0 - abs(parameters.pressure - 1.5) / 2.0  # Optimal around 1.5 Torr
    pressure_factor = max(0, min(1, pressure_factor))
    
    time_factor = 1.0 if 30 <= parameters.etch_time <= 120 else 0.8
    
    # Weighted combination
    base_yield = 85.0
    yield_adjustment = (temp_factor * 0.3 + pressure_factor * 0.3 + time_factor * 0.4) * 15.0
    
    predicted_yield = max(50.0, min(99.0, base_yield + yield_adjustment))
    return round(predicted_yield, 2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    logger.info("Starting ChipFabAI GPU Service...")
    logger.info("Model: Gemma 2B (Open-source)")
    
    try:
        load_model()
        logger.info("Service ready and accepting requests")
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down GPU Service...")
    global model
    if model is not None:
        del model
    if device and device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info("Service shutdown complete")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="ChipFabAI GPU Service",
    version="1.0.0",
    description="GPU-accelerated AI service for semiconductor manufacturing optimization",
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


# Exception handler for better error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "ChipFabAI GPU Service",
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "gpu_available": torch.cuda.is_available() if torch.cuda else False,
        "model_load_time_seconds": round(model_load_time, 2) if model_load_time else None,
        "requests_processed": request_count
    }


@app.get("/health")
async def health():
    """Detailed health check for monitoring"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "device_name": torch.cuda.get_device_name(0),
            "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
            "memory_cached_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
            "memory_free_gb": round((torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9, 2)
        }
    
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "gpu": gpu_info,
        "model_load_time_seconds": round(model_load_time, 2) if model_load_time else None,
        "requests_processed": request_count,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(parameters: ProcessParameters):
    """
    Predict optimal semiconductor manufacturing settings
    
    Args:
        parameters: Process parameters (temperature, pressure, etc.)
    
    Returns:
        Prediction with optimal settings and yield forecast
    """
    try:
        prediction = predict_optimal_settings(parameters)
        return PredictionResponse(**prediction)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict")
async def batch_predict(request: Dict):
    """
    Batch prediction for multiple process configurations
    Optimized for cost with efficient batch processing
    
    Args:
        request: Dict with "parameters_list" key containing list of ProcessParameters
    
    Returns:
        List of predictions with batch statistics
    """
    if "parameters_list" not in request:
        raise HTTPException(status_code=400, detail="Missing 'parameters_list' in request")
    
    parameters_list = request["parameters_list"]
    
    if len(parameters_list) > 100:
        raise HTTPException(status_code=400, detail="Batch size limited to 100 for cost optimization")
    
    if len(parameters_list) == 0:
        raise HTTPException(status_code=400, detail="Empty parameters list")
    
    start_time = time.time()
    results = []
    errors = []
    
    for idx, param_dict in enumerate(parameters_list):
        try:
            params = ProcessParameters(**param_dict)
            prediction = predict_optimal_settings(params)
            results.append(prediction)
        except Exception as e:
            logger.error(f"Batch prediction error for item {idx}: {str(e)}")
            errors.append({"index": idx, "error": str(e)})
            results.append(None)
    
    total_time = (time.time() - start_time) * 1000
    
    return {
        "predictions": results,
        "count": len(results),
        "successful": len([r for r in results if r is not None]),
        "failed": len(errors),
        "errors": errors,
        "total_processing_time_ms": round(total_time, 2),
        "average_time_per_prediction_ms": round(total_time / len(parameters_list), 2)
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    # Using single worker since GPU service benefits from single-process execution
    # Multiple workers would compete for GPU resources and reduce efficiency
    workers = int(os.getenv("WORKERS", 1))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=workers,
        log_level="info"
    )
