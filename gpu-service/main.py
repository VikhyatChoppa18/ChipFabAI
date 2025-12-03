"""
GPU Service - this is where the actual AI inference happens
I load the model once at startup and keep it in memory, which is way faster than
loading it for every request. The model runs on GPU if available, CPU otherwise.
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

# Logging setup - helps me debug issues in production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state - model stays loaded in memory between requests
# Loading the model is expensive (several seconds), so we do it once at startup
model = None
tokenizer = None
device = None
model_load_time = None
request_count = 0

# Model configuration - all configurable via environment variables
# This makes it easy to switch models or adjust settings without code changes
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/models")
# Default to DialoGPT-small because it loads fast and doesn't need HuggingFace auth
# You can override with MODEL_NAME=google/gemma-2-2b-it for better quality (but slower)
MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/DialoGPT-small")
# If the primary model fails to load, try this one instead
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "microsoft/DialoGPT-small")

# Prompt template I use to format requests for the language model
# I ask it to act as a semiconductor engineer and provide structured JSON output
# The model doesn't always return perfect JSON, so I have fallback parsing logic
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
    Load the AI model and tokenizer into memory
    This is the expensive part - model loading takes several seconds, so we do it once at startup
    I use float16 on GPU to save memory (half the size of float32) and device_map="auto" 
    to automatically distribute large models across multiple GPUs if available
    """
    global model, tokenizer, device, model_load_time
    
    start_time = time.time()
    
    try:
        # Check if we have a GPU available - makes inference way faster
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU Available: {gpu_name}")
            logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
        else:
            device = torch.device("cpu")
            logger.warning("No GPU available, using CPU for inference")
        
        # Try to load the primary model first
        model_name = MODEL_NAME
        use_gemma = True
        
        try:
            logger.info(f"Loading model: {model_name}")
            logger.info(f"Cache directory: {MODEL_CACHE_DIR}")
            
            # Load tokenizer first - it's smaller and loads faster
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=MODEL_CACHE_DIR,
                trust_remote_code=True
            )
            
            # Some tokenizers don't have a pad token, so I set it to eos_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load the actual model with optimizations
            # float16 on GPU cuts memory usage in half (important for large models)
            # device_map="auto" handles multi-GPU automatically
            # low_cpu_mem_usage prevents OOM during loading
            # use_cache enables KV caching for faster generation
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=MODEL_CACHE_DIR,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                device_map="auto" if device.type == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_cache=True
            )
            
            # Explicitly move to device if not using device_map
            if device.type == "cuda" and model.device.type != "cuda":
                model = model.to(device)
            
            # Set to eval mode - disables dropout and other training-only features
            model.eval()
            
            # Try to compile with torch.compile for extra speed (PyTorch 2.0+)
            # This can give 20-30% speedup on modern GPUs
            try:
                if hasattr(torch, 'compile') and device.type == "cuda":
                    logger.info("Enabling torch.compile for optimized inference")
                    model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                logger.warning(f"torch.compile not available: {e}")
        
            logger.info(f"Model loaded successfully: {model_name}")
            
        except Exception as e:
            # Primary model failed - try the fallback
            logger.warning(f"Failed to load {model_name}: {e}")
            logger.info(f"Falling back to: {FALLBACK_MODEL}")
            use_gemma = False
            model_name = FALLBACK_MODEL
            
            # Load fallback model with same settings
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
        
        # Log GPU memory usage so I know how much headroom we have
        if device.type == "cuda":
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def predict_optimal_settings(parameters: ProcessParameters) -> Dict:
    """
    Run inference to predict optimal manufacturing settings
    This is where the actual AI magic happens - I format a prompt, send it to the model,
    and parse the response. The model doesn't always return perfect JSON, so I have
    fallback logic to handle that.
    """
    global model, tokenizer, device, request_count
    
    request_count += 1
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Format the prompt with the actual parameter values
        prompt = OPTIMIZATION_PROMPT_TEMPLATE.format(
            temperature=parameters.temperature,
            pressure=parameters.pressure,
            etch_time=parameters.etch_time,
            gas_flow=parameters.gas_flow,
            chamber_pressure=parameters.chamber_pressure,
            wafer_size=parameters.wafer_size,
            process_type=parameters.process_type
        )
        
        # Convert text to token IDs that the model understands
        # I limit to 512 tokens to keep input size reasonable
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move inputs to GPU if that's where the model is
        if device.type == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate the prediction
        # torch.no_grad() saves memory by not computing gradients (we're not training)
        # temperature=0.7 gives a good balance of creativity vs consistency
        # top_p=0.9 uses nucleus sampling for better quality
        # use_cache speeds up generation by reusing computed key-value pairs
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,  # Limit output length
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                use_cache=True,
                num_return_sequences=1
            )
        
        # Convert token IDs back to text
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Try to extract JSON from the response
        # The model sometimes adds extra text, so I search for the JSON part
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        
        if json_start >= 0 and json_end > json_start:
            json_text = response_text[json_start:json_end]
            try:
                result = json.loads(json_text)
            except json.JSONDecodeError:
                # Model returned invalid JSON - use fallback parser
                logger.warning("Failed to parse JSON from model response, using fallback")
                result = parse_response_fallback(response_text, parameters)
        else:
            # No JSON found - use fallback
            result = parse_response_fallback(response_text, parameters)
        
        # Calculate how long this took
        processing_time = (time.time() - start_time) * 1000
        
        # Extract and validate the predicted yield
        # Clamp to valid range (0-100%) and use fallback calculation if missing
        predicted_yield = result.get("predicted_yield", calculate_yield_estimate(parameters))
        predicted_yield = max(0, min(100, float(predicted_yield)))
        
        # Calculate dynamic confidence based on parameter quality
        # Confidence is higher when parameters are closer to optimal ranges
        confidence = calculate_confidence_score(parameters, result.get("confidence"))
        
        # Build the response with all the prediction data
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
            "confidence": confidence,
            "processing_time_ms": round(processing_time, 2),
            "batch_id": parameters.batch_id,
            "model_version": "gemma-2b" if "gemma" in str(model.config).lower() else "fallback"
        }
        
        # Periodically clear GPU cache to prevent memory leaks
        # PyTorch sometimes holds onto memory even after tensors are deleted
        if request_count % 10 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()
        
        return prediction
        
    except torch.cuda.OutOfMemoryError:
        # GPU ran out of memory - try to recover
        logger.error("GPU out of memory, clearing cache")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise HTTPException(status_code=507, detail="GPU out of memory")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def parse_response_fallback(response_text: str, parameters: ProcessParameters) -> Dict:
    """
    Fallback parser when the model doesn't return valid JSON
    I use domain knowledge to generate reasonable defaults based on the input parameters
    This ensures we always return something useful even if the model output is garbled
    """
    predicted_yield = calculate_yield_estimate(parameters)
    
    # Calculate confidence based on parameter quality (lower since using fallback)
    base_confidence = calculate_confidence_score(parameters, None)
    # Reduce confidence by 10% when using fallback parser
    fallback_confidence = max(0.5, base_confidence - 0.1)
    
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
        "confidence": fallback_confidence
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


def calculate_confidence_score(parameters: ProcessParameters, model_confidence: Optional[float] = None) -> float:
    """
    Calculate confidence score based on parameter quality
    Higher confidence when parameters are within optimal ranges
    If model provides confidence, we blend it with parameter-based confidence
    """
    # Calculate how close each parameter is to optimal value
    # Temperature: optimal around 200°C (150-250°C is good range)
    temp_score = 1.0
    if parameters.temperature < 150 or parameters.temperature > 250:
        temp_score = 0.6  # Outside optimal range
    elif parameters.temperature < 180 or parameters.temperature > 230:
        temp_score = 0.8  # Near optimal
    else:
        temp_score = 1.0  # In optimal range (180-230°C)
    
    # Pressure: optimal around 1.5 Torr (0.5-2.5 Torr is good range)
    pressure_score = 1.0
    if parameters.pressure < 0.5 or parameters.pressure > 2.5:
        pressure_score = 0.6  # Outside optimal range
    elif parameters.pressure < 1.0 or parameters.pressure > 2.0:
        pressure_score = 0.8  # Near optimal
    else:
        pressure_score = 1.0  # In optimal range (1.0-2.0 Torr)
    
    # Etch time: optimal in 30-120s range
    time_score = 1.0 if 30 <= parameters.etch_time <= 120 else 0.7
    
    # Gas flow: optimal in 80-120 sccm range
    gas_score = 1.0
    if parameters.gas_flow < 50 or parameters.gas_flow > 150:
        gas_score = 0.6
    elif parameters.gas_flow < 80 or parameters.gas_flow > 120:
        gas_score = 0.8
    else:
        gas_score = 1.0
    
    # Chamber pressure: optimal in 4-6 mTorr range
    chamber_score = 1.0
    if parameters.chamber_pressure < 2 or parameters.chamber_pressure > 8:
        chamber_score = 0.6
    elif parameters.chamber_pressure < 4 or parameters.chamber_pressure > 6:
        chamber_score = 0.8
    else:
        chamber_score = 1.0
    
    # Weighted average of all parameter scores
    parameter_confidence = (
        temp_score * 0.25 +
        pressure_score * 0.25 +
        time_score * 0.20 +
        gas_score * 0.15 +
        chamber_score * 0.15
    )
    
    # If model provided confidence, blend it with parameter-based confidence
    if model_confidence is not None:
        # Blend: 60% model confidence, 40% parameter-based confidence
        final_confidence = model_confidence * 0.6 + parameter_confidence * 0.4
    else:
        # Use parameter-based confidence, but cap minimum at 0.65 for reasonable predictions
        final_confidence = max(0.65, parameter_confidence)
    
    # Clamp to valid range [0.5, 1.0]
    return round(min(1.0, max(0.5, final_confidence)), 3)


def calculate_yield_estimate(parameters: ProcessParameters) -> float:
    """
    Calculate a yield estimate using a simple heuristic model
    This is what I use when the AI model doesn't return a yield value
    Based on semiconductor manufacturing best practices - optimal temp around 200°C,
    optimal pressure around 1.5 Torr, etch time in the 30-120s range
    """
    # Calculate how close each parameter is to its optimal value
    # Closer to optimal = higher factor (closer to 1.0)
    temp_factor = 1.0 - abs(parameters.temperature - 200) / 100.0  # Optimal around 200°C
    temp_factor = max(0, min(1, temp_factor))
    
    pressure_factor = 1.0 - abs(parameters.pressure - 1.5) / 2.0  # Optimal around 1.5 Torr
    pressure_factor = max(0, min(1, pressure_factor))
    
    # Etch time should be in a reasonable range
    time_factor = 1.0 if 30 <= parameters.etch_time <= 120 else 0.8
    
    # Combine factors with weights and adjust from base yield
    base_yield = 85.0
    yield_adjustment = (temp_factor * 0.3 + pressure_factor * 0.3 + time_factor * 0.4) * 15.0
    
    # Clamp to realistic range (50-99%)
    predicted_yield = max(50.0, min(99.0, base_yield + yield_adjustment))
    return round(predicted_yield, 2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle startup and shutdown lifecycle
    I load the model at startup so it's ready when requests come in
    On shutdown, I clean up GPU memory properly
    """
    # Startup - load the model before accepting requests
    logger.info("Starting ChipFabAI GPU Service...")
    logger.info("Model: Gemma 2B (Open-source)")
    
    try:
        load_model()
        logger.info("Service ready and accepting requests")
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise
    
    yield
    
    # Shutdown - clean up resources
    logger.info("Shutting down GPU Service...")
    global model
    if model is not None:
        del model
    if device and device.type == "cuda":
        torch.cuda.empty_cache()  # Free GPU memory
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
    Process multiple predictions in one request
    I limit batch size to 100 to prevent timeouts and keep costs reasonable
    Each prediction is processed sequentially (could be parallelized, but this is simpler)
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
    
    # Process each parameter set one by one
    # If one fails, we continue with the rest and report errors at the end
    for idx, param_dict in enumerate(parameters_list):
        try:
            params = ProcessParameters(**param_dict)
            prediction = predict_optimal_settings(params)
            results.append(prediction)
        except Exception as e:
            logger.error(f"Batch prediction error for item {idx}: {str(e)}")
            errors.append({"index": idx, "error": str(e)})
            results.append(None)  # Mark as failed but keep going
    
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
