# ChipFabAI Architecture


```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
│                    Cloud Run Service (Port 80)                   │
│              - Material-UI Dashboard                             │
│              - Anime.js Animations                                │
│              - Real-time Charts (Recharts)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ HTTPS/REST API
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    API Gateway (FastAPI)                         │
│                   Cloud Run Service (Port 8080)                  │
│              - Request Orchestration                             │
│              - Data Processing                                   │
│              - Caching & Rate Limiting                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
┌───────────────────────────┐  ┌────────────────────────────┐
│   GPU Service (FastAPI)   │  │  Data Processor (Job)      │
│  Cloud Run Service (GPU)   │  │  Cloud Run Job             │
│  - NVIDIA L4 GPU           │  │  - Batch Processing        │
│  - Gemma 2B Model          │  │  - Historical Data         │
│  - Model Inference         │  │  - Feature Engineering     │
│  - Quantized Inference     │  └────────────────────────────┘
└───────────────────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Cloud Storage      │
                  │  - Process Data      │
                  │  - Model Cache       │
                  │  - Historical Data   │
                  └──────────────────────┘
```


- **Technology**: React 18, Material-UI, Anime.js
- **Features**:
  - Interactive parameter input forms
  - Real-time prediction visualization
  - Historical data charts
  - Animated UI elements
- **Deployment**: Cloud Run Service
- **Resources**: 512Mi RAM, 1 CPU

- **Technology**: FastAPI (Python)
- **Features**:
  - Request routing and orchestration
  - Error handling and retries
  - Response caching
  - Cost savings calculation
- **Deployment**: Cloud Run Service
- **Resources**: 2Gi RAM, 2 CPU

- **Technology**: FastAPI, PyTorch, Transformers, Gemma 2B
- **Features**:
  - Model inference on NVIDIA L4 GPU
  - Quantized model for efficiency
  - Batch prediction support
  - Real-time optimization
- **Deployment**: Cloud Run Service (GPU-enabled)
- **Resources**: 16Gi RAM, 4 CPU, 1x NVIDIA L4 GPU
- **Region**: europe-west4 (GPU availability)

- **Technology**: Python, Pandas, NumPy
- **Features**:
  - Historical data processing
  - Feature engineering
  - Statistical analysis
  - Insight generation
- **Deployment**: Cloud Run Job
- **Resources**: 4Gi RAM, 2 CPU


1. **User Input** → Frontend receives process parameters
2. **API Request** → Frontend sends POST to API Gateway
3. **Orchestration** → API Gateway forwards to GPU Service
4. **Model Inference** → GPU Service runs Gemma model
5. **Response** → Results flow back through API Gateway to Frontend
6. **Visualization** → Frontend displays predictions with animations


- **Model**: Gemma 2B (quantized, float16)
- **GPU**: NVIDIA L4 (24GB VRAM)
- **Batch Processing**: Supports up to 100 concurrent predictions
- **Latency**: <500ms for single prediction
- **Throughput**: ~20 predictions/second


- **Auto-scaling**: Cloud Run automatically scales based on demand
- **Min Instances**: 0 (cost-effective)
- **Max Instances**: 10 (configurable)
- **Concurrency**: 80 requests per instance


- **HTTPS**: All services use HTTPS
- **CORS**: Configured for cross-origin requests
- **Authentication**: Optional (can be added for production)
- **Rate Limiting**: Implemented in API Gateway


- **GPU**: Only used when processing requests
- **Auto-scaling**: Scales to zero when idle
- **Quantized Model**: Reduces memory and inference time
- **Caching**: Reduces redundant GPU computations


- **Logging**: All services log to Cloud Logging
- **Metrics**: Cloud Run provides built-in metrics
- **Tracing**: Distributed tracing support
- **Alerts**: Configurable alerting for errors


This architecture directly supports:
- **CHIPS Act**: Optimizes semiconductor manufacturing
- **White House AI Plan**: Demonstrates AI in critical infrastructure
- **National Security**: Strengthens domestic chip production
- **Economic Competitiveness**: Reduces manufacturing costs

