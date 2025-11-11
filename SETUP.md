# ChipFabAI Setup Guide


This guide explains how to set up the ChipFabAI project after cloning from GitHub.


- Python 3.11+
- Node.js 18+
- Google Cloud SDK (for deployment)
- Docker (for containerization)



```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Python dependencies
pip install -r api-gateway/requirements.txt
pip install -r gpu-service/requirements.txt
pip install -r data-processor/requirements.txt
```


```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Return to project root
cd ..
```


```bash
# Check Python packages
pip list | grep -E "(fastapi|torch|transformers)"

# Check Node.js packages
cd frontend && npm list --depth=0 && cd ..
```



Create a `.env` file in the project root (optional):

```bash
# GPU Service
MODEL_NAME=microsoft/DialoGPT-small
MODEL_CACHE_DIR=/tmp/models
PORT=8080

# API Gateway
GPU_SERVICE_URL=http://localhost:8080
PORT=8081
ENABLE_CACHING=true
CACHE_TTL=300

# Frontend
REACT_APP_API_URL=http://localhost:8081
```



```bash
# Start all services
./start_services_demo.sh

# Stop all services
./stop_services_demo.sh
```


**Terminal 1 - GPU Service:**
```bash
cd gpu-service
source ../venv/bin/activate
python main.py
```

**Terminal 2 - API Gateway:**
```bash
cd api-gateway
source ../venv/bin/activate
export GPU_SERVICE_URL=http://localhost:8080
python main.py
```

**Terminal 3 - Frontend:**
```bash
cd frontend
export REACT_APP_API_URL=http://localhost:8081
npm start
```



```bash
# Run production tests
python test_production.py

# Run performance tests
./test_all_performance.sh

# Run all services tests
python test_all_services.py
```



```bash
# Set project ID
export PROJECT_ID=your-project-id
export REGION=europe-west4

# Deploy all services
./deploy-demo.sh
```


```bash
# Build and run with Docker Compose
docker-compose up --build
```



```bash
# Clear pip cache
pip cache purge

# Reinstall dependencies
pip install --force-reinstall -r api-gateway/requirements.txt
```


```bash
# Clear npm cache
npm cache clean --force

# Remove node_modules and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
```


```bash
# Check if ports are in use
lsof -i :8080  # GPU Service
lsof -i :8081  # API Gateway
lsof -i :3000  # Frontend

# Kill processes if needed
kill -9 <PID>
```


```bash
# Set HuggingFace cache directory
export HF_HOME=/tmp/huggingface
export TRANSFORMERS_CACHE=/tmp/transformers_cache

# Or use model cache directory
export MODEL_CACHE_DIR=/tmp/models
```


```
Mgp_Sys/
├── api-gateway/          # API Gateway service
│   ├── main.py
│   ├── cache.py
│   ├── load_balancer.py
│   ├── requirements.txt
│   └── Dockerfile
├── gpu-service/          # GPU inference service
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── data-processor/       # Data processing service
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/             # React frontend
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── Dockerfile
├── tests/                # Test files
├── docs/                 # Documentation
└── scripts/              # Deployment scripts
```


1.  Set up virtual environment
2.  Install dependencies
3.  Configure environment variables
4.  Start services
5.  Run tests
6.  Deploy to Cloud Run


- [README.md](README.md) - Project overview
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment 
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Testing 

---

**Note**: The `venv` and `node_modules` directories are not included in the repository. They should be created locally using the steps above.

