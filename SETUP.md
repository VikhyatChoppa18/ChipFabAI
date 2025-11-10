# ChipFabAI Setup Guide

## 🚀 Quick Setup

This guide explains how to set up the ChipFabAI project after cloning from GitHub.

## 📋 Prerequisites

- Python 3.11+
- Node.js 18+
- Google Cloud SDK (for deployment)
- Docker (for containerization)

## 🔧 Local Development Setup

### 1. Create Python Virtual Environment

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

### 2. Install Frontend Dependencies

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Return to project root
cd ..
```

### 3. Verify Installation

```bash
# Check Python packages
pip list | grep -E "(fastapi|torch|transformers)"

# Check Node.js packages
cd frontend && npm list --depth=0 && cd ..
```

## 🎯 Configuration

### Environment Variables

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

## 🚀 Running Services

### Option 1: Automated Start (Recommended)

```bash
# Start all services
./start_services_demo.sh

# Stop all services
./stop_services_demo.sh
```

### Option 2: Manual Start

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

## 🧪 Testing

### Run Tests

```bash
# Run production tests
python test_production.py

# Run performance tests
./test_all_performance.sh

# Run all services tests
python test_all_services.py
```

## 📦 Deployment

### Google Cloud Run Deployment

```bash
# Set project ID
export PROJECT_ID=your-project-id
export REGION=europe-west4

# Deploy all services
./deploy-demo.sh
```

### Local Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build
```

## 🔍 Troubleshooting

### Python Dependencies Issues

```bash
# Clear pip cache
pip cache purge

# Reinstall dependencies
pip install --force-reinstall -r api-gateway/requirements.txt
```

### Node.js Dependencies Issues

```bash
# Clear npm cache
npm cache clean --force

# Remove node_modules and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Port Conflicts

```bash
# Check if ports are in use
lsof -i :8080  # GPU Service
lsof -i :8081  # API Gateway
lsof -i :3000  # Frontend

# Kill processes if needed
kill -9 <PID>
```

### Model Download Issues

```bash
# Set HuggingFace cache directory
export HF_HOME=/tmp/huggingface
export TRANSFORMERS_CACHE=/tmp/transformers_cache

# Or use model cache directory
export MODEL_CACHE_DIR=/tmp/models
```

## 📊 Project Structure

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

## 🎯 Next Steps

1. ✅ Set up virtual environment
2. ✅ Install dependencies
3. ✅ Configure environment variables
4. ✅ Start services
5. ✅ Run tests
6. ✅ Deploy to Cloud Run

## 📚 Additional Resources

- [README.md](README.md) - Project overview
- [PERFORMANCE_FEATURES.md](PERFORMANCE_FEATURES.md) - Performance features
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Testing guide

---

**Note**: The `venv` and `node_modules` directories are not included in the repository. They should be created locally using the steps above.

