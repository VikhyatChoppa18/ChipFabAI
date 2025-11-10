# Project Size Optimization

## 📊 Size Reduction Summary

The project has been optimized from **8GB to 1.7MB** by removing unnecessary files.

## 🗑️ Removed Files

### Large Directories (Not in Git)
- `venv/` - 5.8GB (Python virtual environment)
- `frontend/node_modules/` - 490MB (Node.js dependencies)
- `__pycache__/` - Python cache files
- `*.pyc`, `*.pyo` - Compiled Python files

### Large Files (Not in Git)
- Model files (`.pth`, `.pt`, `.bin`, `.safetensors`) - Downloaded at runtime
- Log files (`.log`) - Generated at runtime
- Build artifacts - Generated during build
- Temporary files - Generated during execution

## ✅ Files in Repository

The repository now contains only:
- Source code (`.py`, `.js`, `.jsx`)
- Configuration files (`.txt`, `.yaml`, `.json`)
- Documentation (`.md`)
- Dockerfiles
- Deployment scripts
- Sample data file (`sample_data.csv`)

## 📦 Dependency Management

### Python Dependencies
Dependencies are defined in `requirements.txt` files:
- `api-gateway/requirements.txt`
- `gpu-service/requirements.txt`
- `data-processor/requirements.txt`

**To recreate venv:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r api-gateway/requirements.txt
pip install -r gpu-service/requirements.txt
pip install -r data-processor/requirements.txt
```

### Node.js Dependencies
Dependencies are defined in `frontend/package.json`.

**To recreate node_modules:**
```bash
cd frontend
npm install
```

## 🚀 Deployment

### Cloud Run Deployment
Cloud Run will automatically install dependencies during build:
- Python dependencies from `requirements.txt`
- Node.js dependencies from `package.json`
- Models downloaded at runtime from HuggingFace

### Local Development
See [SETUP.md](SETUP.md) for local development setup instructions.

## 📋 .gitignore Configuration

The `.gitignore` file ensures large files are not committed:
- `venv/` - Python virtual environment
- `node_modules/` - Node.js dependencies
- `*.pth`, `*.pt`, `*.bin` - Model files
- `*.log` - Log files
- `__pycache__/` - Python cache
- `build/`, `dist/` - Build artifacts

## 🔍 Verifying Repository Size

```bash
# Check repository size
du -sh .

# Check what's tracked in git
git ls-files | wc -l

# Check largest files in repository
git ls-files | xargs du -h | sort -rh | head -20
```

## ✅ Best Practices

1. **Never commit:**
   - `venv/` directory
   - `node_modules/` directory
   - Model files (`.pth`, `.pt`, `.bin`)
   - Log files (`.log`)
   - Cache files (`__pycache__/`, `.pytest_cache/`)

2. **Always commit:**
   - Source code (`.py`, `.js`, `.jsx`)
   - Configuration files (`.txt`, `.yaml`, `.json`)
   - Documentation (`.md`)
   - Dockerfiles
   - Deployment scripts

3. **Use .gitignore:**
   - Keep `.gitignore` updated
   - Review before committing
   - Use `git status` to verify

## 🎯 Size Targets

- **Repository size:** < 10MB (currently 1.7MB ✅)
- **Deployment size:** < 2GB (models downloaded at runtime)
- **Build time:** < 10 minutes (with dependency caching)

## 📚 Related Documentation

- [SETUP.md](SETUP.md) - Setup instructions
- [README.md](README.md) - Project overview
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide

---

**Last Updated:** 2025-01-08  
**Repository Size:** 1.7MB  
**Status:** ✅ Optimized

