#!/bin/bash

# Cleanup script to remove large files and reduce project size
# This script removes files that should not be in the repository

set -e

echo "🧹 Cleaning up large files from project..."
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Remove venv if it exists (should be recreated with requirements.txt)
if [ -d "venv" ]; then
    echo -e "${YELLOW}Removing venv directory (5.8GB)...${NC}"
    rm -rf venv
    echo -e "${GREEN}✅ Removed venv${NC}"
fi

# Remove node_modules if it exists (should be recreated with npm install)
if [ -d "frontend/node_modules" ]; then
    echo -e "${YELLOW}Removing frontend/node_modules directory (490MB)...${NC}"
    rm -rf frontend/node_modules
    echo -e "${GREEN}✅ Removed node_modules${NC}"
fi

# Remove Python cache files
echo -e "${YELLOW}Removing Python cache files...${NC}"
find . -type d -name "__pycache__" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -not -path "./.git/*" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -not -path "./.git/*" -delete 2>/dev/null || true
echo -e "${GREEN}✅ Removed Python cache files${NC}"

# Remove log files
echo -e "${YELLOW}Removing log files...${NC}"
find . -type f -name "*.log" -not -path "./.git/*" -not -path "./venv/*" -not -path "./frontend/node_modules/*" -delete 2>/dev/null || true
echo -e "${GREEN}✅ Removed log files${NC}"

# Remove PID files
echo -e "${YELLOW}Removing PID files...${NC}"
find . -type f -name "*.pid" -not -path "./.git/*" -delete 2>/dev/null || true
echo -e "${GREEN}✅ Removed PID files${NC}"

# Remove large model files (should be downloaded at runtime)
echo -e "${YELLOW}Removing cached model files...${NC}"
find . -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" -o -name "*.onnx" \) -not -path "./.git/*" -not -path "./venv/*" -not -path "./frontend/node_modules/*" -delete 2>/dev/null || true
echo -e "${GREEN}✅ Removed model files${NC}"

# Remove temporary files
echo -e "${YELLOW}Removing temporary files...${NC}"
find . -type f \( -name "*.tmp" -o -name "*.temp" -o -name "*.bak" -o -name "*.backup" -o -name "*.swp" -o -name "*.swo" \) -not -path "./.git/*" -delete 2>/dev/null || true
echo -e "${GREEN}✅ Removed temporary files${NC}"

# Remove OS files
echo -e "${YELLOW}Removing OS files...${NC}"
find . -type f -name ".DS_Store" -not -path "./.git/*" -delete 2>/dev/null || true
find . -type f -name "Thumbs.db" -not -path "./.git/*" -delete 2>/dev/null || true
echo -e "${GREEN}✅ Removed OS files${NC}"

# Remove test coverage files
echo -e "${YELLOW}Removing test coverage files...${NC}"
find . -type d -name ".pytest_cache" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".coverage" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "htmlcov" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name ".coverage" -not -path "./.git/*" -delete 2>/dev/null || true
echo -e "${GREEN}✅ Removed test coverage files${NC}"

# Remove build directories
echo -e "${YELLOW}Removing build directories...${NC}"
find . -type d -name "build" -not -path "./.git/*" -not -path "./frontend/node_modules/*" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "dist" -not -path "./.git/*" -not -path "./frontend/node_modules/*" -exec rm -rf {} + 2>/dev/null || true
echo -e "${GREEN}✅ Removed build directories${NC}"

# Calculate new size
echo ""
echo -e "${GREEN}📊 Calculating new project size...${NC}"
NEW_SIZE=$(du -sh . 2>/dev/null | cut -f1)
echo -e "${GREEN}✅ New project size: ${NEW_SIZE}${NC}"

echo ""
echo -e "${GREEN}✅ Cleanup complete!${NC}"
echo ""
echo "📋 Next steps:"
echo "  1. Review .gitignore to ensure all large files are ignored"
echo "  2. Commit changes: git add .gitignore .gcloudignore"
echo "  3. Verify files are not tracked: git status"
echo "  4. For deployment, recreate venv and node_modules as needed"
echo ""

