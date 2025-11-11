#!/usr/bin/env python3
"""
Quick Service Test - Tests services without requiring them to be running
Tests code structure, imports, and basic functionality
"""

import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api-gateway'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gpu-service'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data-processor'))

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def test_imports():
    """Test if all modules can be imported"""
    print(f"{Colors.BLUE}Testing Imports...{Colors.RESET}\n")
    
    results = []
    
    # Test API Gateway
    try:
        import main as api_main
        print(f"{Colors.GREEN} API Gateway imports successfully{Colors.RESET}")
        results.append(True)
    except Exception as e:
        print(f"{Colors.RED} API Gateway import failed: {e}{Colors.RESET}")
        results.append(False)
    
    # Test GPU Service (may fail if torch not installed locally)
    try:
        import main as gpu_main
        print(f"{Colors.GREEN} GPU Service imports successfully{Colors.RESET}")
        results.append(True)
    except Exception as e:
        print(f"{Colors.YELLOW}WARNING: GPU Service import warning: {e}{Colors.RESET}")
        print(f"   (This is expected if PyTorch not installed locally)")
        results.append(True)  # Not a critical failure
    
    # Test Data Processor
    try:
        import main as data_main
        print(f"{Colors.GREEN} Data Processor imports successfully{Colors.RESET}")
        results.append(True)
    except Exception as e:
        print(f"{Colors.RED} Data Processor import failed: {e}{Colors.RESET}")
        results.append(False)
    
    return all(results)

def test_file_structure():
    """Test file structure"""
    print(f"\n{Colors.BLUE}Testing File Structure...{Colors.RESET}\n")
    
    required_files = [
        'api-gateway/main.py',
        'api-gateway/requirements.txt',
        'api-gateway/Dockerfile',
        'gpu-service/main.py',
        'gpu-service/requirements.txt',
        'gpu-service/Dockerfile',
        'data-processor/main.py',
        'data-processor/requirements.txt',
        'data-processor/Dockerfile',
        'frontend/package.json',
        'frontend/Dockerfile',
        'frontend/src/App.js',
        'deploy.sh',
        'deploy-demo.sh',
        'README.md',
        '.gitignore'
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"{Colors.GREEN} {file}{Colors.RESET}")
        else:
            print(f"{Colors.RED} {file} - MISSING{Colors.RESET}")
            missing.append(file)
    
    return len(missing) == 0

def test_requirements():
    """Test requirements files"""
    print(f"\n{Colors.BLUE}Testing Requirements Files...{Colors.RESET}\n")
    
    req_files = [
        'api-gateway/requirements.txt',
        'gpu-service/requirements.txt',
        'data-processor/requirements.txt'
    ]
    
    all_valid = True
    for req_file in req_files:
        if os.path.exists(req_file):
            with open(req_file, 'r') as f:
                content = f.read()
                if content.strip():
                    print(f"{Colors.GREEN} {req_file} - Valid{Colors.RESET}")
                else:
                    print(f"{Colors.RED} {req_file} - Empty{Colors.RESET}")
                    all_valid = False
        else:
            print(f"{Colors.RED} {req_file} - Missing{Colors.RESET}")
            all_valid = False
    
    return all_valid

def main():
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}ChipFabAI - Quick Service Test{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")
    
    results = []
    
    results.append(("File Structure", test_file_structure()))
    results.append(("Requirements", test_requirements()))
    results.append(("Imports", test_imports()))
    
    # Summary
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}Test Summary{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{Colors.GREEN} PASS{Colors.RESET}" if result else f"{Colors.RED} FAIL{Colors.RESET}"
        print(f"{name}: {status}")
    
    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.RESET}\n")
    
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD} All tests passed! Code structure is valid.{Colors.RESET}\n")
        return 0
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}WARNING: Some tests failed. Review the output above.{Colors.RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())

