#!/usr/bin/env python3
"""
Comprehensive Test Suite for ChipFabAI
Tests all services and functionalities
"""

import requests
import json
import time
import sys
from typing import Dict, List

# Configuration
GPU_SERVICE_URL = "http://localhost:8080"
API_GATEWAY_URL = "http://localhost:8081"
FRONTEND_URL = "http://localhost:3000"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_header(text: str):
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}\n")

def print_test(test_name: str):
    print(f"{Colors.YELLOW} {test_name}...{Colors.RESET}")

def print_success(message: str):
    print(f"{Colors.GREEN} {message}{Colors.RESET}")

def print_error(message: str):
    print(f"{Colors.RED} {message}{Colors.RESET}")

def print_info(message: str):
    print(f"   {message}")

def test_service_health(service_name: str, url: str, timeout: int = 5) -> bool:
    """Test if a service is running and healthy"""
    print_test(f"Testing {service_name} Health")
    
    try:
        response = requests.get(f"{url}/health", timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            print_success(f"{service_name} is healthy")
            print_info(f"Status: {data.get('status', 'unknown')}")
            return True
        else:
            print_error(f"{service_name} returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error(f"{service_name} is not running at {url}")
        print_info("Start it with appropriate command")
        return False
    except Exception as e:
        print_error(f"{service_name} error: {str(e)}")
        return False

def test_gpu_service():
    """Test GPU Service functionality"""
    print_header("Testing GPU Service")
    
    results = {
        "health": False,
        "root": False,
        "predict": False,
        "batch_predict": False
    }
    
    # Test health endpoint
    results["health"] = test_service_health("GPU Service", GPU_SERVICE_URL)
    
    # Test root endpoint
    print_test("Testing GPU Service Root Endpoint")
    try:
        response = requests.get(f"{GPU_SERVICE_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success("Root endpoint working")
            print_info(f"Service: {data.get('service', 'unknown')}")
            print_info(f"Model loaded: {data.get('model_loaded', False)}")
            print_info(f"Device: {data.get('device', 'unknown')}")
            results["root"] = True
        else:
            print_error(f"Root endpoint returned {response.status_code}")
    except Exception as e:
        print_error(f"Root endpoint error: {str(e)}")
    
    # Test prediction endpoint
    print_test("Testing GPU Service Prediction")
    test_data = {
        "temperature": 200.0,
        "pressure": 1.5,
        "etch_time": 60.0,
        "gas_flow": 100.0,
        "chamber_pressure": 5.0,
        "wafer_size": 300,
        "process_type": "etching"
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{GPU_SERVICE_URL}/predict",
            json=test_data,
            timeout=60
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print_success("Prediction successful")
            print_info(f"Predicted Yield: {result.get('predicted_yield')}%")
            print_info(f"Confidence: {result.get('confidence', 0) * 100:.1f}%")
            print_info(f"Processing Time: {result.get('processing_time_ms', 0):.0f}ms")
            print_info(f"Total Time: {elapsed:.2f}s")
            results["predict"] = True
        else:
            print_error(f"Prediction failed: {response.status_code}")
            print_info(f"Response: {response.text[:200]}")
    except Exception as e:
        print_error(f"Prediction error: {str(e)}")
    
    # Test batch prediction
    print_test("Testing GPU Service Batch Prediction")
    batch_data = [test_data, {**test_data, "temperature": 205.0}]
    
    try:
        response = requests.post(
            f"{GPU_SERVICE_URL}/batch-predict",
            json={"parameters_list": batch_data},
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print_success(f"Batch prediction successful ({result.get('count', 0)} predictions)")
            results["batch_predict"] = True
        else:
            print_error(f"Batch prediction failed: {response.status_code}")
    except Exception as e:
        print_error(f"Batch prediction error: {str(e)}")
    
    return results

def test_api_gateway():
    """Test API Gateway functionality"""
    print_header("Testing API Gateway")
    
    results = {
        "health": False,
        "root": False,
        "predict": False,
        "optimize": False,
        "sample_data": False,
        "batch_predict": False
    }
    
    # Test health endpoint
    results["health"] = test_service_health("API Gateway", API_GATEWAY_URL)
    
    # Test root endpoint
    print_test("Testing API Gateway Root Endpoint")
    try:
        response = requests.get(f"{API_GATEWAY_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success("Root endpoint working")
            print_info(f"Service: {data.get('service', 'unknown')}")
            results["root"] = True
        else:
            print_error(f"Root endpoint returned {response.status_code}")
    except Exception as e:
        print_error(f"Root endpoint error: {str(e)}")
    
    # Test sample data endpoint
    print_test("Testing API Gateway Sample Data")
    try:
        response = requests.get(f"{API_GATEWAY_URL}/api/v1/sample-data", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success("Sample data endpoint working")
            print_info(f"Sample parameters retrieved")
            results["sample_data"] = True
        else:
            print_error(f"Sample data failed: {response.status_code}")
    except Exception as e:
        print_error(f"Sample data error: {str(e)}")
    
    # Test prediction through API Gateway
    print_test("Testing API Gateway Prediction")
    test_data = {
        "temperature": 200.0,
        "pressure": 1.5,
        "etch_time": 60.0,
        "gas_flow": 100.0,
        "chamber_pressure": 5.0,
        "wafer_size": 300,
        "process_type": "etching"
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_GATEWAY_URL}/api/v1/predict",
            json=test_data,
            timeout=60
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print_success("Prediction through API Gateway successful")
            print_info(f"Predicted Yield: {result.get('predicted_yield')}%")
            print_info(f"Confidence: {result.get('confidence', 0) * 100:.1f}%")
            print_info(f"Processing Time: {result.get('processing_time_ms', 0):.0f}ms")
            print_info(f"Total Time: {elapsed:.2f}s")
            results["predict"] = True
        else:
            print_error(f"Prediction failed: {response.status_code}")
            print_info(f"Response: {response.text[:200]}")
    except Exception as e:
        print_error(f"Prediction error: {str(e)}")
    
    # Test optimize endpoint
    print_test("Testing API Gateway Optimize")
    try:
        response = requests.post(
            f"{API_GATEWAY_URL}/api/v1/optimize",
            json=test_data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print_success("Optimize endpoint working")
            print_info(f"Optimization Score: {result.get('optimization_score', 0)}")
            print_info(f"Cost Savings: ${result.get('cost_savings_estimate', {}).get('savings_per_wafer_usd', 0):.2f} per wafer")
            results["optimize"] = True
        else:
            print_error(f"Optimize failed: {response.status_code}")
    except Exception as e:
        print_error(f"Optimize error: {str(e)}")
    
    # Test batch prediction
    print_test("Testing API Gateway Batch Prediction")
    batch_data = [test_data, {**test_data, "temperature": 205.0}]
    
    try:
        response = requests.post(
            f"{API_GATEWAY_URL}/api/v1/batch-predict",
            json={"parameters_list": batch_data},
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print_success(f"Batch prediction successful ({result.get('count', 0)} predictions)")
            results["batch_predict"] = True
        else:
            print_error(f"Batch prediction failed: {response.status_code}")
    except Exception as e:
        print_error(f"Batch prediction error: {str(e)}")
    
    return results

def test_data_processor():
    """Test Data Processor functionality"""
    print_header("Testing Data Processor")
    
    results = {
        "import": False,
        "functions": False
    }
    
    # Test if data processor can be imported
    print_test("Testing Data Processor Import")
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data-processor'))
        from main import process_historical_data, clean_data, engineer_features, calculate_statistics, generate_insights
        print_success("Data processor imports successfully")
        print_info("All functions available: process_historical_data, clean_data, engineer_features, calculate_statistics, generate_insights")
        results["import"] = True
        results["functions"] = True
    except Exception as e:
        print_error(f"Data processor import error: {str(e)}")
    
    return results

def test_frontend():
    """Test Frontend availability"""
    print_header("Testing Frontend")
    
    results = {
        "available": False
    }
    
    print_test("Testing Frontend Availability")
    try:
        response = requests.get(FRONTEND_URL, timeout=5)
        if response.status_code == 200:
            print_success("Frontend is accessible")
            results["available"] = True
        else:
            print_info(f"Frontend returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print_info("Frontend is not running (expected if not started)")
        print_info("Start it with: cd frontend && npm start")
    except Exception as e:
        print_info(f"Frontend check: {str(e)}")
    
    return results

def test_integration():
    """Test integration between services"""
    print_header("Testing Service Integration")
    
    results = {
        "gpu_to_api": False
    }
    
    # Test if API Gateway can connect to GPU Service
    print_test("Testing API Gateway -> GPU Service Integration")
    try:
        # Check API Gateway health (which checks GPU service)
        response = requests.get(f"{API_GATEWAY_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            gpu_service_health = data.get('gpu_service', {})
            if gpu_service_health.get('status') == 'healthy':
                print_success("API Gateway can communicate with GPU Service")
                results["gpu_to_api"] = True
            else:
                print_error(f"GPU Service health check failed: {gpu_service_health}")
        else:
            print_error(f"API Gateway health check failed: {response.status_code}")
    except Exception as e:
        print_error(f"Integration test error: {str(e)}")
    
    return results

def print_summary(all_results: Dict):
    """Print test summary"""
    print_header("Test Summary")
    
    total_tests = 0
    passed_tests = 0
    
    for service, results in all_results.items():
        print(f"\n{Colors.BLUE}{service.upper()}:{Colors.RESET}")
        for test, passed in results.items():
            total_tests += 1
            if passed:
                passed_tests += 1
                print(f"  {Colors.GREEN} {test}{Colors.RESET}")
            else:
                print(f"  {Colors.RED} {test}{Colors.RESET}")
    
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"Total Tests: {total_tests}")
    print(f"{Colors.GREEN}Passed: {passed_tests}{Colors.RESET}")
    print(f"{Colors.RED}Failed: {total_tests - passed_tests}{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}\n")
    
    if passed_tests == total_tests:
        print(f"{Colors.GREEN} All tests passed!{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.YELLOW}WARNING: Some tests failed. Check the output above.{Colors.RESET}")
        return 1

def main():
    """Main test function"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}ChipFabAI Comprehensive Test Suite{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}\n")
    
    # Check which services are running
    print(f"{Colors.YELLOW}Checking which services are running...{Colors.RESET}\n")
    
    all_results = {}
    
    # Test GPU Service
    try:
        gpu_results = test_gpu_service()
        all_results["GPU Service"] = gpu_results
    except Exception as e:
        print_error(f"GPU Service test failed: {str(e)}")
        all_results["GPU Service"] = {}
    
    # Test API Gateway
    try:
        api_results = test_api_gateway()
        all_results["API Gateway"] = api_results
    except Exception as e:
        print_error(f"API Gateway test failed: {str(e)}")
        all_results["API Gateway"] = {}
    
    # Test Data Processor
    try:
        data_results = test_data_processor()
        all_results["Data Processor"] = data_results
    except Exception as e:
        print_error(f"Data Processor test failed: {str(e)}")
        all_results["Data Processor"] = {}
    
    # Test Frontend
    try:
        frontend_results = test_frontend()
        all_results["Frontend"] = frontend_results
    except Exception as e:
        print_error(f"Frontend test failed: {str(e)}")
        all_results["Frontend"] = {}
    
    # Test Integration
    try:
        integration_results = test_integration()
        all_results["Integration"] = integration_results
    except Exception as e:
        print_error(f"Integration test failed: {str(e)}")
        all_results["Integration"] = {}
    
    # Print summary
    exit_code = print_summary(all_results)
    
    # Print instructions
    print(f"\n{Colors.YELLOW}To start services:{Colors.RESET}")
    print("  1. GPU Service: cd gpu-service && source ../venv/bin/activate && python main.py")
    print("  2. API Gateway: cd api-gateway && source ../venv/bin/activate && export GPU_SERVICE_URL=http://localhost:8080 && python main.py")
    print("  3. Frontend: cd frontend && npm install && export REACT_APP_API_URL=http://localhost:8081 && npm start")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()

