#!/usr/bin/env python3
"""
Test Running Services - Actual Feature Testing
Tests all endpoints and features with real HTTP requests
"""

import requests
import json
import time
import sys
import os
from datetime import datetime

# Configuration - Auto-detect GPU service port
def detect_gpu_port():
    """Detect GPU service port (try 8082 first, then 8080)"""
    for port in [8082, 8080]:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Check if it's our service (has model_loaded or status)
                    if 'model_loaded' in data or (data.get('status') == 'healthy' and 'gpu' in data):
                        return port
                except:
                    pass
        except:
            pass
    return 8082  # Default to 8082 to avoid nginx conflict

GPU_PORT = detect_gpu_port()
GPU_SERVICE_URL = os.getenv("GPU_SERVICE_URL", f"http://localhost:{GPU_PORT}")
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://localhost:8081")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*70}{Colors.RESET}\n")

def print_test(name):
    print(f"{Colors.YELLOW} {name}...{Colors.RESET}")

def print_success(msg):
    print(f"{Colors.GREEN} {msg}{Colors.RESET}")

def print_error(msg):
    print(f"{Colors.RED} {msg}{Colors.RESET}")

def print_info(msg):
    print(f"   {msg}")

def test_service_availability():
    """Test if services are running"""
    print_header("Service Availability Check")
    
    results = {}
    
    # Test GPU Service
    print_test("GPU Service Availability")
    try:
        # Try health endpoint first
        response = requests.get(f"{GPU_SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            try:
                data = response.json()
                print_success("GPU Service is running")
                print_info(f"Status: {data.get('status')}")
                print_info(f"Model loaded: {data.get('model_loaded')}")
                results['gpu'] = True
            except:
                # If not JSON, check if it's HTML (nginx or other service)
                if 'html' in response.text.lower():
                    print_error("Port 8080 is running nginx/HTML, not GPU Service")
                    print_info("Kill the process: lsof -ti:8080 | xargs kill -9")
                    results['gpu'] = False
                else:
                    print_success("GPU Service is running (non-JSON response)")
                    results['gpu'] = True
        else:
            print_error(f"GPU Service returned {response.status_code}")
            results['gpu'] = False
    except requests.exceptions.ConnectionError:
        print_error("GPU Service is not running on port 8080")
        print_info("Start it with: cd gpu-service && source ../venv/bin/activate && python main.py")
        results['gpu'] = False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        results['gpu'] = False
    
    # Test API Gateway
    print_test("API Gateway Availability")
    try:
        response = requests.get(f"{API_GATEWAY_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success("API Gateway is running")
            print_info(f"Status: {data.get('status')}")
            results['api'] = True
        else:
            print_error(f"API Gateway returned {response.status_code}")
            results['api'] = False
    except requests.exceptions.ConnectionError:
        print_error("API Gateway is not running on port 8081")
        print_info("Start it with: cd api-gateway && source ../venv/bin/activate && export GPU_SERVICE_URL=http://localhost:8080 && python main.py")
        results['api'] = False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        results['api'] = False
    
    return results

def test_gpu_service_features():
    """Test GPU Service features"""
    print_header("GPU Service Features")
    
    results = {}
    
    # Test Health Endpoint
    print_test("Health Endpoint")
    try:
        response = requests.get(f"{GPU_SERVICE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print_success("Health check passed")
            print_info(f"Status: {data.get('status')}")
            print_info(f"Model loaded: {data.get('model_loaded')}")
            if data.get('gpu'):
                print_info(f"GPU: {data['gpu'].get('device_name', 'N/A')}")
            results['health'] = True
        else:
            print_error(f"Health check failed: {response.status_code}")
            results['health'] = False
    except Exception as e:
        print_error(f"Health check error: {str(e)}")
        results['health'] = False
    
    # Test Prediction Endpoint
    print_test("Prediction Endpoint")
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
            timeout=60,
            headers={"Content-Type": "application/json"}
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print_success("Prediction successful")
            print_info(f"Predicted Yield: {result.get('predicted_yield')}%")
            print_info(f"Confidence: {result.get('confidence', 0) * 100:.1f}%")
            print_info(f"Processing Time: {result.get('processing_time_ms', 0):.0f}ms")
            print_info(f"Total Time: {elapsed:.2f}s")
            
            # Validate response structure
            required = ['predicted_yield', 'optimal_temperature', 'optimal_pressure', 
                       'risk_factors', 'recommendations', 'confidence']
            missing = [f for f in required if f not in result]
            if missing:
                print_error(f"Missing fields: {missing}")
            else:
                print_success("Response structure valid")
            
            results['predict'] = True
        else:
            print_error(f"Prediction failed: {response.status_code}")
            print_info(f"Response: {response.text[:200]}")
            results['predict'] = False
    except Exception as e:
        print_error(f"Prediction error: {str(e)}")
        results['predict'] = False
    
    # Test Batch Prediction
    print_test("Batch Prediction Endpoint")
    batch_data = {
        "parameters_list": [
            test_data,
            {**test_data, "temperature": 205.0}
        ]
    }
    
    try:
        response = requests.post(
            f"{GPU_SERVICE_URL}/batch-predict",
            json=batch_data,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print_success(f"Batch prediction successful ({result.get('count', 0)} predictions)")
            results['batch'] = True
        else:
            print_error(f"Batch prediction failed: {response.status_code}")
            results['batch'] = False
    except Exception as e:
        print_error(f"Batch prediction error: {str(e)}")
        results['batch'] = False
    
    return results

def test_api_gateway_features():
    """Test API Gateway features"""
    print_header("API Gateway Features")
    
    results = {}
    
    # Test Health Endpoint
    print_test("Health Endpoint")
    try:
        response = requests.get(f"{API_GATEWAY_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print_success("Health check passed")
            print_info(f"Status: {data.get('status')}")
            gpu_service = data.get('gpu_service', {})
            if gpu_service.get('status') == 'healthy':
                print_success("GPU Service connectivity: OK")
            else:
                print_error(f"GPU Service connectivity: {gpu_service.get('status')}")
            results['health'] = True
        else:
            print_error(f"Health check failed: {response.status_code}")
            results['health'] = False
    except Exception as e:
        print_error(f"Health check error: {str(e)}")
        results['health'] = False
    
    # Test Sample Data Endpoint
    print_test("Sample Data Endpoint")
    try:
        response = requests.get(f"{API_GATEWAY_URL}/api/v1/sample-data", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success("Sample data endpoint working")
            print_info(f"Sample parameters retrieved")
            results['sample'] = True
        else:
            print_error(f"Sample data failed: {response.status_code}")
            results['sample'] = False
    except Exception as e:
        print_error(f"Sample data error: {str(e)}")
        results['sample'] = False
    
    # Test Prediction Endpoint
    print_test("Prediction Endpoint")
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
            results['predict'] = True
        else:
            print_error(f"Prediction failed: {response.status_code}")
            print_info(f"Response: {response.text[:200]}")
            results['predict'] = False
    except Exception as e:
        print_error(f"Prediction error: {str(e)}")
        results['predict'] = False
    
    # Test Optimize Endpoint
    print_test("Optimize Endpoint")
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
            cost_savings = result.get('cost_savings_estimate', {})
            print_info(f"Cost Savings: ${cost_savings.get('savings_per_wafer_usd', 0):.2f} per wafer")
            results['optimize'] = True
        else:
            print_error(f"Optimize failed: {response.status_code}")
            results['optimize'] = False
    except Exception as e:
        print_error(f"Optimize error: {str(e)}")
        results['optimize'] = False
    
    # Test Batch Prediction
    print_test("Batch Prediction Endpoint")
    batch_data = {
        "parameters_list": [
            test_data,
            {**test_data, "temperature": 205.0}
        ]
    }
    
    try:
        response = requests.post(
            f"{API_GATEWAY_URL}/api/v1/batch-predict",
            json=batch_data,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print_success(f"Batch prediction successful ({result.get('count', 0)} predictions)")
            print_info(f"Successful: {result.get('successful', 0)}")
            print_info(f"Failed: {result.get('failed', 0)}")
            results['batch'] = True
        else:
            print_error(f"Batch prediction failed: {response.status_code}")
            results['batch'] = False
    except Exception as e:
        print_error(f"Batch prediction error: {str(e)}")
        results['batch'] = False
    
    # Test Input Validation
    print_test("Input Validation")
    invalid_data = {
        "temperature": -10.0,  # Invalid: negative
        "pressure": 1.5,
        "etch_time": 60.0,
        "gas_flow": 100.0,
        "chamber_pressure": 5.0,
        "wafer_size": 300,
        "process_type": "etching"
    }
    
    try:
        response = requests.post(
            f"{API_GATEWAY_URL}/api/v1/predict",
            json=invalid_data,
            timeout=10
        )
        
        if response.status_code == 422:  # Validation error
            print_success("Validation working correctly (rejected invalid input)")
            results['validation'] = True
        else:
            print_error(f"Expected 422, got {response.status_code}")
            results['validation'] = False
    except Exception as e:
        print_error(f"Validation test error: {str(e)}")
        results['validation'] = False
    
    return results

def main():
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}ChipFabAI - Running Services Feature Test{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")
    print(f"{Colors.YELLOW}Testing actual running services with HTTP requests...{Colors.RESET}\n")
    
    # Test service availability
    availability = test_service_availability()
    
    if not availability.get('gpu') or not availability.get('api'):
        print(f"\n{Colors.RED}{Colors.BOLD}WARNING: Services not running! Please start them first.{Colors.RESET}\n")
        print(f"{Colors.YELLOW}To start services:{Colors.RESET}")
        print("  1. GPU Service: cd gpu-service && source ../venv/bin/activate && python main.py")
        print("  2. API Gateway: cd api-gateway && source ../venv/bin/activate && export GPU_SERVICE_URL=http://localhost:8080 && python main.py")
        return 1
    
    # Test features
    gpu_results = test_gpu_service_features()
    api_results = test_api_gateway_features()
    
    # Summary
    print_header("Test Summary")
    
    all_results = {
        "GPU Service": gpu_results,
        "API Gateway": api_results
    }
    
    total_tests = 0
    passed_tests = 0
    
    for service, results in all_results.items():
        print(f"\n{Colors.BOLD}{service}:{Colors.RESET}")
        for test, passed in results.items():
            total_tests += 1
            if passed:
                passed_tests += 1
                print(f"  {Colors.GREEN} {test}{Colors.RESET}")
            else:
                print(f"  {Colors.RED} {test}{Colors.RESET}")
    
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"Total Tests: {total_tests}")
    print(f"{Colors.GREEN}Passed: {passed_tests}{Colors.RESET}")
    print(f"{Colors.RED}Failed: {total_tests - passed_tests}{Colors.RESET}")
    print(f"{Colors.BOLD}Pass Rate: {(passed_tests/total_tests*100) if total_tests > 0 else 0:.1f}%{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")
    
    if passed_tests == total_tests:
        print(f"{Colors.GREEN}{Colors.BOLD} All features working correctly!{Colors.RESET}\n")
        return 0
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}WARNING: Some features failed. Check the output above.{Colors.RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())

