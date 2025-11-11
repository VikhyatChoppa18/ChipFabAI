#!/usr/bin/env python3
"""
Production Test Suite for ChipFabAI
Comprehensive testing for GPU category submission
Tests all features, endpoints, and production readiness
"""

import os
import requests
import json
import time
import sys
from typing import Dict, List
from datetime import datetime

# Configuration
GPU_SERVICE_URL = os.getenv("GPU_SERVICE_URL", "http://localhost:8080")
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

class ProductionTestSuite:
    def __init__(self):
        self.results = {
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "tests": []
        }
    
    def print_header(self, text: str):
        print(f"\n{Colors.BLUE}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}")
        print(f"{Colors.BLUE}{'='*70}{Colors.RESET}\n")
    
    def print_test(self, test_name: str):
        print(f"{Colors.YELLOW} {test_name}...{Colors.RESET}")
    
    def print_success(self, message: str):
        print(f"{Colors.GREEN} {message}{Colors.RESET}")
        self.results["passed"] += 1
    
    def print_error(self, message: str):
        print(f"{Colors.RED} {message}{Colors.RESET}")
        self.results["failed"] += 1
    
    def print_warning(self, message: str):
        print(f"{Colors.YELLOW}WARNING: {message}{Colors.RESET}")
        self.results["warnings"] += 1
    
    def print_info(self, message: str):
        print(f"   {message}")
    
    def test_gpu_service_health(self) -> bool:
        """Test GPU Service health endpoint"""
        self.print_test("GPU Service Health Check")
        try:
            response = requests.get(f"{GPU_SERVICE_URL}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.print_success("GPU Service is healthy")
                self.print_info(f"Status: {data.get('status')}")
                self.print_info(f"Model loaded: {data.get('model_loaded')}")
                self.print_info(f"Device: {data.get('device')}")
                if data.get('gpu'):
                    gpu = data['gpu']
                    self.print_info(f"GPU: {gpu.get('device_name')}")
                    self.print_info(f"GPU Memory: {gpu.get('memory_total_gb')} GB")
                return True
            else:
                self.print_error(f"GPU Service returned status {response.status_code}")
                return False
        except Exception as e:
            self.print_error(f"GPU Service health check failed: {str(e)}")
            return False
    
    def test_gpu_service_prediction(self) -> bool:
        """Test GPU Service prediction endpoint"""
        self.print_test("GPU Service Prediction")
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
                self.print_success("Prediction successful")
                self.print_info(f"Predicted Yield: {result.get('predicted_yield')}%")
                self.print_info(f"Confidence: {result.get('confidence', 0) * 100:.1f}%")
                self.print_info(f"Processing Time: {result.get('processing_time_ms', 0):.0f}ms")
                self.print_info(f"Total Time: {elapsed:.2f}s")
                
                # Validate response structure
                required_fields = ['predicted_yield', 'optimal_temperature', 'optimal_pressure', 
                                 'risk_factors', 'recommendations', 'confidence']
                missing = [f for f in required_fields if f not in result]
                if missing:
                    self.print_warning(f"Missing fields: {missing}")
                else:
                    self.print_success("Response structure valid")
                
                return True
            else:
                self.print_error(f"Prediction failed: {response.status_code}")
                self.print_info(f"Response: {response.text[:200]}")
                return False
        except Exception as e:
            self.print_error(f"Prediction error: {str(e)}")
            return False
    
    def test_api_gateway_health(self) -> bool:
        """Test API Gateway health endpoint"""
        self.print_test("API Gateway Health Check")
        try:
            response = requests.get(f"{API_GATEWAY_URL}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.print_success("API Gateway is healthy")
                self.print_info(f"Status: {data.get('status')}")
                gpu_service = data.get('gpu_service', {})
                if gpu_service.get('status') == 'healthy':
                    self.print_success("GPU Service connectivity: OK")
                else:
                    self.print_warning(f"GPU Service connectivity: {gpu_service.get('status')}")
                return True
            else:
                self.print_error(f"API Gateway returned status {response.status_code}")
                return False
        except Exception as e:
            self.print_error(f"API Gateway health check failed: {str(e)}")
            return False
    
    def test_api_gateway_prediction(self) -> bool:
        """Test API Gateway prediction endpoint"""
        self.print_test("API Gateway Prediction")
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
                timeout=60,
                headers={"Content-Type": "application/json"}
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                self.print_success("Prediction through API Gateway successful")
                self.print_info(f"Predicted Yield: {result.get('predicted_yield')}%")
                self.print_info(f"Confidence: {result.get('confidence', 0) * 100:.1f}%")
                self.print_info(f"Processing Time: {result.get('processing_time_ms', 0):.0f}ms")
                self.print_info(f"Total Time: {elapsed:.2f}s")
                return True
            else:
                self.print_error(f"Prediction failed: {response.status_code}")
                self.print_info(f"Response: {response.text[:200]}")
                return False
        except Exception as e:
            self.print_error(f"Prediction error: {str(e)}")
            return False
    
    def test_api_gateway_optimize(self) -> bool:
        """Test API Gateway optimize endpoint"""
        self.print_test("API Gateway Optimize Endpoint")
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
            response = requests.post(
                f"{API_GATEWAY_URL}/api/v1/optimize",
                json=test_data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                self.print_success("Optimize endpoint working")
                self.print_info(f"Optimization Score: {result.get('optimization_score', 0)}")
                cost_savings = result.get('cost_savings_estimate', {})
                self.print_info(f"Cost Savings: ${cost_savings.get('savings_per_wafer_usd', 0):.2f} per wafer")
                return True
            else:
                self.print_error(f"Optimize failed: {response.status_code}")
                return False
        except Exception as e:
            self.print_error(f"Optimize error: {str(e)}")
            return False
    
    def test_batch_prediction(self) -> bool:
        """Test batch prediction"""
        self.print_test("Batch Prediction")
        batch_data = {
            "parameters_list": [
                {
                    "temperature": 200.0,
                    "pressure": 1.5,
                    "etch_time": 60.0,
                    "gas_flow": 100.0,
                    "chamber_pressure": 5.0,
                    "wafer_size": 300,
                    "process_type": "etching"
                },
                {
                    "temperature": 205.0,
                    "pressure": 1.6,
                    "etch_time": 65.0,
                    "gas_flow": 105.0,
                    "chamber_pressure": 5.5,
                    "wafer_size": 300,
                    "process_type": "etching"
                }
            ]
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{API_GATEWAY_URL}/api/v1/batch-predict",
                json=batch_data,
                timeout=120
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                self.print_success(f"Batch prediction successful ({result.get('count', 0)} predictions)")
                self.print_info(f"Successful: {result.get('successful', 0)}")
                self.print_info(f"Failed: {result.get('failed', 0)}")
                self.print_info(f"Total Time: {elapsed:.2f}s")
                self.print_info(f"Average Time: {result.get('average_time_per_prediction_ms', 0):.0f}ms per prediction")
                return True
            else:
                self.print_error(f"Batch prediction failed: {response.status_code}")
                return False
        except Exception as e:
            self.print_error(f"Batch prediction error: {str(e)}")
            return False
    
    def test_validation(self) -> bool:
        """Test input validation"""
        self.print_test("Input Validation")
        
        # Test invalid temperature
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
                self.print_success("Validation working correctly (rejected invalid input)")
                return True
            else:
                self.print_warning(f"Expected 422, got {response.status_code}")
                return False
        except Exception as e:
            self.print_error(f"Validation test error: {str(e)}")
            return False
    
    def test_performance(self) -> bool:
        """Test performance metrics"""
        self.print_test("Performance Test")
        test_data = {
            "temperature": 200.0,
            "pressure": 1.5,
            "etch_time": 60.0,
            "gas_flow": 100.0,
            "chamber_pressure": 5.0,
            "wafer_size": 300,
            "process_type": "etching"
        }
        
        times = []
        for i in range(5):
            try:
                start = time.time()
                response = requests.post(
                    f"{API_GATEWAY_URL}/api/v1/predict",
                    json=test_data,
                    timeout=60
                )
                elapsed = time.time() - start
                if response.status_code == 200:
                    times.append(elapsed)
            except:
                pass
        
        if times:
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            
            self.print_info(f"Average response time: {avg_time:.2f}s")
            self.print_info(f"Min: {min_time:.2f}s, Max: {max_time:.2f}s")
            
            if avg_time < 5.0:
                self.print_success("Performance acceptable (<5s average)")
                return True
            else:
                self.print_warning(f"Performance slow (>{5.0}s average)")
                return False
        else:
            self.print_error("Performance test failed - no successful requests")
            return False
    
    def test_frontend(self) -> bool:
        """Test frontend availability"""
        self.print_test("Frontend Availability")
        try:
            response = requests.get(FRONTEND_URL, timeout=5)
            if response.status_code == 200:
                self.print_success("Frontend is accessible")
                return True
            else:
                self.print_warning(f"Frontend returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            self.print_warning("Frontend is not running (expected if not started)")
            return False
        except Exception as e:
            self.print_warning(f"Frontend check: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all production tests"""
        self.print_header("ChipFabAI Production Test Suite")
        self.print_info(f"GPU Service URL: {GPU_SERVICE_URL}")
        self.print_info(f"API Gateway URL: {API_GATEWAY_URL}")
        self.print_info(f"Frontend URL: {FRONTEND_URL}")
        self.print_info(f"Test Time: {datetime.now().isoformat()}\n")
        
        # GPU Service Tests
        self.print_header("GPU Service Tests")
        self.test_gpu_service_health()
        self.test_gpu_service_prediction()
        
        # API Gateway Tests
        self.print_header("API Gateway Tests")
        self.test_api_gateway_health()
        self.test_api_gateway_prediction()
        self.test_api_gateway_optimize()
        self.test_batch_prediction()
        
        # Production Readiness Tests
        self.print_header("Production Readiness Tests")
        self.test_validation()
        self.test_performance()
        self.test_frontend()
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        self.print_header("Test Summary")
        
        total = self.results["passed"] + self.results["failed"]
        pass_rate = (self.results["passed"] / total * 100) if total > 0 else 0
        
        print(f"{Colors.BOLD}Total Tests: {total}{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {self.results['passed']}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {self.results['failed']}{Colors.RESET}")
        print(f"{Colors.YELLOW}Warnings: {self.results['warnings']}{Colors.RESET}")
        print(f"{Colors.BOLD}Pass Rate: {pass_rate:.1f}%{Colors.RESET}\n")
        
        if self.results["failed"] == 0:
            print(f"{Colors.GREEN}{Colors.BOLD} All critical tests passed!{Colors.RESET}")
            print(f"{Colors.GREEN} System is production-ready for GPU category submission{Colors.RESET}\n")
            return 0
        else:
            print(f"{Colors.RED}{Colors.BOLD}WARNING: Some tests failed. Please review and fix issues.{Colors.RESET}\n")
            return 1


if __name__ == "__main__":
    import os
    
    # Allow environment variable overrides
    GPU_SERVICE_URL = os.getenv("GPU_SERVICE_URL", "http://localhost:8080")
    API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://localhost:8081")
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
    
    suite = ProductionTestSuite()
    exit_code = suite.run_all_tests()
    sys.exit(exit_code)

