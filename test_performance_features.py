"""
Comprehensive Test Suite for Performance Features
Tests caching, load balancing, and performance optimizations
Validates production-ready performance characteristics
"""

import asyncio
import time
import requests
import json
from typing import Dict, List
import statistics


class Colors:
    """ANSI color codes for terminal output"""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class PerformanceTestSuite:
    """Test suite for performance features"""
    
    def __init__(self, api_gateway_url: str = "http://localhost:8081"):
        self.api_gateway_url = api_gateway_url
        self.results = []
        
    def print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BLUE}{Colors.BOLD}{text}{Colors.RESET}")
        print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.RESET}\n")
    
    def print_test(self, test_name: str):
        """Print test name"""
        print(f"{Colors.YELLOW}Testing: {test_name}...{Colors.RESET}", end=" ", flush=True)
    
    def print_pass(self, message: str = "PASS"):
        """Print pass message"""
        print(f"{Colors.GREEN}PASS: {message}{Colors.RESET}")
    
    def print_fail(self, message: str = "FAIL"):
        """Print fail message"""
        print(f"{Colors.RED}FAIL: {message}{Colors.RESET}")
    
    def test_cache_functionality(self) -> Dict:
        """Test caching functionality"""
        self.print_test("Cache Functionality")
        results = {"passed": 0, "failed": 0, "tests": []}
        
        try:
            # Test parameters for caching
            test_params = {
                "temperature": 200.0,
                "pressure": 1.5,
                "etch_time": 60.0,
                "gas_flow": 100.0,
                "chamber_pressure": 5.0,
                "wafer_size": 300,
                "process_type": "etching"
            }
            
            # First request (cache miss)
            start_time = time.time()
            response1 = requests.post(
                f"{self.api_gateway_url}/api/v1/predict",
                json=test_params,
                timeout=30
            )
            first_request_time = time.time() - start_time
            
            if response1.status_code != 200:
                self.print_fail(f"First request failed: {response1.status_code}")
                results["failed"] += 1
                results["tests"].append({"test": "First request", "status": "failed"})
                return results
            
            result1 = response1.json()
            results["tests"].append({
                "test": "First request (cache miss)",
                "status": "passed",
                "response_time": first_request_time
            })
            results["passed"] += 1
            
            # Second request with same parameters (should be cached)
            start_time = time.time()
            response2 = requests.post(
                f"{self.api_gateway_url}/api/v1/predict",
                json=test_params,
                timeout=30
            )
            cached_request_time = time.time() - start_time
            
            if response2.status_code != 200:
                self.print_fail(f"Cached request failed: {response2.status_code}")
                results["failed"] += 1
                results["tests"].append({"test": "Cached request", "status": "failed"})
                return results
            
            result2 = response2.json()
            
            # Verify cache is working (cached request should be faster)
            speedup = first_request_time / cached_request_time if cached_request_time > 0 else 0
            
            if speedup > 1.5:  # Cached request should be at least 1.5x faster
                self.print_pass(f"Cache working (speedup: {speedup:.2f}x)")
                results["tests"].append({
                    "test": "Cache speedup",
                    "status": "passed",
                    "speedup": speedup,
                    "first_request_time": first_request_time,
                    "cached_request_time": cached_request_time
                })
                results["passed"] += 1
            else:
                self.print_fail(f"Cache not working effectively (speedup: {speedup:.2f}x)")
                results["tests"].append({
                    "test": "Cache speedup",
                    "status": "failed",
                    "speedup": speedup
                })
                results["failed"] += 1
            
            # Verify results are identical
            if result1.get("predicted_yield") == result2.get("predicted_yield"):
                self.print_pass("Cached results are identical")
                results["tests"].append({"test": "Result consistency", "status": "passed"})
                results["passed"] += 1
            else:
                self.print_fail("Cached results differ")
                results["tests"].append({"test": "Result consistency", "status": "failed"})
                results["failed"] += 1
                
        except Exception as e:
            self.print_fail(f"Error: {str(e)}")
            results["failed"] += 1
            results["tests"].append({"test": "Cache functionality", "status": "error", "error": str(e)})
        
        return results
    
    def test_cache_statistics(self) -> Dict:
        """Test cache statistics endpoint"""
        self.print_test("Cache Statistics")
        results = {"passed": 0, "failed": 0, "tests": []}
        
        try:
            response = requests.get(f"{self.api_gateway_url}/health", timeout=10)
            if response.status_code != 200:
                self.print_fail(f"Health check failed: {response.status_code}")
                results["failed"] += 1
                return results
            
            data = response.json()
            cache_stats = data.get("cache")
            
            if cache_stats:
                self.print_pass("Cache statistics available")
                results["tests"].append({
                    "test": "Cache statistics",
                    "status": "passed",
                    "stats": cache_stats
                })
                results["passed"] += 1
                
                # Verify cache has meaningful statistics
                if "hit_rate" in cache_stats and "size" in cache_stats:
                    self.print_pass(f"Cache hit rate: {cache_stats.get('hit_rate', 0)}%")
                    results["tests"].append({
                        "test": "Cache metrics",
                        "status": "passed",
                        "hit_rate": cache_stats.get("hit_rate", 0),
                        "size": cache_stats.get("size", 0)
                    })
                    results["passed"] += 1
                else:
                    self.print_fail("Cache statistics incomplete")
                    results["failed"] += 1
            else:
                self.print_fail("Cache statistics not available")
                results["failed"] += 1
                
        except Exception as e:
            self.print_fail(f"Error: {str(e)}")
            results["failed"] += 1
            results["tests"].append({"test": "Cache statistics", "status": "error", "error": str(e)})
        
        return results
    
    def test_metrics_endpoint(self) -> Dict:
        """Test metrics endpoint"""
        self.print_test("Metrics Endpoint")
        results = {"passed": 0, "failed": 0, "tests": []}
        
        try:
            response = requests.get(f"{self.api_gateway_url}/api/v1/metrics", timeout=10)
            if response.status_code != 200:
                self.print_fail(f"Metrics endpoint failed: {response.status_code}")
                results["failed"] += 1
                return results
            
            data = response.json()
            
            if "cache" in data or "load_balancer" in data or "requests_processed" in data:
                self.print_pass("Metrics endpoint working")
                results["tests"].append({
                    "test": "Metrics endpoint",
                    "status": "passed",
                    "metrics": data
                })
                results["passed"] += 1
            else:
                self.print_fail("Metrics endpoint missing data")
                results["failed"] += 1
                
        except Exception as e:
            self.print_fail(f"Error: {str(e)}")
            results["failed"] += 1
            results["tests"].append({"test": "Metrics endpoint", "status": "error", "error": str(e)})
        
        return results
    
    def test_concurrent_requests(self) -> Dict:
        """Test concurrent request handling"""
        self.print_test("Concurrent Request Handling")
        results = {"passed": 0, "failed": 0, "tests": []}
        
        try:
            import concurrent.futures
            
            test_params = {
                "temperature": 200.0,
                "pressure": 1.5,
                "etch_time": 60.0,
                "gas_flow": 100.0,
                "chamber_pressure": 5.0,
                "wafer_size": 300,
                "process_type": "etching"
            }
            
            def make_request():
                start = time.time()
                response = requests.post(
                    f"{self.api_gateway_url}/api/v1/predict",
                    json=test_params,
                    timeout=30
                )
                elapsed = time.time() - start
                return response.status_code == 200, elapsed
            
            # Make 10 concurrent requests
            num_requests = 10
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(make_request) for _ in range(num_requests)]
                results_list = [f.result() for f in concurrent.futures.as_completed(futures)]
            
            successful = sum(1 for success, _ in results_list if success)
            response_times = [elapsed for _, elapsed in results_list]
            
            if successful == num_requests:
                self.print_pass(f"All {num_requests} concurrent requests succeeded")
                results["tests"].append({
                    "test": "Concurrent requests",
                    "status": "passed",
                    "successful": successful,
                    "total": num_requests
                })
                results["passed"] += 1
            else:
                self.print_fail(f"Only {successful}/{num_requests} requests succeeded")
                results["tests"].append({
                    "test": "Concurrent requests",
                    "status": "failed",
                    "successful": successful,
                    "total": num_requests
                })
                results["failed"] += 1
            
            # Check average response time
            avg_response_time = statistics.mean(response_times)
            if avg_response_time < 5.0:  # Should be under 5 seconds
                self.print_pass(f"Average response time: {avg_response_time:.2f}s")
                results["tests"].append({
                    "test": "Response time",
                    "status": "passed",
                    "avg_response_time": avg_response_time
                })
                results["passed"] += 1
            else:
                self.print_fail(f"Average response time too high: {avg_response_time:.2f}s")
                results["tests"].append({
                    "test": "Response time",
                    "status": "failed",
                    "avg_response_time": avg_response_time
                })
                results["failed"] += 1
                
        except Exception as e:
            self.print_fail(f"Error: {str(e)}")
            results["failed"] += 1
            results["tests"].append({"test": "Concurrent requests", "status": "error", "error": str(e)})
        
        return results
    
    def test_load_balancer_health(self) -> Dict:
        """Test load balancer health checks"""
        self.print_test("Load Balancer Health")
        results = {"passed": 0, "failed": 0, "tests": []}
        
        try:
            response = requests.get(f"{self.api_gateway_url}/health", timeout=10)
            if response.status_code != 200:
                self.print_fail(f"Health check failed: {response.status_code}")
                results["failed"] += 1
                return results
            
            data = response.json()
            features = data.get("features", {})
            
            if features.get("load_balancing_enabled") is not None:
                self.print_pass("Load balancer status available")
                results["tests"].append({
                    "test": "Load balancer status",
                    "status": "passed",
                    "enabled": features.get("load_balancing_enabled", False)
                })
                results["passed"] += 1
            else:
                self.print_fail("Load balancer status not available")
                results["failed"] += 1
            
            # Check if load balancer stats are available when enabled
            if features.get("load_balancing_enabled"):
                lb_stats = data.get("load_balancer")
                if lb_stats:
                    self.print_pass("Load balancer statistics available")
                    results["tests"].append({
                        "test": "Load balancer statistics",
                        "status": "passed",
                        "stats": lb_stats
                    })
                    results["passed"] += 1
                else:
                    self.print_fail("Load balancer statistics not available")
                    results["failed"] += 1
                    
        except Exception as e:
            self.print_fail(f"Error: {str(e)}")
            results["failed"] += 1
            results["tests"].append({"test": "Load balancer health", "status": "error", "error": str(e)})
        
        return results
    
    def run_all_tests(self) -> Dict:
        """Run all performance tests"""
        self.print_header("Performance Features Test Suite")
        
        all_results = {
            "cache_functionality": self.test_cache_functionality(),
            "cache_statistics": self.test_cache_statistics(),
            "metrics_endpoint": self.test_metrics_endpoint(),
            "concurrent_requests": self.test_concurrent_requests(),
            "load_balancer_health": self.test_load_balancer_health()
        }
        
        # Calculate totals
        total_passed = sum(r["passed"] for r in all_results.values())
        total_failed = sum(r["failed"] for r in all_results.values())
        total_tests = total_passed + total_failed
        
        # Print summary
        self.print_header("Test Summary")
        print(f"{Colors.GREEN}Passed: {total_passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {total_failed}{Colors.RESET}")
        print(f"Total: {total_tests}")
        
        if total_tests > 0:
            pass_rate = (total_passed / total_tests) * 100
            print(f"Pass Rate: {pass_rate:.1f}%")
        
        return {
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_tests": total_tests,
            "pass_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
            "results": all_results
        }


def main():
    """Main test function"""
    import sys
    
    api_gateway_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8081"
    
    print(f"\n{Colors.BLUE}{Colors.BOLD}ChipFabAI Performance Features Test Suite{Colors.RESET}")
    print(f"{Colors.BLUE}API Gateway URL: {api_gateway_url}{Colors.RESET}\n")
    
    suite = PerformanceTestSuite(api_gateway_url)
    results = suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if results["total_failed"] == 0 else 1)


if __name__ == "__main__":
    main()

