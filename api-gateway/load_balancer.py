"""
Load Balancing and Request Distribution for ChipFabAI API Gateway
Implements round-robin, least-connections, and health-check-based routing
Optimized for high-availability production deployment
"""

import asyncio
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import httpx
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    HEALTH_BASED = "health_based"
    RANDOM = "random"


@dataclass
class BackendServer:
    """Represents a backend server instance"""
    url: str
    active_connections: int = 0
    last_health_check: Optional[datetime] = None
    is_healthy: bool = True
    response_time: float = 0.0
    failure_count: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    
    def get_success_rate(self) -> float:
        """Calculate success rate for this backend"""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    def get_health_score(self) -> float:
        """
        Calculate health score (0-1) based on multiple factors
        Higher score = healthier backend
        """
        if not self.is_healthy:
            return 0.0
        
        # Factors: success rate (40%), response time (30%), active connections (30%)
        success_rate = self.get_success_rate()
        response_time_factor = max(0, 1 - (self.response_time / 5.0))  # Normalize to 5s
        connection_factor = max(0, 1 - (self.active_connections / 100))  # Normalize to 100
        
        score = (success_rate * 0.4 + response_time_factor * 0.3 + connection_factor * 0.3)
        return max(0.0, min(1.0, score))


class LoadBalancer:
    """
    Intelligent load balancer for distributing requests across multiple backend servers
    Supports multiple strategies and automatic health checking
    """
    
    def __init__(
        self,
        backend_urls: List[str],
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.HEALTH_BASED,
        health_check_interval: int = 30,
        health_check_timeout: float = 5.0,
        max_failures: int = 3
    ):
        """
        Initialize load balancer
        
        Args:
            backend_urls: List of backend server URLs
            strategy: Load balancing strategy to use
            health_check_interval: Seconds between health checks
            health_check_timeout: Timeout for health checks
            max_failures: Maximum consecutive failures before marking unhealthy
        """
        self.backends = [
            BackendServer(url=url) for url in backend_urls
        ]
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout
        self.max_failures = max_failures
        self.current_index = 0  # For round-robin
        self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(health_check_timeout))
        self.lock = asyncio.Lock()
        self.health_check_task: Optional[asyncio.Task] = None
        
    async def start_health_checks(self):
        """Start background health checking task"""
        if self.health_check_task is None:
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("Started health check loop")
    
    async def stop_health_checks(self):
        """Stop background health checking task"""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped health check loop")
    
    async def _health_check_loop(self):
        """Continuously check health of all backends"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_all_backends()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def _check_all_backends(self):
        """Check health of all backend servers"""
        tasks = [self._check_backend_health(backend) for backend in self.backends]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_backend_health(self, backend: BackendServer):
        """Check health of a single backend server"""
        try:
            start_time = time.time()
            response = await self.http_client.get(f"{backend.url}/health", timeout=self.health_check_timeout)
            response_time = time.time() - start_time
            
            async with self.lock:
                backend.last_health_check = datetime.now(timezone.utc)
                backend.response_time = response_time
                
                if response.status_code == 200:
                    backend.is_healthy = True
                    backend.failure_count = 0
                    logger.debug(f"Backend {backend.url} is healthy (response time: {response_time:.2f}s)")
                else:
                    backend.failure_count += 1
                    if backend.failure_count >= self.max_failures:
                        backend.is_healthy = False
                        logger.warning(f"Backend {backend.url} marked as unhealthy (failures: {backend.failure_count})")
        except Exception as e:
            async with self.lock:
                backend.failure_count += 1
                backend.response_time = self.health_check_timeout * 2  # Penalty for timeout
                if backend.failure_count >= self.max_failures:
                    backend.is_healthy = False
                logger.warning(f"Health check failed for {backend.url}: {e}")
    
    def _get_healthy_backends(self) -> List[BackendServer]:
        """Get list of healthy backend servers"""
        return [b for b in self.backends if b.is_healthy]
    
    async def select_backend(self) -> Optional[BackendServer]:
        """
        Select backend server based on load balancing strategy
        
        Returns:
            Selected backend server or None if no healthy backends available
        """
        healthy_backends = self._get_healthy_backends()
        
        if not healthy_backends:
            logger.error("No healthy backends available")
            return None
        
        async with self.lock:
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                backend = healthy_backends[self.current_index % len(healthy_backends)]
                self.current_index += 1
                return backend
            
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return min(healthy_backends, key=lambda b: b.active_connections)
            
            elif self.strategy == LoadBalancingStrategy.HEALTH_BASED:
                # Select backend with highest health score
                return max(healthy_backends, key=lambda b: b.get_health_score())
            
            elif self.strategy == LoadBalancingStrategy.RANDOM:
                import random
                return random.choice(healthy_backends)
            
            else:
                # Default to round-robin
                backend = healthy_backends[self.current_index % len(healthy_backends)]
                self.current_index += 1
                return backend
    
    async def execute_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        timeout: float = 60.0
    ) -> Tuple[Optional[Dict], Optional[BackendServer]]:
        """
        Execute request through load balancer
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request data for POST requests
            timeout: Request timeout in seconds
            
        Returns:
            Tuple of (response_data, backend_used) or (None, None) on failure
        """
        backend = await self.select_backend()
        if backend is None:
            return None, None
        
        # Increment active connections
        async with self.lock:
            backend.active_connections += 1
            backend.total_requests += 1
        
        try:
            url = f"{backend.url}/{endpoint.lstrip('/')}"
            start_time = time.time()
            
            if method.upper() == "POST":
                response = await self.http_client.post(
                    url,
                    json=data,
                    timeout=timeout
                )
            else:
                response = await self.http_client.get(url, timeout=timeout)
            
            response_time = time.time() - start_time
            
            # Update backend statistics
            async with self.lock:
                backend.response_time = response_time
                backend.active_connections -= 1
                
                if response.status_code < 400:
                    backend.successful_requests += 1
                    return response.json(), backend
                else:
                    backend.failure_count += 1
                    if backend.failure_count >= self.max_failures:
                        backend.is_healthy = False
                    return None, backend
                    
        except Exception as e:
            async with self.lock:
                backend.active_connections -= 1
                backend.failure_count += 1
                if backend.failure_count >= self.max_failures:
                    backend.is_healthy = False
                logger.error(f"Request failed for {backend.url}: {e}")
            return None, backend
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get load balancer statistics for monitoring
        
        Returns:
            Dictionary with load balancer statistics
        """
        healthy_count = len(self._get_healthy_backends())
        total_requests = sum(b.total_requests for b in self.backends)
        total_successful = sum(b.successful_requests for b in self.backends)
        
        return {
            "strategy": self.strategy.value,
            "total_backends": len(self.backends),
            "healthy_backends": healthy_count,
            "total_requests": total_requests,
            "successful_requests": total_successful,
            "success_rate": round(total_successful / total_requests * 100, 2) if total_requests > 0 else 0,
            "backends": [
                {
                    "url": b.url,
                    "healthy": b.is_healthy,
                    "active_connections": b.active_connections,
                    "total_requests": b.total_requests,
                    "success_rate": round(b.get_success_rate() * 100, 2),
                    "health_score": round(b.get_health_score(), 2),
                    "response_time": round(b.response_time, 3)
                }
                for b in self.backends
            ]
        }
    
    async def close(self):
        """Clean up resources"""
        await self.stop_health_checks()
        await self.http_client.aclose()

