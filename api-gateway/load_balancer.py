"""
Load balancer for distributing requests across multiple GPU service instances
I built this to support horizontal scaling - when you have multiple GPU services running,
this intelligently routes requests to the healthiest/least loaded one
Supports several strategies: round-robin, least connections, health-based, and random
"""

import asyncio
import time
import logging
from typing import List, Dict, Optional, Tuple, Any
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
    """
    Tracks state for a single GPU service backend
    I keep track of health, load, and performance metrics to make smart routing decisions
    """
    url: str
    active_connections: int = 0
    last_health_check: Optional[datetime] = None
    is_healthy: bool = True
    response_time: float = 0.0
    failure_count: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    
    def get_success_rate(self) -> float:
        """What percentage of requests to this backend succeeded"""
        if self.total_requests == 0:
            return 1.0  # No data yet, assume it's good
        return self.successful_requests / self.total_requests
    
    def get_health_score(self) -> float:
        """
        Calculate a composite health score from 0-1
        I combine success rate, response time, and current load to get a single number
        Higher is better - this is what the health-based strategy uses to pick backends
        """
        if not self.is_healthy:
            return 0.0  # Unhealthy backends get zero score
        
        # Weighted combination: success rate matters most (40%), then response time (30%), then load (30%)
        success_rate = self.get_success_rate()
        # Normalize response time - 5 seconds is the threshold where it starts hurting the score
        response_time_factor = max(0, 1 - (self.response_time / 5.0))
        # Normalize active connections - 100 concurrent is where it starts to matter
        connection_factor = max(0, 1 - (self.active_connections / 100))
        
        score = (success_rate * 0.4 + response_time_factor * 0.3 + connection_factor * 0.3)
        return max(0.0, min(1.0, score))


class LoadBalancer:
    """
    Load balancer that distributes requests across multiple GPU service backends
    I run health checks in the background to know which backends are up
    Different strategies let you pick how to route requests based on your needs
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
        Initialize the load balancer with a list of backend URLs
        
        The health check runs every 30 seconds by default - frequent enough to catch
        issues quickly, but not so often that it adds overhead
        max_failures prevents flapping - a backend needs to fail 3 times in a row
        before we mark it unhealthy (avoids false positives from transient network issues)
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
        """
        Background loop that continuously checks backend health
        Runs forever until cancelled during shutdown
        """
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_all_backends()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def _check_all_backends(self):
        """
        Check all backends in parallel - much faster than doing them sequentially
        I use gather with return_exceptions so one failing backend doesn't break the others
        """
        tasks = [self._check_backend_health(backend) for backend in self.backends]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_backend_health(self, backend: BackendServer):
        """
        Check if a single backend is healthy by hitting its /health endpoint
        I track response time here too - slow backends get lower health scores
        """
        try:
            start_time = time.time()
            response = await self.http_client.get(f"{backend.url}/health", timeout=self.health_check_timeout)
            response_time = time.time() - start_time
            
            async with self.lock:
                backend.last_health_check = datetime.now(timezone.utc)
                backend.response_time = response_time
                
                if response.status_code == 200:
                    backend.is_healthy = True
                    backend.failure_count = 0  # Reset on success
                    logger.debug(f"Backend {backend.url} is healthy (response time: {response_time:.2f}s)")
                else:
                    backend.failure_count += 1
                    if backend.failure_count >= self.max_failures:
                        backend.is_healthy = False
                        logger.warning(f"Backend {backend.url} marked as unhealthy (failures: {backend.failure_count})")
        except Exception as e:
            # Network error or timeout - treat as failure
            async with self.lock:
                backend.failure_count += 1
                # Penalize timeouts heavily in the response time metric
                backend.response_time = self.health_check_timeout * 2
                if backend.failure_count >= self.max_failures:
                    backend.is_healthy = False
                logger.warning(f"Health check failed for {backend.url}: {e}")
    
    def _get_healthy_backends(self) -> List[BackendServer]:
        """Get list of healthy backend servers"""
        return [b for b in self.backends if b.is_healthy]
    
    async def select_backend(self) -> Optional[BackendServer]:
        """
        Pick which backend to use for the next request based on the selected strategy
        Only considers healthy backends - unhealthy ones are automatically excluded
        """
        healthy_backends = self._get_healthy_backends()
        
        if not healthy_backends:
            logger.error("No healthy backends available")
            return None
        
        async with self.lock:
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                # Simple round-robin - just cycle through backends in order
                backend = healthy_backends[self.current_index % len(healthy_backends)]
                self.current_index += 1
                return backend
            
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                # Send to the backend with fewest active connections
                # Good for balancing load when requests take different amounts of time
                return min(healthy_backends, key=lambda b: b.active_connections)
            
            elif self.strategy == LoadBalancingStrategy.HEALTH_BASED:
                # My favorite - picks the backend with the best health score
                # This considers success rate, response time, and current load
                return max(healthy_backends, key=lambda b: b.get_health_score())
            
            elif self.strategy == LoadBalancingStrategy.RANDOM:
                # Random selection - sometimes useful for testing
                import random
                return random.choice(healthy_backends)
            
            else:
                # Fallback to round-robin if strategy is unknown
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
        Execute a request through the load balancer
        This picks a backend, sends the request, and tracks the results
        Returns the response data and which backend was used (useful for debugging)
        """
        backend = await self.select_backend()
        if backend is None:
            return None, None  # No healthy backends available
        
        # Track that this backend is handling a request
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
            
            # Update metrics based on how the request went
            async with self.lock:
                backend.response_time = response_time
                backend.active_connections -= 1  # Request finished
                
                if response.status_code < 400:
                    backend.successful_requests += 1
                    return response.json(), backend
                else:
                    # HTTP error - count as failure
                    backend.failure_count += 1
                    if backend.failure_count >= self.max_failures:
                        backend.is_healthy = False  # Too many failures, mark unhealthy
                    return None, backend
                    
        except Exception as e:
            # Network error or timeout
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

