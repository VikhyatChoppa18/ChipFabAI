"""
Intelligent Caching Layer for ChipFabAI API Gateway
Implements LRU cache with TTL for prediction results and response caching
Optimized for high-performance production deployment
"""

import time
import hashlib
import json
from typing import Optional, Dict, Any
from collections import OrderedDict
import threading
import logging

logger = logging.getLogger(__name__)


class LRUCacheWithTTL:
    """
    Thread-safe LRU (Least Recently Used) cache with TTL (Time To Live)
    Implements efficient caching with automatic expiration and size limits
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """
        Initialize LRU cache with TTL
        
        Args:
            max_size: Maximum number of items to cache (evicts oldest when full)
            default_ttl: Default time-to-live in seconds (5 minutes)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict = OrderedDict()
        self.expiry_times: Dict[str, float] = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def _generate_key(self, data: Dict[str, Any]) -> str:
        """
        Generate cache key from request data using MD5 hash
        Creates consistent keys for identical requests
        """
        # Sort keys to ensure consistent hashing
        sorted_data = json.dumps(data, sort_keys=True)
        return hashlib.md5(sorted_data.encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cached item has expired based on TTL"""
        if key not in self.expiry_times:
            return True
        return time.time() > self.expiry_times[key]
    
    def _evict_lru(self):
        """Evict least recently used item when cache is full"""
        if len(self.cache) >= self.max_size:
            # Remove oldest item (first in OrderedDict)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            if oldest_key in self.expiry_times:
                del self.expiry_times[oldest_key]
            self.evictions += 1
            logger.debug(f"Evicted cache key: {oldest_key}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve item from cache if not expired
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self.lock:
            if key in self.cache:
                if self._is_expired(key):
                    # Remove expired item
                    del self.cache[key]
                    del self.expiry_times[key]
                    self.misses += 1
                    return None
                
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                logger.debug(f"Cache HIT: {key}")
                return value
            
            self.misses += 1
            logger.debug(f"Cache MISS: {key}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Store item in cache with TTL
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        with self.lock:
            ttl = ttl or self.default_ttl
            
            # Evict if cache is full
            if key not in self.cache and len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Store item and expiry time
            self.cache[key] = value
            self.expiry_times[key] = time.time() + ttl
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            logger.debug(f"Cached item: {key} (TTL: {ttl}s)")
    
    def delete(self, key: str):
        """Remove item from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            if key in self.expiry_times:
                del self.expiry_times[key]
            logger.debug(f"Deleted cache key: {key}")
    
    def clear(self):
        """Clear all cached items"""
        with self.lock:
            self.cache.clear()
            self.expiry_times.clear()
            logger.info("Cache cleared")
    
    def cleanup_expired(self):
        """Remove all expired items from cache"""
        with self.lock:
            expired_keys = [
                key for key in self.expiry_times 
                if self._is_expired(key)
            ]
            for key in expired_keys:
                if key in self.cache:
                    del self.cache[key]
                del self.expiry_times[key]
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(hit_rate, 2),
                "evictions": self.evictions,
                "default_ttl": self.default_ttl
            }


# Global cache instance for prediction results
prediction_cache = LRUCacheWithTTL(max_size=1000, default_ttl=300)  # 5 minute TTL


def get_cache_key(parameters: Dict[str, Any]) -> str:
    """
    Generate cache key from prediction parameters
    
    Args:
        parameters: Process parameters dictionary
        
    Returns:
        MD5 hash of parameters as cache key
    """
    # Remove non-deterministic fields for caching
    cacheable_params = {
        k: v for k, v in parameters.items() 
        if k not in ['batch_id']  # Exclude request-specific fields
    }
    sorted_params = json.dumps(cacheable_params, sort_keys=True)
    return hashlib.md5(sorted_params.encode()).hexdigest()


def cached_predict(cache_key: str, prediction_func, *args, **kwargs) -> Any:
    """
    Decorator-like function for caching prediction results
    
    Args:
        cache_key: Cache key for the prediction
        prediction_func: Function to call if cache miss
        *args: Arguments for prediction function
        **kwargs: Keyword arguments for prediction function
        
    Returns:
        Cached or newly computed prediction result
    """
    # Try to get from cache
    cached_result = prediction_cache.get(cache_key)
    if cached_result is not None:
        logger.info(f"Cache HIT for prediction: {cache_key[:8]}...")
        # Add cache indicator to response
        if isinstance(cached_result, dict):
            cached_result['_cached'] = True
        return cached_result
    
    # Cache miss - compute prediction
    logger.info(f"Cache MISS for prediction: {cache_key[:8]}...")
    result = prediction_func(*args, **kwargs)
    
    # Store in cache
    if result is not None:
        # Create copy to avoid modifying cached object
        result_copy = result.copy() if isinstance(result, dict) else result
        prediction_cache.set(cache_key, result_copy)
        if isinstance(result, dict):
            result['_cached'] = False
    
    return result

