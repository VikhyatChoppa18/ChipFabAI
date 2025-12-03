"""
Caching layer for the API Gateway - stores prediction results to avoid hitting the GPU service
I use an LRU (Least Recently Used) cache with TTL (Time To Live) expiration
This gives us the best of both worlds: fast repeated queries and fresh data
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
    Thread-safe LRU cache with TTL expiration
    LRU means when the cache is full, we evict the least recently used item
    TTL means items automatically expire after a set time
    I made it thread-safe because FastAPI handles multiple requests concurrently
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """
        Initialize the cache
        
        max_size: How many items to keep before evicting old ones
        default_ttl: How long items stay valid (300 seconds = 5 minutes)
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
        Generate a cache key by hashing the request parameters
        I sort the keys first so the same parameters always produce the same hash
        even if they're in a different order in the dict
        """
        # Sort keys to ensure consistent hashing regardless of dict order
        sorted_data = json.dumps(data, sort_keys=True)
        return hashlib.md5(sorted_data.encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """Check if a cached item has passed its expiration time"""
        if key not in self.expiry_times:
            return True  # No expiry time means it's expired
        return time.time() > self.expiry_times[key]
    
    def _evict_lru(self):
        """
        Remove the least recently used item when the cache is full
        OrderedDict keeps items in insertion order, so the first one is the oldest
        """
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
        Get an item from the cache if it exists and hasn't expired
        When we access an item, we move it to the end (marking it as recently used)
        This is how LRU works - frequently accessed items stay, old ones get evicted
        """
        with self.lock:
            if key in self.cache:
                if self._is_expired(key):
                    # Item expired - remove it and count as a miss
                    del self.cache[key]
                    del self.expiry_times[key]
                    self.misses += 1
                    return None
                
                # Move to end to mark as recently used (LRU behavior)
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
        Store an item in the cache with an expiration time
        If the cache is full and this is a new key, evict the LRU item first
        """
        with self.lock:
            ttl = ttl or self.default_ttl
            
            # If cache is full and this is a new key, make room by evicting LRU
            if key not in self.cache and len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Store the value and record when it expires
            self.cache[key] = value
            self.expiry_times[key] = time.time() + ttl
            
            # Move to end to mark as recently used
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
        """
        Remove all expired items from the cache
        I call this periodically from a background task to keep memory usage down
        Without this, expired items would just sit there taking up space
        """
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
        Get cache performance statistics
        Useful for monitoring - hit rate tells you how effective the cache is
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


# Global cache instance - shared across all requests
# 1000 items is enough for most use cases, and 5 minute TTL keeps data fresh
prediction_cache = LRUCacheWithTTL(max_size=1000, default_ttl=300)


def get_cache_key(parameters: Dict[str, Any]) -> str:
    """
    Generate a cache key from the prediction parameters
    I exclude batch_id because it's request-specific and shouldn't affect caching
    Same process parameters should hit the cache regardless of which batch they're in
    """
    # Remove fields that don't affect the prediction result
    cacheable_params = {
        k: v for k, v in parameters.items() 
        if k not in ['batch_id']  # batch_id is just for tracking, not prediction
    }
    sorted_params = json.dumps(cacheable_params, sort_keys=True)
    return hashlib.md5(sorted_params.encode()).hexdigest()


def cached_predict(cache_key: str, prediction_func, *args, **kwargs) -> Any:
    """
    Helper function for caching prediction results
    Checks cache first, and if it's a miss, calls the prediction function and caches the result
    I don't actually use this in the current code, but it's here if you want a simpler interface
    """
    # Try cache first
    cached_result = prediction_cache.get(cache_key)
    if cached_result is not None:
        logger.info(f"Cache HIT for prediction: {cache_key[:8]}...")
        # Mark as cached in the response (useful for debugging)
        if isinstance(cached_result, dict):
            cached_result['_cached'] = True
        return cached_result
    
    # Cache miss - need to actually compute the prediction
    logger.info(f"Cache MISS for prediction: {cache_key[:8]}...")
    result = prediction_func(*args, **kwargs)
    
    # Store the result for next time
    if result is not None:
        # Make a copy before caching to avoid issues with mutable objects
        result_copy = result.copy() if isinstance(result, dict) else result
        prediction_cache.set(cache_key, result_copy)
        if isinstance(result, dict):
            result['_cached'] = False
    
    return result

