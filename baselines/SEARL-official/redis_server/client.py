# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Demonstrate the Redis cache client.'
import redis
import json
import hashlib
from typing import Any, Optional

class RedisClient:
    'Wrap Redis cache operations.'

    def __init__(self, host='localhost', port=6379, db=0, password=None, default_permanent=False):
        'Initialize the Redis client.'
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True
            )

            self.client.ping()
            self.default_permanent = default_permanent
            print(f"Redis connected: {host}:{port}")
            if default_permanent:
                print('Persistent caching enabled by default.')
        except redis.ConnectionError as e:
            print(f"Redis connection failed: {e}")
            raise

    def set_cache(self, key: str, value: Any, ttl: int = None) -> bool:
        'Set a cache entry.'
        try:

            json_value = json.dumps(value, ensure_ascii=False)

            if ttl is None:

                if self.default_permanent:
                    ttl = 0
                else:
                    ttl = 2592000

            if ttl == 0:

                result = self.client.set(key, json_value)
                print(f"Persistent cache set: {key} -> {len(json_value)} bytes")
            else:

                result = self.client.setex(key, ttl, json_value)
                print(f"Cache set: {key} -> {len(json_value)} bytes (TTL: {ttl}s)")

            return result
        except Exception as e:
            print(f"Cache update failed: {e}")
            return False

    def get_cache(self, key: str) -> Optional[Any]:
        'Get a cache entry.'
        try:
            json_value = self.client.get(key)
            if json_value:
                value = json.loads(json_value)
                print(f"Cache hit: {key}")
                return value
            else:
                print(f"Cache miss: {key}")
                return None
        except Exception as e:
            print(f"Cache lookup failed: {e}")
            return None

    def delete_cache(self, key: str) -> bool:
        'Delete a cache entry.'
        try:
            result = self.client.delete(key)
            print(f"Cache deleted: {key}")
            return bool(result)
        except Exception as e:
            print(f"Cache deletion failed: {e}")
            return False

    def generate_key(self, query: str, prefix: str = "search") -> str:
        'Generate a cache key.'

        hash_value = hashlib.md5(query.encode('utf-8')).hexdigest()
        return f"{prefix}:{hash_value}"

    def get_stats(self) -> dict:
        'Return Redis statistics.'
        try:
            info = self.client.info()
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }
        except Exception as e:
            print(f"Failed to get statistics: {e}")
            return {}

    def set_permanent_cache(self, key: str, value: Any) -> bool:
        'Store a cache entry without expiration.'
        return self.set_cache(key, value, ttl=0)

    def set_temporary_cache(self, key: str, value: Any, ttl: int = 3600) -> bool:
        'Store a cache entry with an explicit expiration.'
        return self.set_cache(key, value, ttl=ttl)

def demo_usage():
    'Demonstrate RedisClient usage.'
    print('Redis client demonstration')
    print("=" * 50)

    try:
        redis_client = RedisClient()
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    print('\nCache operations:')

    test_data = {
        "query": 'Python Redis tutorial',
        "results": [
            {"title": 'Redis documentation', "url": "https://redis.io/docs"},
            {"title": 'Python Redis library', "url": "https://pypi.org/project/redis"}
        ],
        "timestamp": "2025-09-08T12:00:00Z"
    }

    key = redis_client.generate_key('Python Redis tutorial')

    print('Setting a cache entry with a 30-day expiry...')
    redis_client.set_cache(key, test_data)

    permanent_key = redis_client.generate_key('Persistent cache test')
    print('Setting a persistent cache entry...')
    redis_client.set_permanent_cache(permanent_key, test_data)

    temp_key = redis_client.generate_key('Temporary cache test')
    print('Setting a cache entry with a five-minute expiry...')
    redis_client.set_temporary_cache(temp_key, test_data, ttl=300)

    cached_data = redis_client.get_cache(key)
    if cached_data:
        print(f"Retrieved data: {cached_data['query']}")

    cached_data2 = redis_client.get_cache(key)

    redis_client.delete_cache(key)

    print('\nRedis statistics:')
    stats = redis_client.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print('\nDemonstration completed.')

if __name__ == "__main__":
    # demo test
    demo_usage()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
