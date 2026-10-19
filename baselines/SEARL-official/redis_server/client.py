# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'\nRedis\u5ba2\u6237\u7aef\u4f7f\u7528\u793a\u4f8b\n\u6f14\u793a\u5982\u4f55\u8fde\u63a5\u548c\u4f7f\u7528Redis\u7f13\u5b58\u670d\u52a1\n'
import redis
import json
import hashlib
from typing import Any, Optional

class RedisClient:
    'Redis\u5ba2\u6237\u7aef\u5c01\u88c5\u7c7b'

    def __init__(self, host='localhost', port=6379, db=0, password=None, default_permanent=False):
        '\n        \u521d\u59cb\u5316Redis\u5ba2\u6237\u7aef\n        '
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
            print(f"\u2705 Redis\u8fde\u63a5\u6210\u529f: {host}:{port}")
            if default_permanent:
                print('\U0001f512 \u9ed8\u8ba4\u6c38\u4e45\u7f13\u5b58\u6a21\u5f0f\u5df2\u542f\u7528')
        except redis.ConnectionError as e:
            print(f"\u274c Redis\u8fde\u63a5\u5931\u8d25: {e}")
            raise

    def set_cache(self, key: str, value: Any, ttl: int = None) -> bool:
        '\n        \u8bbe\u7f6e\u7f13\u5b58\n        '
        try:

            json_value = json.dumps(value, ensure_ascii=False)

            if ttl is None:

                if self.default_permanent:
                    ttl = 0
                else:
                    ttl = 2592000

            if ttl == 0:

                result = self.client.set(key, json_value)
                print(f"\U0001f4dd \u6c38\u4e45\u7f13\u5b58\u8bbe\u7f6e: {key} -> {len(json_value)} bytes")
            else:

                result = self.client.setex(key, ttl, json_value)
                print(f"\U0001f4dd \u7f13\u5b58\u8bbe\u7f6e: {key} -> {len(json_value)} bytes (TTL: {ttl}s)")

            return result
        except Exception as e:
            print(f"\u274c \u7f13\u5b58\u8bbe\u7f6e\u5931\u8d25: {e}")
            return False

    def get_cache(self, key: str) -> Optional[Any]:
        '\n        \u83b7\u53d6\u7f13\u5b58\n        '
        try:
            json_value = self.client.get(key)
            if json_value:
                value = json.loads(json_value)
                print(f"\U0001f4d6 \u7f13\u5b58\u547d\u4e2d: {key}")
                return value
            else:
                print(f"\U0001f50d \u7f13\u5b58\u672a\u547d\u4e2d: {key}")
                return None
        except Exception as e:
            print(f"\u274c \u7f13\u5b58\u83b7\u53d6\u5931\u8d25: {e}")
            return None

    def delete_cache(self, key: str) -> bool:
        '\n        \u5220\u9664\u7f13\u5b58\n        '
        try:
            result = self.client.delete(key)
            print(f"\U0001f5d1\ufe0f \u7f13\u5b58\u5220\u9664: {key}")
            return bool(result)
        except Exception as e:
            print(f"\u274c \u7f13\u5b58\u5220\u9664\u5931\u8d25: {e}")
            return False

    def generate_key(self, query: str, prefix: str = "search") -> str:
        '\n        \u751f\u6210\u7f13\u5b58\u952e\n        '

        hash_value = hashlib.md5(query.encode('utf-8')).hexdigest()
        return f"{prefix}:{hash_value}"

    def get_stats(self) -> dict:
        '\n        \u83b7\u53d6Redis\u7edf\u8ba1\u4fe1\u606f\n        '
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
            print(f"\u274c \u83b7\u53d6\u7edf\u8ba1\u4fe1\u606f\u5931\u8d25: {e}")
            return {}

    def set_permanent_cache(self, key: str, value: Any) -> bool:
        '\n        \u8bbe\u7f6e\u6c38\u4e45\u7f13\u5b58\uff08\u6c38\u4e0d\u8fc7\u671f\uff09\n        '
        return self.set_cache(key, value, ttl=0)

    def set_temporary_cache(self, key: str, value: Any, ttl: int = 3600) -> bool:
        '\n        \u8bbe\u7f6e\u4e34\u65f6\u7f13\u5b58\uff08\u6307\u5b9a\u8fc7\u671f\u65f6\u95f4\uff09\n        '
        return self.set_cache(key, value, ttl=ttl)

def demo_usage():
    '\u6f14\u793aRedis\u5ba2\u6237\u7aef\u7684\u4f7f\u7528\u65b9\u6cd5'
    print('\U0001f680 Redis\u5ba2\u6237\u7aef\u4f7f\u7528\u6f14\u793a')
    print("=" * 50)

    try:
        redis_client = RedisClient()
    except Exception as e:
        print(f"\u521d\u59cb\u5316\u5931\u8d25: {e}")
        return

    print('\n\U0001f4dd \u6f14\u793a\u7f13\u5b58\u64cd\u4f5c:')

    test_data = {
        "query": 'Python Redis\u6559\u7a0b',
        "results": [
            {"title": 'Redis\u5b98\u65b9\u6587\u6863', "url": "https://redis.io/docs"},
            {"title": 'Python Redis\u5e93', "url": "https://pypi.org/project/redis"}
        ],
        "timestamp": "2025-09-08T12:00:00Z"
    }

    key = redis_client.generate_key('Python Redis\u6559\u7a0b')

    print('\u8bbe\u7f6e30\u5929\u8fc7\u671f\u7f13\u5b58...')
    redis_client.set_cache(key, test_data)

    permanent_key = redis_client.generate_key('\u6c38\u4e45\u7f13\u5b58\u6d4b\u8bd5')
    print('\u8bbe\u7f6e\u6c38\u4e45\u7f13\u5b58...')
    redis_client.set_permanent_cache(permanent_key, test_data)

    temp_key = redis_client.generate_key('\u4e34\u65f6\u7f13\u5b58\u6d4b\u8bd5')
    print('\u8bbe\u7f6e5\u5206\u949f\u4e34\u65f6\u7f13\u5b58...')
    redis_client.set_temporary_cache(temp_key, test_data, ttl=300)

    cached_data = redis_client.get_cache(key)
    if cached_data:
        print(f"\u83b7\u53d6\u5230\u7684\u6570\u636e: {cached_data['query']}")

    cached_data2 = redis_client.get_cache(key)

    redis_client.delete_cache(key)

    print('\n\U0001f4ca Redis\u7edf\u8ba1\u4fe1\u606f:')
    stats = redis_client.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print('\n\u2705 \u6f14\u793a\u5b8c\u6210!')

if __name__ == "__main__":
    # demo test
    demo_usage()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
