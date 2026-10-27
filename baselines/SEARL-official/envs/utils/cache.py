# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Provide Redis caching for MCP calls and other components.'
import json
import hashlib
from typing import Any, Optional, Dict, Union
from redis_server.client import RedisClient

class RedisCacheManager:
    'Cache component results through RedisClient.'

    def __init__(self,
                 redis_host='localhost',
                 redis_port=6379,
                 default_ttl=3600,
                 prefix="cache",
                 enable_logging=True):
        'Initialize the Redis cache manager.'
        self.redis_client = RedisClient(host=redis_host, port=redis_port)
        self.default_ttl = default_ttl
        self.prefix = prefix
        self.enable_logging = enable_logging

        if self.enable_logging:
            print(f"RedisCacheManager initialized: {prefix} (TTL: {default_ttl}s)")

    def get_cache_key(self,
                     key_type: str,
                     identifier: Union[str, Dict, Any],
                     sub_key: Optional[str] = None) -> str:
        'Build a cache key.'

        if isinstance(identifier, dict):

            identifier_str = json.dumps(identifier, sort_keys=True, ensure_ascii=False)
        elif isinstance(identifier, (list, tuple)):

            identifier_str = json.dumps(identifier, ensure_ascii=False)
        else:

            identifier_str = str(identifier)

        hash_value = hashlib.md5(identifier_str.encode('utf-8')).hexdigest()

        if sub_key:
            full_key = f"{key_type}:{sub_key}:{hash_value}"
        else:
            full_key = f"{key_type}:{hash_value}"

        return self.redis_client.generate_key(full_key, prefix=self.prefix)

    def get(self, cache_key: str) -> Optional[Any]:
        'Read a cache entry.'
        try:
            result = self.redis_client.get_cache(cache_key)
            if result is not None and self.enable_logging:
                print(f"Cache hit: {cache_key}")
            return result
        except Exception as e:
            if self.enable_logging:
                print(f"Cache lookup failed: {cache_key} - {e}")
            return None

    def set(self,
            cache_key: str,
            value: Any,
            ttl: Optional[int] = None) -> bool:
        'Write a cache entry.'
        try:
            if ttl is None:
                ttl = self.default_ttl

            success = self.redis_client.set_cache(cache_key, value, ttl)
            if success and self.enable_logging:
                print(f"Cache set: {cache_key} (TTL: {ttl}s)")
            return success
        except Exception as e:
            if self.enable_logging:
                print(f"Cache update failed: {cache_key} - {e}")
            return False

    def delete(self, cache_key: str) -> bool:
        'Delete a cache entry.'
        try:
            success = self.redis_client.delete_cache(cache_key)
            if success and self.enable_logging:
                print(f"Cache deleted: {cache_key}")
            return success
        except Exception as e:
            if self.enable_logging:
                print(f"Cache deletion failed: {cache_key} - {e}")
            return False

    def should_cache(self,
                    key_type: str,
                    value: Any,
                    custom_rules: Optional[Dict] = None) -> bool:
        'Check whether a result should be cached.'

        if value is None:
            return False

        if isinstance(value, str):
            if not value.strip():
                return False

            error_indicators = ["error", "failed", "timeout", "exception", 'FAIL']
            if any(indicator in value.lower() for indicator in error_indicators):
                return False

        if isinstance(value, dict):

            if "error" in value or "failed" in value:
                return False

        if custom_rules:
            for rule_name, rule_func in custom_rules.items():
                if not rule_func(value):
                    if self.enable_logging:
                        print(f"Cache rejected (rule: {rule_name}): {key_type}")
                    return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        'Return cache statistics.'
        try:
            stats = self.redis_client.get_stats()
            stats['prefix'] = self.prefix
            stats['default_ttl'] = self.default_ttl
            return stats
        except Exception as e:
            if self.enable_logging:
                print(f"Failed to get statistics: {e}")
            return {}

    def clear_by_prefix(self, key_type: str) -> int:
        'Clear cache entries by key prefix.'
        try:

            if self.enable_logging:
                print(f"Clearing cache: {key_type} (not implemented)")
            return 0
        except Exception as e:
            if self.enable_logging:
                print(f"Cache cleanup failed: {e}")
            return 0

    def cache_mcp_tool(self, tool_name: str, tool_args: Dict, result: Any, ttl: Optional[int] = None) -> bool:
        'Cache an MCP tool result.'
        cache_key = self.get_cache_key("mcp_tool", f"{tool_name}:{tool_args}")
        if self.should_cache("mcp_tool", result):
            return self.set(cache_key, result, ttl)
        return False

    def get_mcp_tool(self, tool_name: str, tool_args: Dict) -> Optional[Any]:
        'Read a cached MCP tool result.'
        cache_key = self.get_cache_key("mcp_tool", f"{tool_name}:{tool_args}")
        return self.get(cache_key)

_cache_managers = {}

def get_cache_manager(**kwargs) -> RedisCacheManager:
    'Get the MCP cache-manager singleton.'
    cache_key = f"mcp_tool:{hash(str(kwargs))}"

    if cache_key not in _cache_managers:
        default_config = {
            "prefix": "mcp_tool",
            "default_ttl": 0,
        }
        default_config.update(kwargs)
        _cache_managers[cache_key] = RedisCacheManager(**default_config)

    return _cache_managers[cache_key]

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
