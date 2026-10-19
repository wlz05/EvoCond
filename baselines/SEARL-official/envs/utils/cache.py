# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'\nRedis\u7f13\u5b58\u7ba1\u7406\u5668\n\u7528\u4e8eMCP\u5de5\u5177\u8c03\u7528\u548c\u5176\u4ed6\u7ec4\u4ef6\u7684\u7f13\u5b58\u529f\u80fd\n'
import json
import hashlib
from typing import Any, Optional, Dict, Union
from redis_server.client import RedisClient

class RedisCacheManager:
    'Redis\u7f13\u5b58\u7ba1\u7406\u5668 - \u57fa\u4e8eRedisClient\u7684\u901a\u7528\u7f13\u5b58\u670d\u52a1'

    def __init__(self,
                 redis_host='localhost',
                 redis_port=6379,
                 default_ttl=3600,
                 prefix="cache",
                 enable_logging=True):
        '\u521d\u59cb\u5316Redis\u7f13\u5b58\u7ba1\u7406\u5668'
        self.redis_client = RedisClient(host=redis_host, port=redis_port)
        self.default_ttl = default_ttl
        self.prefix = prefix
        self.enable_logging = enable_logging

        if self.enable_logging:
            print(f"\U0001f680 RedisCacheManager\u521d\u59cb\u5316\u5b8c\u6210: {prefix} (TTL: {default_ttl}s)")

    def get_cache_key(self,
                     key_type: str,
                     identifier: Union[str, Dict, Any],
                     sub_key: Optional[str] = None) -> str:
        '\u751f\u6210\u7f13\u5b58\u952e'

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
        '\u83b7\u53d6\u7f13\u5b58'
        try:
            result = self.redis_client.get_cache(cache_key)
            if result is not None and self.enable_logging:
                print(f"\U0001f3af \u7f13\u5b58\u547d\u4e2d: {cache_key}")
            return result
        except Exception as e:
            if self.enable_logging:
                print(f"\u274c \u7f13\u5b58\u83b7\u53d6\u5931\u8d25: {cache_key} - {e}")
            return None

    def set(self,
            cache_key: str,
            value: Any,
            ttl: Optional[int] = None) -> bool:
        '\u8bbe\u7f6e\u7f13\u5b58'
        try:
            if ttl is None:
                ttl = self.default_ttl

            success = self.redis_client.set_cache(cache_key, value, ttl)
            if success and self.enable_logging:
                print(f"\U0001f4be \u7f13\u5b58\u8bbe\u7f6e: {cache_key} (TTL: {ttl}s)")
            return success
        except Exception as e:
            if self.enable_logging:
                print(f"\u274c \u7f13\u5b58\u8bbe\u7f6e\u5931\u8d25: {cache_key} - {e}")
            return False

    def delete(self, cache_key: str) -> bool:
        '\u5220\u9664\u7f13\u5b58'
        try:
            success = self.redis_client.delete_cache(cache_key)
            if success and self.enable_logging:
                print(f"\U0001f5d1\ufe0f \u7f13\u5b58\u5220\u9664: {cache_key}")
            return success
        except Exception as e:
            if self.enable_logging:
                print(f"\u274c \u7f13\u5b58\u5220\u9664\u5931\u8d25: {cache_key} - {e}")
            return False

    def should_cache(self,
                    key_type: str,
                    value: Any,
                    custom_rules: Optional[Dict] = None) -> bool:
        '\u5224\u65ad\u662f\u5426\u5e94\u8be5\u7f13\u5b58'

        if value is None:
            return False

        if isinstance(value, str):
            if not value.strip():
                return False

            error_indicators = ["error", "failed", "timeout", "exception", '\u274c']
            if any(indicator in value.lower() for indicator in error_indicators):
                return False

        if isinstance(value, dict):

            if "error" in value or "failed" in value:
                return False

        if custom_rules:
            for rule_name, rule_func in custom_rules.items():
                if not rule_func(value):
                    if self.enable_logging:
                        print(f"\U0001f6ab \u7f13\u5b58\u88ab\u62d2\u7edd (\u89c4\u5219: {rule_name}): {key_type}")
                    return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        '\u83b7\u53d6\u7f13\u5b58\u7edf\u8ba1\u4fe1\u606f'
        try:
            stats = self.redis_client.get_stats()
            stats['prefix'] = self.prefix
            stats['default_ttl'] = self.default_ttl
            return stats
        except Exception as e:
            if self.enable_logging:
                print(f"\u274c \u83b7\u53d6\u7edf\u8ba1\u4fe1\u606f\u5931\u8d25: {e}")
            return {}

    def clear_by_prefix(self, key_type: str) -> int:
        '\u6839\u636e\u952e\u7c7b\u578b\u6e05\u7406\u7f13\u5b58'
        try:

            if self.enable_logging:
                print(f"\U0001f9f9 \u6e05\u7406\u7f13\u5b58: {key_type} (\u529f\u80fd\u5f85\u5b9e\u73b0)")
            return 0
        except Exception as e:
            if self.enable_logging:
                print(f"\u274c \u6e05\u7406\u7f13\u5b58\u5931\u8d25: {e}")
            return 0

    def cache_mcp_tool(self, tool_name: str, tool_args: Dict, result: Any, ttl: Optional[int] = None) -> bool:
        '\u7f13\u5b58MCP\u5de5\u5177\u8c03\u7528\u7ed3\u679c'
        cache_key = self.get_cache_key("mcp_tool", f"{tool_name}:{tool_args}")
        if self.should_cache("mcp_tool", result):
            return self.set(cache_key, result, ttl)
        return False

    def get_mcp_tool(self, tool_name: str, tool_args: Dict) -> Optional[Any]:
        '\u83b7\u53d6MCP\u5de5\u5177\u8c03\u7528\u7f13\u5b58'
        cache_key = self.get_cache_key("mcp_tool", f"{tool_name}:{tool_args}")
        return self.get(cache_key)

_cache_managers = {}

def get_cache_manager(**kwargs) -> RedisCacheManager:
    '\u83b7\u53d6MCP\u5de5\u5177\u7f13\u5b58\u7ba1\u7406\u5668\u5355\u4f8b'
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
