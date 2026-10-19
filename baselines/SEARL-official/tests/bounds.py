# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Mocked retrieval boundaries without requests, MCP, or GPU dependencies."""
import ast
import json
from pathlib import Path
import types
import unittest
from unittest.mock import MagicMock
from urllib.parse import urlsplit

class RequestFailure(Exception):
    pass

def load_search():
    source = Path(__file__).resolve().parents[1] / "envs/tools/local.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"endpoint", "search"}]
    for node in functions:
        node.decorator_list = []
    scope = {"urlsplit": urlsplit, "json": json, "LOCAL_SEARCH_URL": "http://wiki:8000/retrieve",
             "MAX_BYTES": 1024, "requests": types.SimpleNamespace(RequestException=RequestFailure),
             "session": MagicMock()}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)
    return scope

def reply(scope, value=None, *, status=200, raw=None):
    response = MagicMock()
    response.status_code = status
    response.iter_content.return_value = [raw if raw is not None else json.dumps(value).encode()]
    scope["session"].post.return_value.__enter__.return_value = response
    return response

class SearchBoundsTests(unittest.TestCase):
    def setUp(self):
        self.scope = load_search()
        self.search = self.scope["search"]

    def test_only_explicit_retrieval_origin_path_is_allowed(self):
        for value in ("", "http://host", "http://host/other", "file:///retrieve", "http://u:p@host/retrieve",
                      "https://host/retrieve?token=secret", "http://host/retrieve#secret"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.scope["endpoint"](value)
        self.assertEqual(self.scope["endpoint"]("https://host:8443/retrieve"), "https://host:8443/retrieve")

    def test_invalid_config_fails_before_request_and_does_not_echo_it(self):
        self.scope["LOCAL_SEARCH_URL"] = "https://user:SECRET@host/retrieve"
        result = self.search("question")
        self.scope["session"].post.assert_not_called()
        self.assertNotIn("SECRET", result)
        self.assertTrue(result.startswith("Search error:"))

    def test_query_and_topk_are_bounded_before_network(self):
        for query in ("", "   ", 123, "x" * 16001):
            self.assertTrue(self.search(query).startswith("Search error:"))
        for topk in (True, 0, 11, 1.5, "3"):
            self.assertTrue(self.search("question", topk).startswith("Search error:"))
        self.scope["session"].post.assert_not_called()

    def test_requests_use_timeouts_streaming_and_no_redirects(self):
        reply(self.scope, {"result": [[{"document": {"contents": "Title\nContent"}}]]})
        result = self.search("question", 1)
        args, kwargs = self.scope["session"].post.call_args
        self.assertEqual(args[0], "http://wiki:8000/retrieve")
        self.assertEqual(kwargs["json"], {"queries": ["question"], "topk": 1, "return_scores": True})
        self.assertEqual(kwargs["timeout"], (5, 10))
        self.assertTrue(kwargs["stream"])
        self.assertFalse(kwargs["allow_redirects"])
        self.assertIn("Content", result)

    def test_redirect_response_is_never_read_or_followed(self):
        response = reply(self.scope, status=302)
        result = self.search("question")
        self.assertIn("302", result)
        response.iter_content.assert_not_called()
        self.assertEqual(self.scope["session"].post.call_count, 1)

    def test_oversized_response_fails_before_json_parse(self):
        response = reply(self.scope, raw=b"x" * 1025)
        self.assertEqual(self.search("question"), "Search error: response exceeded size limit")
        self.assertTrue(response.__exit__ is not None)
        self.scope["session"].post.return_value.__exit__.assert_called_once()

    def test_split_chunks_are_cumulatively_bounded(self):
        response = reply(self.scope)
        response.iter_content.return_value = [b"x" * 600, b"x" * 600]
        self.assertIn("exceeded size", self.search("question"))

    def test_request_exception_does_not_echo_secret_or_endpoint(self):
        self.scope["session"].post.side_effect = RequestFailure("token=SECRET password=HIDDEN https://private")
        result = self.search("question")
        self.assertNotIn("SECRET", result)
        self.assertNotIn("HIDDEN", result)
        self.assertNotIn("private", result)
        self.assertIn("RequestFailure", result)

    def test_malformed_json_or_schema_fails_closed(self):
        for value in (None, [], [1], {"result": [1]}, {"result": [[{}]]}, {"result": [[{"document": {"contents": 1}}]]}):
            with self.subTest(value=value):
                reply(self.scope, value)
                self.assertTrue(self.search("question").startswith("Search error:"))
        reply(self.scope, raw=b"{malformed SECRET")
        self.assertNotIn("SECRET", self.search("question"))

    def test_only_requested_number_of_passages_is_returned(self):
        reply(self.scope, {"result": [[{"document": {"contents": f"Title{i}\nContent{i}"}} for i in range(4)]]})
        result = self.search("question", 2)
        self.assertIn("Content1", result)
        self.assertNotIn("Content2", result)

    def test_no_documents_is_explicit(self):
        reply(self.scope, {"result": []})
        self.assertEqual(self.search("question"), "No relevant documents found")

    def test_deep_json_error_does_not_escape_or_reveal_payload(self):
        self.scope["MAX_BYTES"] = 10000
        reply(self.scope, raw=("[" * 2000 + '"SECRET"' + "]" * 2000).encode())
        result = self.search("question")
        self.assertTrue(result.startswith("Search error:"))
        self.assertNotIn("SECRET", result)

if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
