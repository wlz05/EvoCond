# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'CPU mocks only; these tests do not attest a deployed sandbox.'
from copy import deepcopy
from datetime import datetime, timezone
import json
import re
import unittest
from unittest.mock import patch
from control.dynamic import SandboxFusionBackend
from control.isolation import (CHECKS, VerifiedSandboxFusion, canonical_hash,
                                     finite_inputs, function_schema, strict_json,
                                     validate_arguments, validate_deployment)
def deployment():
    config = {"format": "searl_isolation_config_v1", "endpoint": "http://sandbox:8080",
              "deployment_id": "CPU_MOCK_NOT_A_DEPLOYMENT", "isolation": "container",
              "max_run_seconds": 30, "memory_limit_mb": 256, "max_output_bytes": 4096}
    receipt = {"format": "searl_isolation_receipt_v1", "endpoint": config["endpoint"],
               "deployment_id": config["deployment_id"], "config_sha256": canonical_hash(config),
               "inspector": "CPU_MOCK_ONLY", "inspected_at": "2020-01-01T00:00:00Z",
               "valid_until": "2999-01-01T00:00:00Z", "checks": dict.fromkeys(CHECKS, True)}
    return config, receipt
def response(stdout="", stderr="", return_code=0, status="Finished", outer="Success"):
    return {"status": outer, "run_result": {"status": status, "return_code": return_code,
            "stdout": stdout, "stderr": stderr}}
class IsolationTests(unittest.TestCase):
    def test_bool_cannot_replace_inspection(self):
        backend = SandboxFusionBackend("http://sandbox:8080", isolation="container", isolation_verified=True)
        self.assertFalse(backend.isolation_verified)
        with self.assertRaises(RuntimeError):
            backend.run_function("def f(): return 1", "f", {}, 2)
    def test_receipt_binds_config_endpoint_limits_and_time(self):
        config, receipt = deployment()
        self.assertEqual(len(validate_deployment(config, receipt)), 64)
        for key, value in (("endpoint", "http://other:8080"), ("max_run_seconds", 60), ("memory_limit_mb", 512)):
            changed = {**config, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_deployment(changed, receipt)
        for key in CHECKS:
            altered = deepcopy(receipt)
            altered["checks"][key] = False
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_deployment(config, altered)
        with self.assertRaises(ValueError):
            validate_deployment(config, receipt, now=datetime(3000, 1, 1, tzinfo=timezone.utc))
    def test_endpoint_credentials_paths_and_resource_ranges_refused(self):
        config, receipt = deployment()
        for endpoint in ("file:///tmp/runner", "http://u:p@host", "http://host/run_code", "http://host?a=1"):
            altered = {**config, "endpoint": endpoint}
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                validate_deployment(altered, receipt)
        for value in (True, 0, 61, float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_deployment({**config, "max_run_seconds": value}, receipt)
    def test_timeouts_and_memory_reach_actual_http_request(self):
        backend = VerifiedSandboxFusion(*deployment())
        with patch.object(backend, "_post", return_value=response()) as post:
            result = backend.run_source("print(1)", 7)
        self.assertTrue(result["run_success"])
        payload, client_timeout = post.call_args.args
        self.assertEqual(payload["run_timeout"], 7)
        self.assertEqual(payload["compile_timeout"], 7)
        self.assertEqual(payload["memory_limit_MB"], 256)
        self.assertEqual(payload["fetch_files"], [])
        self.assertEqual(client_timeout, 19)
    def test_invalid_timeout_refused_before_network(self):
        backend = VerifiedSandboxFusion(*deployment())
        for value in (0, -1, True, float("nan"), 31):
            with patch.object(backend, "_post") as post, self.subTest(value=value), self.assertRaises(ValueError):
                backend.run_source("pass", value)
            post.assert_not_called()
    def test_success_requires_outer_status_and_zero_return_code(self):
        backend = VerifiedSandboxFusion(*deployment())
        for data in (response(return_code=1), response(outer="Failed"), response(status="TimeLimitExceeded"),
                     response(return_code=None), response(return_code=False)):
            with patch.object(backend, "_post", return_value=data), self.subTest(data=data):
                self.assertFalse(backend.run_source("pass", 3)["run_success"])
    def test_warning_stderr_is_not_automatically_a_failure(self):
        backend = VerifiedSandboxFusion(*deployment())
        with patch.object(backend, "_post", return_value=response(stderr="A warning\n")):
            self.assertTrue(backend.run_source("pass", 3)["run_success"])
    def test_output_overflow_does_not_count_as_success(self):
        from control.isolation import SandboxInfrastructureError
        backend = VerifiedSandboxFusion(*deployment())
        with patch.object(backend, "_post", return_value=response(stdout="x" * 4097)), \
                self.assertRaises(SandboxInfrastructureError):
            backend.run_source("pass", 3)
    def test_transport_and_bad_response_abort_without_exposing_secrets(self):
        from control.isolation import SandboxInfrastructureError
        backend = VerifiedSandboxFusion(*deployment())
        with patch.object(backend, "_post", side_effect=OSError("secret_endpoint_credential")), \
                self.assertRaises(SandboxInfrastructureError) as caught:
            backend.run_source("pass", 3)
        self.assertNotIn("secret", str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)
        for invalid in ({"status": "Success"}, None, {"run_result": []}):
            with self.subTest(invalid=invalid), patch.object(backend, "_post", return_value=invalid), \
                    self.assertRaises(SandboxInfrastructureError):
                backend.run_source("pass", 3)
    def test_python_preprocessor_never_executes_and_marks_model_errors(self):
        from control.isolation import ModelToolInputError, prepare_python_source
        self.assertEqual(prepare_python_source("2 + 2"), "print(2 + 2)")
        self.assertEqual(prepare_python_source("print(4)"), "print(4)")
        self.assertIn("raise RuntimeError", prepare_python_source("raise RuntimeError('not here')"))
        for code in (None, "", "def broken(:", "x" * 32769):
            with self.subTest(code=str(code)[:20]), self.assertRaises(ModelToolInputError):
                prepare_python_source(code)
    def test_function_never_executes_source_on_host(self):
        backend = VerifiedSandboxFusion(*deployment())
        source = "raise RuntimeError('MUST_NOT_EXECUTE_LOCALLY')\ndef identity(value: int): return value"
        def mock_http(payload, timeout):
            marker = re.search(r"__SEARL_RESULT_[a-f0-9]+__", payload["code"]).group()
            return response(stdout="debug\n" + marker + "3\n")
        with patch.object(backend, "_post", side_effect=mock_http):
            result = backend.run_function(source, "identity", {"value": 3}, 3)
        self.assertEqual(result["result"], 3)
        self.assertEqual(result["stdout"], "debug")
    def test_missing_or_invalid_markers_fail(self):
        backend = VerifiedSandboxFusion(*deployment())
        def bad_marker(payload, timeout):
            marker = re.search(r"__SEARL_RESULT_[a-f0-9]+__", payload["code"]).group()
            return response(stdout=marker + "NaN\n")
        for fake in (lambda *_: response(stdout="no marker"), bad_marker):
            with patch.object(backend, "_post", side_effect=fake):
                self.assertFalse(backend.run_function("def f(): return 0", "f", {}, 2)["run_success"])
    def test_arguments_are_validated_without_execution(self):
        source = "def identity(value: int, optional='ok', *, mode: str='a'): return value"
        schema = function_schema(source, "identity")
        self.assertEqual(schema["required"], ["value"])
        validate_arguments(source, "identity", {"value": 3})
        for inputs in ({}, {"value": True}, {"value": 1, "extra": 2}, {"value": 1, "mode": 2}):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                validate_arguments(source, "identity", inputs)
        for value in ({"x": float("nan")}, {"x": object()}, {1: "a"}, {"x": "a" * 65537}):
            with self.assertRaises(ValueError):
                finite_inputs(value)
    def test_duplicate_keys_and_nonfinite_json_are_rejected(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                strict_json(text)
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
