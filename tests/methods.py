# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'CPU method-contract regressions; not observed benchmark results.'
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from control import dynamic
from control.runtime import Runtime, execution_context_fingerprint
from control.evidence import REFERENCE, call_record, resolve_arguments, transition_feedback
from tests import runtimes as runtime_fixture
from tests.continuations import helper_functions
import bootstrap as bootstrap
PLAN = runtime_fixture.PLAN
class MethodContractAudit(unittest.TestCase):
    def setUp(self):
        self.fixture = runtime_fixture.RuntimeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
    def teacher(self):
        return self.fixture.teacher_runtime()
    def add_pair(self, runtime):
        versions = []
        for name in ("source", "target"):
            spec = self.fixture.tool(name)
            created = runtime.dispatcher.create(spec, {"x": 1, "y": 2}, task_id="setup")
            versions.append(created["tool_version"])
        return versions
    def test_alpha_zero_still_retrieves_memory_not_no_memory(self):
        runtime = self.teacher()
        self.add_pair(runtime)
        identity = runtime.begin("sum", PLAN, split="validation")
        context = runtime.context(identity)
        context.update(collection_branch=True, next_alpha=0.0)
        selected = runtime.select(identity)
        self.assertEqual(selected["teacher"]["structure_weight"], 0.0)
        self.assertTrue(any(tool.get("tool_version") for tool in selected["schemas"]))
    def test_traced_dataflow_does_not_measure_counterfactual_benefit(self):
        source = call_record("source-call", "source", "a"*64, True, {}, 3)
        args, bindings = resolve_arguments({"x": {REFERENCE: {"call_id": "source-call", "path": ""}}}, [source])
        # A constant-output target can receive x without needing x. Dataflow is visible; necessity or.
        target = call_record("target-call", "constant_target", "b"*64, True, args, 42, bindings)
        self.assertEqual(transition_feedback(source, target), "verified")
        self.assertNotIn("causal_benefit", target)
    def test_equal_values_do_not_create_edge_evidence(self):
        source = call_record("source-call", "source", "a"*64, True, {}, 3)
        target = call_record("target-call", "target", "b"*64, True, {"x": 3}, 42)
        self.assertEqual(transition_feedback(source, target), "unknown")
    def test_later_same_question_evidence_can_refresh_without_inflating_support(self):
        runtime = self.teacher()
        source_version, target_version = self.add_pair(runtime)
        question = "same independent training question"
        first = runtime.begin(question, PLAN, split="train")
        runtime.observe(first, "source", "first source", success=True, version=source_version,
                        arguments={"x": 1, "y": 2}, execution_result=3)
        runtime.observe(first, "target", "no explicit dependency", success=True, version=target_version,
                        arguments={"x": 3, "y": 0}, execution_result=3)
        runtime.finish(first, task_success=True)
        runtime.set_policy_step(1)
        later = runtime.begin(question, PLAN, split="train")
        source = runtime.observe(later, "source", "new source", success=True, version=source_version,
                                 arguments={"x": 1, "y": 2}, execution_result=3)
        args, bindings = resolve_arguments({"x": {REFERENCE: {"call_id": source["call_id"], "path": ""}}, "y": 0},
                                          runtime.context(later)["calls"])
        runtime.observe(later, "target", "verified dependency", success=True, version=target_version,
                        arguments=args, execution_result=3, bindings=bindings)
        runtime.finish(later, task_success=True)
        observations = runtime.dispatcher.registry["edge_memory"]["edges"][source_version+"->"+target_version]["observations"]
        self.assertEqual(len(observations), 1)
        self.assertEqual(next(iter(observations.values()))["direct_feedback"], "verified")
    def test_step_zero_collection_and_deployment_have_same_nonretrieval_controls(self):
        budget = {"max_turns": 8, "max_context": 8192, "max_prompt": 4096, "max_response": 2048}
        self.fixture.config.update(sample_tool_turns=True, rollout_contract=budget, official_source_sha256="c"*64)
        deployed = self.teacher()
        self.add_pair(deployed)
        root = Path(self.fixture.root)/"collection-audit"
        protocol = {"experiment": {"config": {"run_id": "audit", "budget": budget, "encoder": deployed.config["encoder_path"]},
                    "fingerprints": {"model": {"sha256": deployed.config["policy_hash"]}, "official_source": {"sha256": "c"*64}}},
                    "input_config": {"encoder_id": deployed.config["encoder_id"]}}
        config = bootstrap.runtime_config(protocol, root, "collect")
        config["alignment_path"] = deployed.config["alignment_path"]
        dispatcher = runtime_fixture.dispatcher_class()(config["registry_path"], runtime_fixture.Backend(),
                                                        initial_registry=deployed.dispatcher.registry)
        collecting = Runtime(config, dispatcher, encoder=deployed.encoder)
        prefix_id = collecting.begin("sum", PLAN, split="train")
        snapshot = collecting.capture_continuation(prefix_id)
        branch_id = collecting.resume_continuation(snapshot, 0.0, 42)
        deployed_id = deployed.begin("sum", PLAN, split="train")
        branch = collecting.select(branch_id)
        live = deployed.select(deployed_id)
        self.assertEqual({tool["name"] for tool in branch["schemas"] if not tool.get("tool_version")},
                         {tool["name"] for tool in live["schemas"] if not tool.get("tool_version")})
        self.assertEqual(collecting.config["value_contracts"], deployed.config["value_contracts"])
        self.assertEqual(execution_context_fingerprint(collecting.config), execution_context_fingerprint(deployed.config))
    def test_fitted_teacher_rejects_different_control_context(self):
        deployed = self.teacher()
        config = {**deepcopy(deployed.config), "value_contracts": False}
        with self.assertRaisesRegex(ValueError, "execution context"):
            Runtime(config, deployed.dispatcher, encoder=deployed.encoder)
    def test_paired_sampling_flag_survives_appending_observation(self):
        helper = helper_functions()["append_observations"]
        tokenizer = lambda values, **kwargs: SimpleNamespace(input_ids=[[8] for _ in values])
        state = SimpleNamespace(tokenizer=tokenizer, strict_runtime=True, max_tool_response_length=32,
             loop_responses_token=[[[1], [2]]], tool_use=[[]], mcp_ids=[[]], context_budget=128,
             config_prompt_length=32, pad_token_id=0, loop_cnt=1,
             meta_info={"do_sample": True, "sample_tool_turns": True, "continuation_seed": 42})
        prompt = helper(state, ["observed"], [False], [["tool"]], [0], [0])
        self.assertTrue(prompt.meta_info["do_sample"])
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
