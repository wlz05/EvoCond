# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
from argparse import Namespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from control import guard, controller
from control.controller import (CONTEXT, DIM, EDGE_DIM, FORMAT, catalog, context,
    edge_evidence, intervene, observed_state, routes, scores, subgoals, validate)
from control.observation import build_examples, fit
import workflow as workflow
from gate import write_new
from panel import PROJECT, digest
def tool(name, requires, produces, price=1):
    return {"type": "function", "function": {"name": name,
        "description": f"Find a quiet hotel. The output type of this tool is {produces}. It has a cost of {price} units.",
        "parameters": {"type": "object", "required": requires,
                       "properties": {r: {"type": "string"} for r in requires}}}}
def tools():
    return [tool("Decide_Attraction_Preference", ["TimeInfo"], "AttractionPreference"),
            tool("Search_Attraction_Candidates", ["AttractionPreference"], "AttractionCandidate"),
            tool("Attraction_Full_Planning", ["TimeInfo"], "AttractionCandidate", 1.8),
            tool("Select_Final_Attraction", ["AttractionCandidate"], "TravelAttraction")]
def call(name):
    return {"role": "assistant", "tool_calls": [{"name": name, "arguments": {}}]}
def reply(produces):
    return {"role": "tool", "content": f"Get object <Value01234> (Type: {produces}). Cost: 1"}
def messages():
    return [{"role": "user", "content": "Find a quiet hotel at <TimeInfo00001>"}]
def empty_model():
    return {"format": FORMAT, "alignment": [[float(i == j) for j in range(DIM)] for i in range(DIM)],
            "fusion": [0.0]*CONTEXT, "edge_gate": [0.0]*EDGE_DIM,
            "nodes": {n["version"]: n for n in catalog(tools()).values()}, "edges": []}
def episodes():
    result = []
    for i in range(4):
        history, observations = messages(), []
        for name, produced in (("Decide_Attraction_Preference", "AttractionPreference"),
                              ("Search_Attraction_Candidates", "AttractionCandidate"),
                              ("Select_Final_Attraction", "TravelAttraction")):
            response = call(name)
            observations.append({"messages": deepcopy(history), "tools": tools(), "response": response})
            history += [response, reply(produced)]
        result.append({"index": i, "observations": observations,
                       "result": {"evaluation": {"success": True}, "conversation_history": history}})
    return result
class StateGateTests(unittest.TestCase):
    def test_price_change_does_not_change_functional_version(self):
        a = tool("T", ["TimeInfo"], "Candidates", 1)
        b = tool("T", ["TimeInfo"], "Candidates", 10)
        self.assertEqual(catalog([a])["T"]["version"], catalog([b])["T"]["version"])
        b["function"]["parameters"]["required"] = ["Location"]
        self.assertNotEqual(catalog([a])["T"]["version"], catalog([b])["T"]["version"])
    def test_duplicate_tool_names_rejected(self):
        with self.assertRaises(ValueError):
            catalog([tools()[0], tools()[0]])
    def test_success_requires_observed_type_witness(self):
        nodes = catalog(tools())
        h = messages() + [call("Decide_Attraction_Preference"), {"role": "tool", "content": "OK"}]
        self.assertNotIn("attractionpreference", observed_state(h, nodes)["facts"])
        h[-1] = reply("AttractionPreference")
        self.assertIn("attractionpreference", observed_state(h, nodes)["facts"])
    def test_failed_output_never_becomes_fact(self):
        h = messages() + [call("Decide_Attraction_Preference"), {"role": "tool", "content": "[ERROR] Type: AttractionPreference"}]
        self.assertNotIn("attractionpreference", observed_state(h, catalog(tools()))["facts"])
    def test_goal_update_invalidates_previous_outputs_but_keeps_explicitly_unchanged_time(self):
        h = messages() + [call("Decide_Attraction_Preference"), reply("AttractionPreference"),
                          {"role": "user", "content": "New requirements: a different attraction"}]
        s = observed_state(h, catalog(tools()))
        self.assertTrue(s["changed"])
        self.assertNotIn("attractionpreference", s["facts"])
        self.assertIn("timeinfo", s["facts"])
        self.assertEqual(s["last_valid"], "")
        h[-1]["content"] = "Change the travel time to next week"
        self.assertNotIn("timeinfo", observed_state(h, catalog(tools()))["facts"])
    def test_new_goal_outputs_accumulate_only_within_current_epoch(self):
        h = messages() + [{"role": "user", "content": "New requirements at <TimeInfo00002>"},
                          call("Decide_Attraction_Preference"), reply("AttractionPreference")]
        s = observed_state(h, catalog(tools()))
        self.assertIn("attractionpreference", s["facts"])
        self.assertEqual(s["last_valid"], "Decide_Attraction_Preference")
    def test_state_features_change_after_feedback_and_remove_instance_ids(self):
        nodes = catalog(tools())
        s = observed_state(messages(), nodes)
        h = messages(); h[0]["content"] = h[0]["content"].replace("00001", "99999")
        self.assertEqual(context(s), context(observed_state(h, nodes)))
        h += [call("Decide_Attraction_Preference"), reply("AttractionPreference")]
        self.assertNotEqual(context(s), context(observed_state(h, nodes)))
    def test_route_respects_all_inputs_and_cost_window(self):
        ts = tools() + [tool("Impossible_Cheap", ["TimeInfo", "Missing"], "AttractionCandidate", 0)]
        s = observed_state(messages(), catalog(ts))
        paths, _ = routes(s, ts)
        self.assertGreaterEqual(len(paths), 2)
        for path in paths:
            self.assertNotIn("Impossible_Cheap", path)
            self.assertLessEqual(sum(catalog(ts)[n]["cost"] for n in path), 3.3+1e-8)
            graph = subgoals(path, s, catalog(ts))
            self.assertEqual(graph[-1]["goal_type"], "travelattraction")
    def test_subgoal_dependencies_are_explicit(self):
        nodes = catalog(tools())
        s = observed_state(messages(), nodes)
        path = ["Decide_Attraction_Preference", "Search_Attraction_Candidates", "Select_Final_Attraction"]
        g = subgoals(path, s, nodes)
        self.assertEqual(g[1]["depends_on"], ["s0"])
        self.assertEqual(g[2]["depends_on"], ["s1"])
        with self.assertRaises(ValueError):
            subgoals(path[1:], s, nodes)
    def test_live_controller_does_not_invent_tool_or_mutate_input(self):
        h, ts, m = messages(), tools(), empty_model()
        before = deepcopy((h, ts, m))
        revised, exposed, e = intervene(h, ts, m)
        self.assertEqual((h, ts, m), before)
        self.assertEqual(e["action"], "state_plan")
        self.assertTrue(set(e["path"]).issubset(catalog(ts)))
        self.assertEqual(exposed, ts)
        self.assertEqual(len(revised), len(h)+1)
    def test_missing_schema_abstains(self):
        _, _, e = intervene(messages(), [], empty_model())
        self.assertEqual(e["action"], "state_abstain")
    def test_completed_goal_does_not_prompt_repeated_final_tool(self):
        h = episodes()[0]["result"]["conversation_history"]
        revised, _, event = intervene(h, tools(), empty_model())
        self.assertEqual(event["search"]["reason"], "goal_type_already_observed")
        self.assertEqual(revised, h)
    def test_formatting_failure_delegates_to_ll(self):
        h = messages() + [call("Decide_Attraction_Preference"), {"role": "tool", "content": "[ERROR] invalid argument"}]
        _, _, e = intervene(h, tools(), empty_model())
        self.assertEqual(e["recovery"]["scale"], "ll")
    def test_training_uses_verified_steps_not_episode_failure_for_edges(self):
        eps = episodes()
        eps[0]["result"]["evaluation"]["success"] = False
        model, examples, _ = build_examples(eps)
        self.assertNotIn(0, {e["index"] for e in examples})
        self.assertTrue(all(v == 1 for e in model["edges"] for v in e["outcomes"].values()))
    def test_repeated_teacher_episodes_do_not_inflate_task_support(self):
        model, _, _ = build_examples(episodes()*3)
        nodes = catalog(tools())
        evidence = edge_evidence(model, "Decide_Attraction_Preference", "Search_Attraction_Candidates", nodes)
        self.assertEqual(evidence["support"], 4)
        self.assertEqual(evidence["kind"], "exact")
    def test_unknown_tool_is_not_forbidden(self):
        model = empty_model()
        ts = tools() + [tool("New_Attraction_Planning", ["TimeInfo"], "AttractionCandidate", 1.7)]
        s = observed_state(messages(), catalog(ts))
        paths, _ = routes(s, ts)
        self.assertTrue(any("New_Attraction_Planning" in p for p in paths))
        values, _ = scores(model, s, ["New_Attraction_Planning"], catalog(ts), "learned")
        self.assertEqual(len(values), 1)
    def test_actual_alignment_and_gates_are_trained(self):
        m, report = fit(episodes())
        validate(m)
        self.assertEqual(report["trained_parameters"], 1110)
        self.assertLess(report["alignment_loss_after"], report["alignment_loss_before"])
        self.assertLessEqual(report["imitation_loss_after"], report["imitation_loss_before"])
        self.assertGreater(report["fusion_parameter_norm"], 0)
        self.assertFalse(report["qwen_updated"])
    def test_missing_train_schemas_fails_closed(self):
        eps = episodes()
        for e in eps:
            for o in e["observations"]:
                o["tools"] = []
        with self.assertRaises(ValueError):
            build_examples(eps)
    def test_corrupt_alignment_rejected(self):
        m = empty_model(); m["alignment"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            validate(m)
    def test_node_identity_cannot_silently_change_description(self):
        m = empty_model()
        next(iter(m["nodes"].values()))["description"] = "changed tool behavior"
        with self.assertRaises(ValueError):
            validate(m)
    def test_two_independent_queries_are_not_enough_edge_support(self):
        model, _, _ = build_examples(episodes()[:2]*4)
        e = edge_evidence(model, "Decide_Attraction_Preference", "Search_Attraction_Candidates", catalog(tools()))
        self.assertEqual(e["kind"], "unknown")
    def test_changed_tool_version_has_no_exact_old_edge(self):
        model, _, _ = build_examples(episodes())
        ts = tools()
        ts[1]["function"]["description"] += " Requires a different business contract."
        e = edge_evidence(model, "Decide_Attraction_Preference", "Search_Attraction_Candidates", catalog(ts))
        self.assertNotEqual(e["kind"], "exact")
class StateInterventionAblationTests(unittest.TestCase):
    def reference(self, history, exposed):
        return guard.intervene(history, exposed, False, "full", True, None, 0.0, "exact")
    def assert_fallback(self, actual, expected, reason):
        revised, available, event = actual
        self.assertEqual((revised, available), expected[:2])
        self.assertEqual(event["base_event"], expected[2])
        self.assertEqual(event["reason"], reason)
        self.assertFalse(event["actual_intervention"])
        self.assertFalse(event["prompt_changed"])
        self.assertFalse(event["tools_changed"])
        self.assertFalse(event["route_changed"])
        self.assertTrue(event["fallback"])
    @staticmethod
    def prefer_long_route(model, state, names, nodes, policy):
        return [0.0 if name == "Attraction_Full_Planning" else 1.0 for name in names], {}
    def test_baseline_is_exact_original_evocond_and_does_not_read_model(self):
        history, exposed = messages(), tools()
        expected = self.reference(history, exposed)
        with patch.object(guard, "intervene", wraps=guard.intervene) as invoke:
            actual = intervene(history, exposed, {}, experiment="baseline")
        invoke.assert_called_once_with(history, exposed, False, "full", True, None, 0.0, "exact")
        self.assertEqual(actual[:2], expected[:2])
        self.assertEqual(actual[2]["base_event"], expected[2])
        self.assertFalse(actual[2]["actual_intervention"])
        self.assertFalse(actual[2]["fallback"])
        self.assertEqual(expected[2]["action"], "plan")
    def test_selective_equal_scores_return_exact_baseline_not_state_prompt(self):
        history, exposed = messages(), tools()
        with patch.object(controller, "scores", return_value=([0.0], {})):
            actual = intervene(history, exposed, empty_model(), experiment="selective")
        self.assert_fallback(actual, self.reference(history, exposed), "no_distinct_route_clears_margin")
        self.assertNotIn("schema-grounded subgoals", actual[0][-1]["content"])
    def test_selective_margin_is_strict_and_requires_distinct_path(self):
        history, exposed = messages(), tools()
        candidates = [
            {"path": ["Attraction_Full_Planning", "Select_Final_Attraction"], "score": 0.0, "cost": 2.8, "views": []},
            {"path": ["Decide_Attraction_Preference", "Search_Attraction_Candidates", "Select_Final_Attraction"], "score": 0.02, "cost": 3.0, "views": []}]
        with patch.object(controller, "_rank_paths", return_value=candidates):
            actual = intervene(history, exposed, empty_model(), experiment="selective")
        self.assert_fallback(actual, self.reference(history, exposed), "no_distinct_route_clears_margin")
        candidates[1].update(path=candidates[0]["path"], score=1.0)
        with patch.object(controller, "_rank_paths", return_value=candidates):
            actual = intervene(history, exposed, empty_model(), experiment="selective")
        self.assert_fallback(actual, self.reference(history, exposed), "no_distinct_route_clears_margin")
    def test_prompt_only_never_reads_ranking_and_preserves_legacy_cost_prompt(self):
        history, exposed = messages(), tools()
        with patch.object(controller, "scores", return_value=([0.0], {})):
            legacy = intervene(history, exposed, empty_model())
        with patch.object(controller, "scores", side_effect=AssertionError("prompt_only must not score")):
            actual = intervene(history, exposed, {}, experiment="prompt_only")
        self.assertEqual(actual[:2], legacy[:2])
        self.assertEqual(actual[2]["path"], legacy[2]["baseline_path"])
        self.assertFalse(actual[2]["route_changed"])
        self.assertTrue(actual[2]["prompt_changed"])
        self.assertTrue(actual[2]["actual_intervention"])
    def test_prompt_and_selective_use_identical_formatter_with_different_routes(self):
        history, exposed, model = messages(), tools(), empty_model()
        with patch.object(controller, "planning_note", wraps=controller.planning_note) as formatter:
            prompt = intervene(history, exposed, model, experiment="prompt_only")
            with patch.object(controller, "scores", side_effect=self.prefer_long_route):
                selective = intervene(history, exposed, model, experiment="selective")
        self.assertEqual(formatter.call_count, 2)
        self.assertNotEqual(formatter.call_args_list[0].args[0], formatter.call_args_list[1].args[0])
        self.assertEqual(formatter.call_args_list[0].args[2:], formatter.call_args_list[1].args[2:])
        self.assertEqual(prompt[0][:-1], selective[0][:-1])
        self.assertEqual(prompt[1], selective[1])
        self.assertTrue(selective[2]["route_changed"])
        self.assertTrue(selective[2]["actual_intervention"])
        self.assertEqual(selective[2]["margin"], 0.02)
    def test_all_new_modes_preserve_inputs_and_frozen_controller(self):
        for experiment in ("baseline", "prompt_only", "selective"):
            with self.subTest(experiment=experiment):
                history, exposed, model = messages(), tools(), empty_model()
                before = deepcopy((history, exposed, model))
                intervene(history, exposed, model, experiment=experiment)
                self.assertEqual((history, exposed, model), before)
    def test_recovery_delegates_exactly_in_both_experimental_modes(self):
        history = messages() + [call("Decide_Attraction_Preference"),
                                {"role": "tool", "content": "[ERROR] invalid argument"}]
        expected = self.reference(history, tools())
        self.assertEqual(expected[2]["scale"], "ll")
        for experiment in ("prompt_only", "selective"):
            with patch.object(controller, "scores", side_effect=AssertionError("must preserve repair")):
                actual = intervene(history, tools(), {}, experiment=experiment)
            self.assert_fallback(actual, expected, "baseline_recovery")
    def test_unknown_schema_and_observed_goal_use_exact_reference(self):
        for history, exposed in ((messages(), []), (episodes()[0]["result"]["conversation_history"], tools())):
            for experiment in ("prompt_only", "selective"):
                actual = intervene(history, exposed, empty_model(), experiment=experiment)
                self.assert_fallback(actual, self.reference(history, exposed), "no_type_feasible_candidates")
    def test_experimental_failure_is_observable_and_retains_exact_baseline(self):
        history, exposed = messages(), tools()
        for experiment in ("prompt_only", "selective"):
            with patch.object(controller, "routes", side_effect=RuntimeError("planner unavailable")):
                actual = intervene(history, exposed, empty_model(), experiment=experiment)
            self.assert_fallback(actual, self.reference(history, exposed), "controller_error")
            self.assertEqual(actual[2]["controller_error"]["type"], "RuntimeError")
    def test_nonfinite_ranking_falls_back_without_recording_an_intervention(self):
        with patch.object(controller, "scores", return_value=([float("nan")], {})):
            actual = intervene(messages(), tools(), empty_model(), experiment="selective")
        self.assert_fallback(actual, self.reference(messages(), tools()), "controller_error")
    def test_reference_failure_is_not_swallowed_as_successful_fallback(self):
        with patch.object(guard, "intervene", side_effect=ValueError("broken reference")):
            with self.assertRaisesRegex(ValueError, "broken reference"):
                intervene(messages(), tools(), empty_model(), experiment="selective")
    def test_unknown_policies_and_experiments_fail_before_either_controller(self):
        for policy, experiment in (("unknown", "baseline"), ("learned", "unknown")):
            with patch.object(guard, "intervene", side_effect=AssertionError("must validate first")):
                with self.assertRaises(ValueError):
                    intervene(messages(), tools(), empty_model(), policy, experiment=experiment)
class StateWorkflowTests(unittest.TestCase):
    def fixture(self, root):
        records = [{"task": "dining", "user_requirements": f"distinct requirement word{chr(97+i)}"} for i in range(12)]
        write_new(root / "queries.json", records)
        write_new(root / "model/config.json", {"model_type": "fake"})
        write_new(root / "memory.json", {"metadata": {"source": "costbench_traces_v1", "train_indices": [0], "query_sha256": digest(root / "queries.json")}})
        old = {"query_sha256": digest(root / "queries.json"), "memory_sha256": digest(root / "memory.json"),
               "model_config_sha256": digest(root / "model/config.json"), "tokens": 2048, "refinement": 2,
               "splits": {k: {"indices": v} for k, v in (("train", [1, 2]), ("validation", [3, 4]), ("test", [5, 6]))}}
        write_new(root / "previous/manifest.json", old)
        return Namespace(root=str(root / "new"), previous=str(root / "previous"), queries=str(root / "queries.json"),
                         memory=str(root / "memory.json"), model=str(root / "model"), exclude_root=str(root),
                         test_size=2, tokens=2048, refinement=2, port=8000, seed=1)
    def test_manifest_readonly_then_exact_resume_with_unexposed_test(self):
        with TemporaryDirectory(dir=PROJECT) as temp:
            root = Path(temp)
            args = self.fixture(root)
            manifest = workflow.make_manifest(args)
            self.assertFalse(Path(args.root).exists())
            self.assertTrue(set(manifest["splits"]["test"]["indices"]).isdisjoint(range(7)))
            write_new(Path(args.root) / "manifest.json", manifest)
            self.assertEqual(workflow.make_manifest(args), manifest)
            with patch.object(workflow, "source_hashes", return_value={"changed.py": "new"}):
                with self.assertRaises(ValueError):
                    workflow.make_manifest(args)
    def test_manifest_rejects_different_previous_protocol(self):
        with TemporaryDirectory(dir=PROJECT) as temp:
            args = self.fixture(Path(temp))
            with patch.object(workflow, "digest", return_value="changed"):
                with self.assertRaises(ValueError):
                    workflow.make_manifest(args)
    def test_training_resume_checks_sources_and_never_refits(self):
        with TemporaryDirectory(dir=PROJECT) as temp:
            root = Path(temp)
            manifest = {"seed": 1, "query_sha256": "q", "model_config_sha256": "m", "memory_train_indices": [9],
                        "splits": {"train": {"indices": [0, 1, 2, 3]}, "validation": {"indices": [4]}, "test": {"indices": [5]}}}
            write_new(root / "manifest.json", manifest)
            train_hashes = {"episode": ["result-hash", "trace-hash"]}
            with patch.object(workflow, "training_episodes", return_value=(episodes(), train_hashes)), redirect_stdout(StringIO()):
                first = workflow.train(root, manifest)
                with patch.object(workflow, "fit", side_effect=AssertionError("must not refit")):
                    self.assertEqual(workflow.train(root, manifest), first)
            with patch.object(workflow, "training_episodes", return_value=(episodes(), {})):
                with self.assertRaises(ValueError):
                    workflow.train(root, manifest)
    def test_equal_validation_is_not_promoted(self):
        with TemporaryDirectory(dir=PROJECT) as temp:
            root = Path(temp)
            write_new(root / 'controller.json', empty_model())
            manifest = {"splits": {"validation": {"indices": [1, 2]}}}
            for arm in workflow.ARMS:
                for mode in workflow.MODES:
                    for i in (1, 2):
                        folder = root / "validation" / arm / mode / f"q{i}"
                        data = {"results": [{"query_id": f"id{i}", "tool_creation_seed": 0, "conversation_history": messages(),
                                             "evaluation": {"success": True}, "total_steps": 3}],
                                "stats": {"scores": {"cost_metrics": {"total_agent_cost": 10}}}}
                        write_new(folder / "results_test.json", data)
                        write_new(folder / "episode.json", {"result_sha256": digest(folder / "results_test.json"),
                                  **{k: 1 for k in ("elapsed_seconds", "prompt_tokens", "completion_tokens", "controller_seconds", "state_plans", "reranks")}})
            with redirect_stdout(StringIO()):
                report = workflow.summarize(root, manifest, "validation")
            self.assertFalse(report["promoted"])
            self.assertFalse((root / "test").exists())
    def test_test_stage_requires_same_validated_model(self):
        with TemporaryDirectory(dir=PROJECT) as temp:
            root = Path(temp)
            write_new(root / 'controller.json', empty_model())
            write_new(root / "validation_report.json", {"split": "validation", "controller_sha256": "different", "promoted": True})
            with patch.object(workflow, "train"):
                with self.assertRaises(ValueError):
                    workflow.require_validated_controller(root, {})
    def test_fresh_pool_reports_exhausted_domain_without_reusing_old_test(self):
        records = [{"task": "location", "user_requirements": "location old unique"}]
        records += [{"task": "dining", "user_requirements": f"dining unique word{chr(97+i)}"} for i in range(8)]
        selected, counts = workflow.fresh_sample(records, {0, 1}, 4, 1)
        self.assertFalse(set(selected) & {0, 1})
        self.assertEqual(counts["location"], 0)
        self.assertEqual(len(selected), 4)
    def test_fresh_pool_excludes_text_duplicates_and_refuses_shortage(self):
        records = [{"task": "location", "user_requirements": "same <TimeInfo00001>"},
                   {"task": "location", "user_requirements": "same <TimeInfo00002>"}]
        with self.assertRaises(ValueError):
            workflow.fresh_sample(records, {0}, 1, 1)
    def test_completion_needs_result_and_trace_sidecar(self):
        with TemporaryDirectory(dir=PROJECT) as temp:
            root = Path(temp)
            folder = root / "train/evocond/none/q1"
            result = folder / "results_test.json"
            write_new(result, {"results": [{"evaluation": {"success": True}, "total_steps": 3}],
                               "stats": {"scores": {"cost_metrics": {"total_agent_cost": 4}}}})
            self.assertIsNone(workflow.data_at(root, "train", "evocond", "none", 1))
            write_new(folder / "episode.json", {"result_sha256": digest(result)})
            self.assertIsNotNone(workflow.data_at(root, "train", "evocond", "none", 1))
    def test_runner_lock_blocks_duplicate_and_cleans_only_own_lock(self):
        with TemporaryDirectory(dir=PROJECT) as temp:
            root = Path(temp)
            with workflow.runner_lock(root):
                self.assertTrue((root / "runner.lock").exists())
                with self.assertRaises(RuntimeError):
                    with workflow.runner_lock(root):
                        pass
            self.assertFalse((root / "runner.lock").exists())
    def test_current_lock_owner_is_probed_without_a_signal(self):
        with patch.object(workflow.os, "kill", side_effect=AssertionError("Do not signal a live lock owner")):
            self.assertTrue(workflow.process_exists(workflow.os.getpid()))
    def test_invalid_lock_pid_is_not_treated_as_stale(self):
        for pid in (0, -1, True, "12", None):
            with self.assertRaises(ValueError):
                workflow.process_exists(pid)
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
