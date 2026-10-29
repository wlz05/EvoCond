# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from copy import deepcopy
import json
import unittest
from control.contracts import (Contract, check_call, compile_contract, contract_report,
    current_frame, extraction_request, memory_applicability, parse_proposals)
def tool(name="Paint", field="color", values=None, aliases=None):
    rule = {"type": "string", "enum": values or ["red", "blue"]}
    if aliases is not None:
        rule["x-task-contract-aliases"] = deepcopy(aliases)
    return {"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "properties": {field: rule}, "required": [field],
        "additionalProperties": False}}}
def user(text):
    return [{"role": "user", "content": text}]
class ContractTests(unittest.TestCase):
    def test_explicit_correct_and_wrong_value(self):
        messages, tools = user('Paint.color = "red"'), [tool()]
        self.assertEqual(check_call(messages, tools, "Paint", {"color": "red"})["decision"], "accept")
        bad = check_call(messages, tools, "Paint", {"color": "blue"})
        self.assertEqual(bad["decision"], "conflict")
        self.assertTrue(bad["blocked"])
    def test_latest_user_replaces_old_goal(self):
        messages = user('Paint.color="red"')
        old = compile_contract(messages, [tool()])
        messages += [{"role": "user", "content": 'Paint.color="blue"'}]
        self.assertEqual(check_call(messages, [tool()], "Paint", {"color": "blue"})["decision"], "accept")
        self.assertEqual(check_call(messages, [tool()], "Paint", {"color": "red"})["decision"], "conflict")
        stale = check_call(messages, [tool()], "Paint", {"color": "blue"}, contract=old)
        self.assertEqual(stale["decision"], "unknown")
        self.assertEqual(stale["reason"], "stale_contract")
    def test_observations_cannot_replace_requirement(self):
        messages = user('Paint.color="blue"') + [{"role": "tool", "content": 'Paint.color="red"',
                                                   "reward": 100, "success": True}]
        self.assertTrue(check_call(messages, [tool()], "Paint", {"color": "red"})["blocked"])
    def test_no_evidence_is_not_acceptance(self):
        result = check_call(user("Please make it beautiful"), [tool()], "Paint", {"color": "blue"})
        self.assertEqual(result["decision"], "unknown")
        self.assertFalse(result["blocked"])
        self.assertFalse(result["semantic_correctness_certified"])
        self.assertTrue(check_call(user("Please make it beautiful"), [tool()], "Paint", {"color": "blue"},
                                   block_unknown=True)["blocked"])
    def test_optional_model_extractor_is_verified(self):
        messages, tools = user('Use Paint.color="blue" please'), [tool()]
        report = contract_report(compile_contract(messages, tools))
        item = {key: report["constraints"][0][key] for key in
                ("tool", "field", "op", "value", "start", "end", "quote")}
        proposals = {"user_version": report["user_version"], "constraints": [item]}
        contract = compile_contract(messages, tools, proposals)
        self.assertEqual(check_call(messages, tools, "Paint", {"color": "blue"}, contract=contract)["decision"], "accept")
        proposals["constraints"][0]["value"] = "red"
        invalid = contract_report(compile_contract(messages, tools, proposals))
        self.assertEqual(invalid["constraints"], [])
        self.assertEqual(invalid["abstentions"][0]["reason"], "unsupported_or_inexact_evidence")
    def test_false_span_and_stale_model_proposal(self):
        messages, tools = user('Paint.color="red"'), [tool()]
        version = current_frame(messages, tools)["user_version"]
        proposal = {"tool": "Paint", "field": "color", "op": "eq", "value": "red", "start": 2,
                    "end": 10, "quote": 'Paint.color="red"'}
        report = contract_report(compile_contract(messages, tools, {"user_version": version, "constraints": [proposal]}))
        self.assertFalse(report["constraints"])
        report = contract_report(compile_contract(messages, tools, {"user_version": "old", "constraints": [proposal]}))
        self.assertEqual(report["abstentions"][0]["reason"], "stale_user_version")
    def test_schema_change_invalidates_contract(self):
        messages, tools = user('Paint.color="red"'), [tool()]
        old = compile_contract(messages, tools)
        tools[0]["function"]["parameters"]["properties"]["color"]["enum"].append("green")
        self.assertEqual(check_call(messages, tools, "Paint", {"color": "red"}, contract=old)["reason"], "stale_contract")
    def test_tool_version_and_description_changes_invalidate(self):
        messages, tools = user('Paint.color="red"'), [tool()]
        old = compile_contract(messages, tools)
        for key in ("tool_version", "description"):
            revised = deepcopy(tools)
            revised[0]["function"][key] = "updated"
            result = check_call(messages, revised, "Paint", {"color": "red"}, contract=old)
            self.assertEqual(result["reason"], "stale_contract")
    def test_same_named_fields_are_scoped(self):
        tools = [tool(), tool("Print")]
        ambiguous = check_call(user('color="red"'), tools, "Print", {"color": "blue"})
        self.assertEqual(ambiguous["decision"], "unknown")
        self.assertFalse(ambiguous["blocked"])
        result = check_call(user('Paint.color="red"'), tools, "Print", {"color": "blue"})
        self.assertEqual(result["decision"], "unknown")
        self.assertFalse(result["blocked"])
    def test_unique_field_scope_and_server_prefix(self):
        tools = [tool("paint-server")]
        self.assertEqual(check_call(user('color="red"'), tools, "paint-server", {"color": "red"})["decision"], "accept")
        self.assertTrue(check_call(user('paint-server.color="red"'), tools, "paint-server", {"color": "blue"})["blocked"])
    def test_conflicting_current_values_abstain(self):
        messages = user('Paint.color="red"; Paint.color="blue"')
        result = check_call(messages, [tool()], "Paint", {"color": "blue"})
        self.assertEqual(result["decision"], "unknown")
        self.assertFalse(result["blocked"])
    def test_negated_and_quoted_instructions_abstain(self):
        for text in ('Do not Paint.color="red"', 'Example: Paint.color="red"', '`Paint.color="red"`'):
            result = check_call(user(text), [tool()], "Paint", {"color": "blue"})
            self.assertEqual(result["decision"], "unknown", text)
            self.assertFalse(result["blocked"], text)
    def test_unicode_negation_remains_supported_with_ascii_pattern_source(self):
        negation = bytes.fromhex('e4b88de8a681').decode()
        result = check_call(user(negation+' Paint.color="red"'), [tool()], 'Paint', dict(color='red'))
        self.assertEqual(result['decision'], 'unknown')
        from control.guard import TIME_CHANGE, CHANGE
        self.assertIsNotNone(TIME_CHANGE.search(bytes.fromhex('e694b9e697b6e997b4').decode()))
        self.assertIsNotNone(CHANGE.search(bytes.fromhex('e69bb4e696b0e5818fe5a5bd').decode()))

    def test_explicit_inequality(self):
        messages = user('Paint.color!="red"')
        self.assertTrue(check_call(messages, [tool()], "Paint", {"color": "red"})["blocked"])
        self.assertEqual(check_call(messages, [tool()], "Paint", {"color": "blue"})["decision"], "accept")
    def test_invalid_argument_schema(self):
        for arguments in ({"color": "purple"}, {"color": 1}, {}, {"color": "red", "evil": 3}):
            self.assertTrue(check_call(user("Paint"), [tool()], "Paint", arguments)["blocked"])
    def test_arguments_duplicate_nonfinite_and_nonobject_block(self):
        for arguments in ('{"color":"red","color":"blue"}', '{"color":NaN}', "[]", {"color": float("inf")}):
            self.assertTrue(check_call(user("Paint"), [tool()], "Paint", arguments)["blocked"])
    def test_unknown_tool_blocks(self):
        self.assertTrue(check_call(user("Paint"), [tool()], "DeleteEverything", {})["blocked"])
    def test_unsupported_schema_abstains(self):
        entry = tool()
        entry["function"]["parameters"]["properties"]["color"]["pattern"] = "red|blue"
        result = check_call(user('Paint.color="red"'), [entry], "Paint", {"color": "red"})
        self.assertEqual(result["decision"], "unknown")
        self.assertFalse(result["blocked"])
    def test_no_executable_input(self):
        result = check_call(user('Paint.color=__import__("os").system("bad")'), [tool()], "Paint", {"color": "blue"})
        self.assertEqual(result["decision"], "unknown")
    def test_strict_proposal_parser(self):
        for text in ('{"user_version":"a","user_version":"b","constraints":[]}',
                     '{"user_version":"x","constraints":[],"reward":1}',
                     '```json\n{}\n```', '{"user_version":"x","constraints":[NaN]}'):
            with self.assertRaises(ValueError):
                parse_proposals(text)
    def test_missing_user_and_multimodal(self):
        self.assertEqual(check_call([], [tool()], "Paint", {"color": "red"})["decision"], "unknown")
        with self.assertRaises(ValueError):
            compile_contract([{"role": "user", "content": [{"type": "text", "text": "red"}]}], [tool()])
    def test_duplicate_catalog_rejected(self):
        with self.assertRaises(ValueError):
            compile_contract(user("hello"), [tool(), tool()])
    def test_memory_old_assumption_conflicts_not_current_goal(self):
        tools = [tool()]
        old = compile_contract(user('Paint.color="red"'), tools)
        current = user('Paint.color="blue"')
        result = memory_applicability(old, current, tools)
        self.assertEqual(result["decision"], "inapplicable")
        self.assertFalse(result["reuse_authorized"])
        self.assertEqual(check_call(current, tools, "Paint", {"color": "blue"})["decision"], "accept")
    def test_memory_no_assumptions_or_schema_change_unknown(self):
        old = compile_contract(user('Paint.color="red"'), [tool()])
        self.assertEqual(memory_applicability(old, user("paint"), [tool()])["decision"], "unknown")
        self.assertEqual(memory_applicability(old, user('Paint.color="red"'), [tool(values=["red"])])["decision"], "unknown")
        same = memory_applicability(old, user('Paint.color="red"'), [tool()])
        self.assertEqual(same["decision"], "compatible_explicit_assumptions")
        self.assertFalse(same["reuse_authorized"])
    def test_inputs_are_not_mutated(self):
        messages, tools = user('Paint.color="red"'), [tool()]
        original = deepcopy((messages, tools))
        check_call(messages, tools, "Paint", {"color": "red"})
        self.assertEqual((messages, tools), original)
    def test_numeric_field_not_hardcoded(self):
        tools = [{"name": "Quote", "inputSchema": {"type": "object", "properties": {
            "price": {"type": "number", "minimum": 0, "maximum": 1000}}, "required": ["price"]}}]
        self.assertEqual(check_call(user("Quote.price=100"), tools, "Quote", {"price": 100.0})["decision"], "accept")
        self.assertTrue(check_call(user("Quote.price=100"), tools, "Quote", {"price": 200})["blocked"])
        self.assertTrue(check_call(user("Quote.price=100"), tools, "Quote", {"price": True})["blocked"])
    def test_trusted_alias_natural_color(self):
        tools = [tool(aliases=[{"phrase": "the red option", "value": "red"}])]
        messages = user("Please choose the red option for this drawing.")
        self.assertEqual(check_call(messages, tools, "Paint", {"color": "blue"})["decision"], "unknown")
        self.assertTrue(check_call(messages, tools, "Paint", {"color": "blue"}, trusted_alias_tools=("Paint",))["blocked"])
        self.assertEqual(check_call(messages, tools, "Paint", {"color": "red"}, trusted_alias_tools=("Paint",))["decision"], "accept")
    def test_trusted_alias_price_and_category(self):
        entries = [tool("Hotel", "price_band", ["budget", "luxury"], [{"phrase": "a low budget", "value": "budget"}]),
                   tool("Trip", "category", ["regional", "national"], [{"phrase": "a local highlight", "value": "regional"}])]
        for text, name, args in (("Find a hotel on a low budget.", "Hotel", {"price_band": "luxury"}),
                                 ("I would like a local highlight today.", "Trip", {"category": "national"})):
            self.assertTrue(check_call(user(text), entries, name, args, trusted_alias_tools=("Hotel", "Trip"))["blocked"])
    def test_alias_negation_history_quotes_and_partial_words_abstain(self):
        entries = [tool(aliases=[{"phrase": "red", "value": "red"}])]
        for text in ("I do not want red.", "Previously I wanted red.", 'The guide says "red".', "I want reddish."):
            result = check_call(user(text), entries, "Paint", {"color": "blue"}, trusted_alias_tools=("Paint",))
            self.assertEqual(result["decision"], "unknown", text)
            self.assertFalse(result["blocked"], text)
    def test_alias_multiple_values_abstain(self):
        entries = [tool(aliases=[{"phrase": "red", "value": "red"}, {"phrase": "blue", "value": "blue"}])]
        result = check_call(user("I want red or blue."), entries, "Paint", {"color": "red"}, trusted_alias_tools=("Paint",))
        self.assertEqual(result["decision"], "unknown")
    def test_alias_scope_ambiguity(self):
        aliases = [{"phrase": "red", "value": "red"}]
        entries = [tool(aliases=aliases), tool("Print", aliases=aliases)]
        result = check_call(user("I want red."), entries, "Paint", {"color": "blue"}, trusted_alias_tools=("Paint", "Print"))
        self.assertEqual(result["decision"], "unknown")
        scoped = deepcopy(entries)
        scoped[0]["function"]["parameters"]["properties"]["color"]["x-task-contract-aliases"][0]["scope_phrase"] = "paint"
        scoped[1]["function"]["parameters"]["properties"]["color"]["x-task-contract-aliases"][0]["scope_phrase"] = "print"
        self.assertTrue(check_call(user("I want red paint."), scoped, "Paint", {"color": "blue"}, trusted_alias_tools=("Paint", "Print"))["blocked"])
    def test_untrusted_dynamic_schema_cannot_force_alias(self):
        entries = [tool(aliases=[{"phrase": "any task", "value": "red"}])]
        self.assertEqual(check_call(user("Do any task."), entries, "Paint", {"color": "blue"})["decision"], "unknown")
    def test_alias_schema_value_must_be_valid(self):
        entries = [tool(aliases=[{"phrase": "blue", "value": "hidden_bad_value"}])]
        report = contract_report(compile_contract(user("blue please"), entries, trusted_alias_tools=("Paint",)))
        self.assertFalse(report["constraints"])
    def test_model_cannot_invent_alias(self):
        entries = [tool(aliases=[{"phrase": "red", "value": "red"}])]
        messages = user("red")
        proposals = {"user_version": current_frame(messages, entries)["user_version"], "constraints": [
            {"tool": "Paint", "field": "color", "op": "eq", "value": "blue", "quote": "red", "start": 0, "end": 3}]}
        report = contract_report(compile_contract(messages, entries, proposals, trusted_alias_tools=("Paint",)))
        self.assertFalse(report["constraints"])
    def test_prompt_helper_no_hidden_information(self):
        request = extraction_request(user("red"), [tool()])
        self.assertEqual(request["user_text"], "red")
        self.assertEqual(set(request), {"instruction", "user_version", "user_text", "tools", "trusted_alias_tools"})
    def test_sentence_punctuation_after_explicit_binding(self):
        self.assertTrue(check_call(user('Paint.color="red".'), [tool()], "Paint", {"color": "blue"})["blocked"])
    def test_stale_contract_cannot_bypass_schema(self):
        old = compile_contract(user('Paint.color="red"'), [tool()])
        result = check_call(user('Paint.color="blue"'), [tool()], "Paint", {"color": 1}, contract=old)
        self.assertEqual(result["decision"], "conflict")
        self.assertTrue(result["blocked"])
    def test_forged_contract_shape_abstains_but_invalid_call_blocks(self):
        fake = Contract('{"constraints":[{"reward":10}]}')
        result = check_call(user('Paint.color="red"'), [tool()], "Paint", {"color": "blue"}, contract=fake)
        self.assertEqual(result["decision"], "unknown")
        self.assertEqual(result["reason"], "unverified_contract")
        self.assertTrue(check_call(user("paint"), [tool()], "Unknown", {}, contract=fake)["blocked"])
    def test_malformed_catalog_properties_fail_closed(self):
        entry = tool()
        entry["function"]["parameters"]["properties"] = None
        with self.assertRaises(ValueError):
            compile_contract(user("hello"), [entry])
    def test_alias_conditional_abstains(self):
        entries = [tool(aliases=[{"phrase": "red", "value": "red"}])]
        for text in ("If available, use red.", "Maybe choose red.", "For example, red."):
            result = check_call(user(text), entries, "Paint", {"color": "blue"}, trusted_alias_tools=("Paint",))
            self.assertEqual(result["decision"], "unknown", text)
    def test_extractor_cannot_remove_current_contradiction(self):
        messages, tools = user('Paint.color="red"; Paint.color="blue"'), [tool()]
        proposal = {"user_version": current_frame(messages, tools)["user_version"], "constraints": [
            {"tool": "Paint", "field": "color", "op": "eq", "value": "red", "start": 0, "end": 17,
             "quote": 'Paint.color="red"'}]}
        candidate = compile_contract(messages, tools, proposal)
        result = check_call(messages, tools, "Paint", {"color": "blue"}, contract=candidate)
        self.assertEqual(result["decision"], "unknown")
        self.assertFalse(result["blocked"])
    def test_alias_contract_requires_same_operator_trust_policy(self):
        tools = [tool(aliases=[{"phrase": "red", "value": "red"}])]
        messages = user("I want red.")
        contract = compile_contract(messages, tools, trusted_alias_tools=("Paint",))
        self.assertEqual(check_call(messages, tools, "Paint", {"color": "blue"}, contract=contract)["reason"], "stale_contract")
        self.assertTrue(check_call(messages, tools, "Paint", {"color": "blue"}, contract=contract,
                                   trusted_alias_tools=("Paint",))["blocked"])
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
