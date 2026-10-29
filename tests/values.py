# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import json
import unittest
from control.value import interpret, latest_user, proposal, review_prompt
def visible_tool():
    return {"type": "function", "function": {
        "name": "Shopping_Full_Planning_to_Step2",
        "description": "A visible shopping planner",
        "parameters": {"type": "object", "properties": {
            "ShoppingTier": {"type": "string", "enum": ["affordable_price", "mid_range_brand"]},
            "ShoppingStyle": {"type": "string", "enum": ["brand_and_quality_priority", "uniqueness_and_discovery_priority"]},
            "TimeInfo": {"type": "string"}}}}}
class ValueAuditTests(unittest.TestCase):
    def test_only_visible_preference_fields_are_audited(self):
        call = {"function": {"name": "Shopping_Full_Planning_to_Step2",
                "arguments": json.dumps({"ShoppingTier": "mid_range_brand", "TimeInfo": "<TimeInfo01679>"})}}
        item = proposal(call, [visible_tool()])
        self.assertEqual(list(item["fields"]), ["ShoppingTier"])
        self.assertEqual(item["fields"]["ShoppingTier"]["schema"]["enum"][0], "affordable_price")
        self.assertIsNone(proposal(call, []))
    def test_review_uses_latest_user_not_stale_preference(self):
        messages = [{"role": "user", "content": "I want something expensive."},
                    {"role": "tool", "content": "Old preference."},
                    {"role": "user", "content": "New requirement: affordable brands."}]
        item = proposal({"name": "Shopping_Full_Planning_to_Step2",
                         "arguments": {"ShoppingTier": "mid_range_brand"}}, [visible_tool()])
        user = latest_user(messages)
        prompt = review_prompt(user, item)
        self.assertIn("affordable brands", prompt[1]["content"])
        self.assertNotIn("something expensive", prompt[1]["content"])
    def test_rejection_needs_verbatim_current_user_evidence(self):
        user = "I want affordable brands without breaking the bank."
        item = proposal({"name": "Shopping_Full_Planning_to_Step2",
                         "arguments": {"ShoppingTier": "mid_range_brand"}}, [visible_tool()])
        supported = json.dumps({"decision": "reject", "mismatches": [
            {"field": "ShoppingTier", "evidence": "without breaking the bank",
             "reason": "The proposed tier conflicts with affordability."}]})
        self.assertEqual(interpret(supported, user, item)["decision"], "reject")
        invented = supported.replace("without breaking the bank", "world-renowned luxury stores")
        self.assertEqual(interpret(invented, user, item)["decision"], "uncertain")
        other = supported.replace("ShoppingTier", "TravelBudget")
        self.assertEqual(interpret(other, user, item)["decision"], "uncertain")
    def test_unparseable_and_ambiguous_audit_do_not_reject(self):
        item = proposal({"name": "Shopping_Full_Planning_to_Step2",
                         "arguments": {"ShoppingTier": "mid_range_brand"}}, [visible_tool()])
        self.assertEqual(interpret("not json", "affordable brands", item)["decision"], "uncertain")
        self.assertEqual(interpret('{"decision":"uncertain","mismatches":[]}',
                                   "affordable brands", item)["decision"], "uncertain")
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
