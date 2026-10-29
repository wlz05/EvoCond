# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Verify the EVOLT memory implementation."""
from math import isfinite, nan
from pathlib import Path
import json
import io
from random import Random
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from contextlib import redirect_stdout
from core.agent import Agent
from core.calib import Calib
from core.clock import Clock
from core.data import Corpus, Record, Source, baseline, identify
from core.embed import Embed, cosine, fold, unique, word
from core.env import Spec, World
from core.graph import Graph
from core.index import Aligned, Semantic, Structural
from core.model import Hit, Tool, Trace
from core.novel import Novelty
from core.task import Decomposer
from core.tool import python as calculate
from core.searl import SearlTools, endpoint, printable
from learn.credit import Episode, Step, credit, turn, utility
from learn.bench import Case, Plan, compare, evaluate, suite
from learn.eval import report
from learn.infer import Batch, Qwen, adapter, canonical, distinct, instruction, invoke, requests
from learn.metric import block, boxed, local as judge, measure, prediction, ranks
from learn.teach import Teacher
from learn.train import Draft, Rollout, Trainer
from learn.tune import calls, candidate, content, failures, format, outcome, reliability, shape, similar
from main import STATE, local, parser
import baseline as searl_baseline
class Tests(unittest.TestCase):
    def testembed(self):
        model = Embed(64)
        first = model.vector("search useful web evidence")
        second = model.vector("search useful web evidence")
        self.assertEqual(first, second)
        self.assertAlmostEqual(cosine(first, second), 1.0)
        self.assertEqual(fold("  Web   SEARCH  "), "web search")
        self.assertEqual(word("Web Search"), "websearch")
        self.assertEqual(unique("tool", {"tool"}), "toolb")
    def testtransaction(self):
        agent = Agent(64)
        proposal = agent.stage("search", "find verified facts", schema="query:string")
        key, rejected = agent.resolve(proposal, False, 1.0)
        self.assertIsNone(key)
        self.assertEqual(rejected.mode, "abstain")
        self.assertFalse(agent.graph.tools)
        proposal = agent.stage("search", "find verified facts", schema="query:string")
        key, accepted = agent.resolve(proposal, True, 1.0)
        self.assertEqual(key, "search")
        self.assertEqual(accepted.mode, "create")
        self.assertFalse(agent.pending)
        self.assertEqual(agent.journal[-1]["status"], "committed")
    def teststale(self):
        agent = Agent(64)
        stale = agent.stage("search", "find verified facts")
        agent.register("search", "find verified facts", reward=1.0)
        key, decision = agent.resolve(stale, True, 1.0)
        self.assertEqual(key, "search")
        self.assertEqual(decision.mode, "reuse")
        self.assertEqual(len(agent.graph.tools), 1)
        self.assertIn("stale", [item["status"] for item in agent.journal])
    def testclock(self):
        clock = Clock()
        self.assertEqual(clock.tick(10.0), 10.0)
        self.assertEqual(clock.tick(5.0), 11.0)
        self.assertEqual(clock.tick(), 12.0)
    def testidentity(self):
        agent = Agent(64)
        first, created = agent.register("SEARCH", "Find verified facts", schema="query:string", reward=1.0)
        second, merged = agent.register("Finder", "Find verified facts", schema="query:string", reward=0.5)
        third, split = agent.register("SEARCH", "Find verified facts", schema="query:integer", reward=1.0)
        self.assertTrue(created)
        self.assertFalse(merged)
        self.assertEqual(first, second)
        self.assertTrue(split)
        self.assertNotEqual(first, third)
    def testgraph(self):
        graph = Graph(rate=0.5, halflife=10.0)
        for key in ("plan", "search", "answer"):
            graph.put(Tool(key, key, key, "", [1.0, 0.0]))
        graph.link(["plan", "search", "answer"], 1.0, 10.0, "first")
        edge = graph.edges[("plan", "search", "supports")]
        before = edge.positive
        graph.link(["plan", "search", "answer"], 1.0, 11.0, "first")
        self.assertEqual(edge.positive, before)
        self.assertEqual(edge.count, 1)
        self.assertEqual(len(graph.tasks), 1)
        graph.fade(20.0)
        self.assertLess(edge.positive, before)
        graph.link(["plan", "search"], -1.0, 20.0, "second")
        self.assertIn(("plan", "search", "conflicts"), graph.edges)
        self.assertGreaterEqual(graph.spread({"plan": 1.0}).get("search", 0.0), 0.0)
        self.assertEqual(len(graph.shape("plan")), 27)
        count = graph.edges[("plan", "search", "precedes")].count
        graph.link(["plan", "search"], 1.0, 21.0, "second")
        self.assertEqual(graph.edges[("plan", "search", "precedes")].count, count)
        graph.link(["plan", "search"], 1.0, 22.0, "third")
        self.assertEqual(graph.edges[("plan", "search", "precedes")].count, count + 1)
    def testrank(self):
        agent = Agent(64)
        plan, _ = agent.register("plan", "break a task into steps", reward=1.0)
        search, _ = agent.register("search", "find grounded facts", reward=1.0)
        answer, _ = agent.register("answer", "compose a final response", reward=1.0)
        agent.graph.link([plan, search, answer], 1.0)
        trace = agent.recall("find facts", [plan], 3)
        self.assertEqual(len(trace.features), 11)
        self.assertEqual(len(trace.weights), 2)
        self.assertAlmostEqual(sum(trace.weights), 1.0)
        self.assertIn(search, [hit.key for hit in trace.hits])
        self.assertTrue(all(hit.sources for hit in trace.hits))
    def testablation(self):
        agent = Agent(64)
        agent.register("python", "calculate arithmetic expressions", reward=1.0)
        agent.register("search", "find web evidence", reward=1.0)
        agent.graph.link(["python", "search"], 1.0, task="known")
        weights = {"semantic": [1.0, 0.0], "structure": [0.0, 1.0], "equal": [0.5, 0.5]}
        for fusion, expected in weights.items():
            trace = agent.recall("calculate a sum", ["python"], limit=1, task="probe", fusion=fusion)
            self.assertEqual(trace.weights, expected)
            self.assertEqual(len(trace.hits), 1)
        self.assertAlmostEqual(sum(agent.recall("calculate a sum", limit=1, fusion="teacher").weights), 1.0)
        with self.assertRaises(ValueError):
            agent.recall("calculate a sum", fusion="unknown")
    def testteacher(self):
        teacher = Teacher(rate=0.5)
        hits = [Hit("a", "a", "a", "", 1.0, 0.9, 0.1, 0.2), Hit("b", "b", "b", "", 0.5, 0.1, 0.9, 0.3)]
        first = Trace("q", hits, [], [0.2, 0.8, 0.7, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], teacher.weights())
        second = Trace("q", hits, [], [0.8, 0.2, 0.7, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0], teacher.weights())
        before = [row[:] for row in teacher.matrix]
        advantages = teacher.group([(first, ["a"], 1.0), (second, ["b"], 0.0)])
        self.assertAlmostEqual(sum(advantages), 0.0)
        self.assertNotEqual(before, teacher.matrix)
        self.assertLessEqual(teacher.weights([0.0] * 11, True)[1], teacher.coldcap)
        self.assertEqual(set(teacher.state()["belief"]), {"semantic", "structure"})
        before = teacher.weights(first.features)
        teacher.supervise([(first.features, [1.0, 0.0])] * 4)
        self.assertGreater(teacher.weights(first.features)[0], before[0])
    def testcalib(self):
        scores = [0.1, 0.2, 0.7, 0.9]
        labels = [0, 0, 1, 1]
        model = Calib()
        before = report(scores, labels)
        loss = model.fit(scores, labels)
        mapped = [model.predict(value) for value in scores]
        after = report(mapped, labels)
        self.assertLessEqual(loss, before["brier"])
        self.assertLessEqual(after["brier"], before["brier"])
        self.assertEqual(set(after), {"brier", "ece", "aurc"})
        with self.assertRaises(ValueError):
            model.map(0.5, 0.0)
        agent = Agent(64)
        agent.calibrate("SEMANTIC", scores, labels)
        self.assertTrue(agent.ranker.semsup.fitted)
    def testcredit(self):
        episodes = [Episode([Step("search", 1.0), Step("answer", 0.5)], 1.0), Episode([Step("search", 0.0), Step("answer", 0.0)], 0.0)]
        values = credit(episodes)
        self.assertGreater(values[0][0], values[1][0])
        turns = turn([[1.0, 0.0], [0.0, 2.0]])
        self.assertAlmostEqual(turns[0][0] + turns[1][0], 0.0)
        self.assertAlmostEqual(utility(1.0, [0.5], cost=2.0), 1.4)
    def testtrainer(self):
        agent = Agent(64)
        search, _ = agent.register("search", "find facts", reward=1.0)
        trace = agent.recall("find facts")
        weak = Rollout([Step("search", 0.0)], 0.0, [search], [], trace, verified=True)
        strong = Rollout([Step("search", 0.5)], 1.0, [search], [Draft("answer", "write result")], trace, verified=True)
        result = Trainer(agent).learn([weak, strong])
        self.assertEqual(result["committed"], 1)
        self.assertIn("answer", agent.graph.tools)
        loss = Trainer(agent).warm([(trace, 1.0, 0.0), (trace, 0.0, 1.0)])
        self.assertGreater(loss, 0.0)
    def teststore(self):
        agent = Agent(64)
        agent.register("search", "find verified facts", schema="query:string", reward=1.0)
        proposal = agent.stage("answer", "write a final result")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            agent.save(path)
            restored = Agent.open(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["state"]["dim"] = 65
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                Agent.open(path)
            path.write_text(json.dumps(agent.dump()), encoding="utf-8")
            with self.assertRaises(ValueError):
                Agent.open(path)
        self.assertEqual(set(agent.graph.tools), set(restored.graph.tools))
        self.assertEqual(restored.dump()["version"], 4)
        self.assertIn(proposal.token, restored.pending)
        restored.resolve(restored.pending[proposal.token], True, 1.0)
        self.assertIn("answer", restored.graph.tools)
        self.assertEqual(set(restored.semantic.vectors), set(restored.graph.tools))
    def testmigrate(self):
        agent = Agent(64)
        first, _ = agent.register("first", "first tool", reward=1.0)
        second, _ = agent.register("second", "second tool", reward=1.0)
        agent.graph.link([first, second], 1.0, task="known")
        state = agent.dump()
        state["version"] = 3
        state["graph"].pop("tasks")
        for edge in state["graph"]["edges"]:
            edge.pop("tasks")
            edge["count"] = 5
            edge["positive"] *= 5
        state["teacher"]["bias"] = [0.0, 0.0, 0.0]
        state["teacher"]["matrix"] = [[0.0] * 8 for _ in range(3)]
        restored = Agent.restore(state)
        self.assertEqual(restored.dump()["version"], 4)
        self.assertEqual(len(restored.teacher.bias), 2)
        self.assertTrue(all(edge.count == 1 for edge in restored.graph.edges.values()))
    def testworld(self):
        tool = Spec("SEARCH", {"Query": "STRING"}, {"READY": True}, {"FOUND": "$arg:query"}, ["Finder"])
        world = World({"READY": True}, [tool])
        good = Plan([("FINDER", {"QUERY": "facts"})])
        bad = Plan([("search", {})])
        results = compare(world, [good, bad], {"FOUND": "facts"})
        self.assertTrue(results[0].success)
        self.assertFalse(results[1].success)
        self.assertGreater(results[0].utility, results[1].utility)
        self.assertNotIn("found", world.state)
        world.shift("search", inputs={"query": "integer"})
        self.assertFalse(world.execute("finder", {"query": "facts"})["ok"])
        upper = World({}, [Spec("SAVE", {"VALUE": "STRING"}, {}, {"OUT": "$ARG:VALUE"})])
        self.assertTrue(upper.execute("save", {"value": "X"})["ok"])
        self.assertTrue(upper.verify({"out": "X"}))
        upper.put(Spec("other"))
        before = upper.dump()
        with self.assertRaises(ValueError):
            upper.shift("save", "other")
        self.assertEqual(upper.dump(), before)
        with self.assertRaises(TypeError):
            upper.execute("save", [])
        with self.assertRaises(ValueError):
            upper.execute("save", {"value": nan})
    def testbench(self):
        result = evaluate()
        self.assertEqual(len(result["cases"]), 5)
        self.assertEqual(result["oracle"], 1.0)
        self.assertTrue(all(row["best"] == 0 for row in result["cases"]))
        self.assertEqual({case.action for case in suite()}, {"reuse", "split", "create", "abstain"})
        with self.assertRaises(ValueError):
            evaluate([Case("bad", World(), [], {}, "abstain")])
    def testdata(self):
        self.assertEqual(identify("  GSM8K  "), "gsm8k")
        self.assertEqual(identify("2WikiMultiHopQA"), "2wiki")
        self.assertEqual(identify("AIME-2024"), "aime24")
        self.assertEqual(tuple(baseline()["test"]["datasets"]), ("gsm8k", "math500", "aime24", "hotpotqa", "2wiki", "musique", "bamboogle"))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "rows.jsonl"
            rows = [{"data_source": "HOTPotQA", "question": "Who?", "reward_model": {"ground_truth": "Ada"}, "extra_info": {"index": 1}}, {"data_source": "DR_grpo_mix", "prompt": [{"role": "user", "content": "What is 2+2?"}], "reward_model": {"ground_truth": "4"}, "extra_info": {"index": 2, "split": "TRAIN"}}]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            records = Corpus(root).load("rows.jsonl")
            self.assertEqual([(item.dataset, item.split) for item in records], [("hotpotqa", "test"), ("toolstar", "train")])
            self.assertEqual(Corpus(root).summary(records)["records"], 2)
            gsm = root / "gsm.jsonl"
            gsm.write_text(json.dumps({"dataset": "gsm8k", "question": "What is two plus three?", "answer": "work #### 5"}), encoding="utf-8")
            self.assertEqual(Corpus(root).load("gsm.jsonl")[0].answer, "5")
            self.assertIn("\\boxed", instruction(records[0])[0]["content"])
            first = Batch(root).run(records, "predictions.jsonl", lambda item, hits: "<answer>\\boxed{x}</answer>", limit=1)
            second = Batch(root).run(records, "predictions.jsonl", lambda item, hits: "bad", limit=1)
            self.assertEqual((first["generated"], second["skipped"]), (1, 1))
            self.assertEqual(first["output"], "predictions.jsonl")
            self.assertEqual(len(Source(root).read("predictions.jsonl")), 1)
            self.assertEqual(Batch(root).run(records, "fusion.jsonl", lambda item, hits: "answer", limit=1, config={"protocol": {"fusion": "equal", "budget": 1}})["generated"], 1)
            self.assertEqual(Source(root).read("fusion.jsonl")[0]["protocol"]["budget"], 1)
            with self.assertRaises(ValueError):
                Batch(root).run(records, "fusion.jsonl", lambda item, hits: "answer", limit=1, config={"protocol": {"fusion": "teacher", "budget": 1}})
            saved = root / "saved"
            saved.mkdir()
            self.assertFalse(adapter(saved))
            (saved / "adapter_config.json").write_text("{}", encoding="utf-8")
            self.assertTrue(adapter(saved))
            with self.assertRaises(ValueError):
                Source(root).path("../rows.jsonl")
            with self.assertRaises(ValueError):
                Source(root).path(str(path))
    def testmetric(self):
        self.assertEqual(block("assistant:\n<answer>x<answer>bad</answer>x</answer><answer>\\boxed{42}</answer>"), "\\boxed{42}")
        self.assertEqual(boxed("first \\boxed{1} last \\boxed{\\frac{1}{2}}"), "\\frac{1}{2}")
        self.assertEqual(boxed("last \\fbox 7$"), "7")
        self.assertEqual(block(canonical("<answer>3</answer>")), "\\boxed{3}")
        self.assertEqual(boxed(block(canonical("Answer: $\\boxed{7}$"))), "7")
        question = "Find all integers n that satisfy the equation, separated by commas."
        self.assertTrue(judge("-2,1", "1,-2", question))
        self.assertFalse(judge("-2,1", "1,-2", "Return the ordered pair of integers."))
        self.assertFalse(judge("-2,1,3", "1,-2", question))
        item = Record("unordered", "math500", "test", question, "1,-2", "", {})
        self.assertEqual(measure([item], {item.key: {"output": "<answer>\\boxed{-2,1}</answer>"}})["overall"]["correct"], 1)
        self.assertEqual(outcome("<answer>\\boxed{-2,1}</answer>", "1,-2", question), 1.0)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data.json"
            data.write_text(json.dumps([{"dataset": "GSM8K", "id": 1, "question": "Half?", "answer": "0.5"}, {"dataset": "GSM8K", "id": 2, "question": "Missing?", "answer": "x"}]), encoding="utf-8")
            records = Corpus(root).load("data.json")
            values = {records[0].key: {"output": "<answer>\\boxed{\\frac{1}{2}}</answer>"}, records[1].key: {"output": "x"}}
            result = measure(records, values)
            self.assertEqual(result["datasets"]["gsm8k"]["correct"], 1)
            self.assertEqual(result["datasets"]["gsm8k"]["noanswer"], 1)
            self.assertEqual(result["overall"]["pass@1"], 0.5)
            called = {records[0].key: {"output": '<tool_call>{}</tool_call>\n<tool_response name="python">\n5\n</tool_response>\n<answer>\\boxed{5}</answer>'}}
            self.assertEqual(measure(records[:1], called)["overall"]["toolrate"], 1.0)
            rows = [{"key": records[0].key, "output": "<answer>\\boxed{no}</answer>", "correct": 1}]
            self.assertEqual(measure(records[:1], prediction(rows))["mode"], "labels")
            with self.assertRaises(ValueError):
                measure(records[:1], {records[0].key: {"output": "<answer>\\boxed{x}</answer>"}}, lambda *_: 0.5)
        table = {"a": {"GSM8K": 0.9, "HotpotQA": 0.5}, "b": {"gsm8k": 0.8, "hotpotqa": 0.5}}
        self.assertEqual(ranks(table), {"a": 1.25, "b": 1.75})
    def testnovelty(self):
        vector = [1.0] + [0.0] * 63
        near = [0.99, 0.01] + [0.0] * 62
        for method in ("shrink", "knn", "energy", "hybrid"):
            model = Novelty(64, method=method)
            model.learn("search", vector, 0.9)
            model.learn("search", near, 0.8)
            decision = model.decide(near, {"search": -1.0})
            self.assertEqual(decision.mode, "reuse")
            self.assertEqual(Novelty.load(model.dump()).method, method)
        with self.assertRaises(ValueError):
            Novelty(64, method="bad")
    def testindex(self):
        graph = Graph()
        graph.put(Tool("search", "search", "search", "", [1.0] + [0.0] * 63))
        semantic = Semantic.load({"vectors": {"search": 1}, "version": 2}, 64)
        self.assertFalse(semantic.vectors)
        semantic.rebuild(graph)
        self.assertEqual(set(semantic.vectors), {"search"})
        structural = Structural.load({"vectors": {"search": 1}}, graph)
        self.assertEqual(set(structural.vectors), {"search"})
        structural = Structural.load({"vectors": {"search": [0.0]}}, graph)
        self.assertEqual(len(structural.vectors["search"]), 27)
        self.assertTrue(Aligned.check(graph, semantic, structural))
        agent = Agent(64)
        agent.graph.put(Tool("manual", "manual", "manual", "", agent.vector("manual", "manual")))
        self.assertIn("manual", [hit.key for hit in agent.recall("manual").hits])
    def testtask(self):
        model = Decomposer()
        self.assertEqual(model.split("Search facts; then write answer"), ["search facts", "write answer"])
        self.assertEqual(model.split("ignored", ["SEARCH Facts", "WRITE Answer"]), ["search facts", "write answer"])
        agent = Agent(64)
        agent.register("search", "find facts", reward=1.0)
        first = agent.recall("find and write", subtasks=["find facts", "write answer"], task="case one")
        second = agent.recall("find and write", subtasks=["find facts", "write answer"], task="CASE ONE")
        self.assertEqual(first.task, second.task)
        self.assertEqual(first.subtasks, ["find facts", "write answer"])
    def testcase(self):
        agent = Agent(64)
        search, _ = agent.register("SEARCH", "Find VERIFIED Facts", reward=1.0)
        answer, _ = agent.register("Answer", "WRITE the Final RESPONSE", reward=1.0)
        trace = agent.recall("FIND verified FACTS", ["SEARCH"])
        self.assertEqual(trace.query, "find verified facts")
        trials = agent.graph.tools[answer].trials
        agent.commit(trace, ["ANSWER"], 1.0)
        self.assertEqual(agent.graph.tools[answer].trials, trials + 1)
        data = agent.dump()
        data["graph"]["tools"][0]["key"] = data["graph"]["tools"][0]["key"].upper()
        data["novelty"]["clusters"][0]["label"] = data["novelty"]["clusters"][0]["label"].upper()
        restored = Agent.restore(data)
        self.assertIn(search, restored.graph.tools)
    def testrelative(self):
        self.assertEqual(local("memory.json").parent, STATE)
        for value in ("../memory.json", str(STATE / "memory.json"), "C:\\memory.json", "C:memory.json", "/memory.json"):
            with self.assertRaises(ValueError):
                local(value)
            with self.assertRaises(ValueError):
                Source().path(value)
        self.assertEqual(parser().parse_args(["bench"]).command, "bench")
        self.assertEqual(parser().parse_args(["data"]).command, "data")
        self.assertEqual(parser().parse_args(["infer", "data/a.jsonl", "result/a.jsonl"]).model, "autodl-tmp/model/qwen4b")
        self.assertEqual(parser().parse_args(["infer", "data/a.jsonl", "result/a.jsonl", "--mode", "tool"]).mode, "tool")
        options = parser().parse_args(["infer", "data/a.jsonl", "result/a.jsonl", "--mode", "evolt", "--fusion", "semantic", "--retrieve-limit", "1", "--restrict-tools"])
        self.assertEqual((options.fusion, options.retrieve_limit, options.restrict_tools), ("semantic", 1, True))
        train = parser().parse_args(["train"])
        self.assertEqual((train.batch, train.accum, train.generations, train.seed, train.tokens, train.ability, train.save, train.rate), (1, 4, 4, 2, 768, "all", 25, 1e-6))
    def testbounds(self):
        agent = Agent(64)
        with self.assertRaises(ValueError):
            agent.register("", "text")
        with self.assertRaises(ValueError):
            agent.recall("  ")
        with self.assertRaises(ValueError):
            agent.register("tool", "text", reward=nan)
        with self.assertRaises(ValueError):
            agent.graph.link([], nan)
        state = agent.dump()
        state["journal"] = [{"stamp": "bad"}, {"stamp": 1.0}]
        self.assertEqual(len(Agent.restore(state).journal), 1)
        pending = Agent(64)
        proposal = pending.stage("tool", "text")
        state = pending.dump()
        state["pending"][0]["key"] = "BAD"
        self.assertNotIn(proposal.token, Agent.restore(state).pending)
    def teststress(self):
        agent = Agent(64)
        random = Random(17)
        for pos in range(200):
            name = "tool" + chr(97 + pos % 20)
            schema = "value:string" if pos % 7 else "value:integer"
            key, _ = agent.register(name.upper(), f"handle task {pos % 20}", reward=random.uniform(-1.0, 1.0), schema=schema)
            trace = agent.recall(f"HANDLE TASK {random.randrange(20)}", [key], random.randrange(1, 6))
            if trace.hits:
                agent.commit(trace, [trace.hits[0].key], random.uniform(-1.0, 1.0))
        restored = agent.clone()
        self.assertEqual(set(restored.graph.tools), set(restored.semantic.vectors))
        self.assertFalse(restored.pending)
        self.assertTrue(all(edge.source in restored.graph.tools and edge.target in restored.graph.tools for edge in restored.graph.edges.values()))
        self.assertTrue(all(isfinite(hit.score) for hit in restored.recall("handle task", limit=20).hits))
        stamps = [tool.used for tool in restored.graph.tools.values()] + [edge.used for edge in restored.graph.edges.values()]
        self.assertGreaterEqual(restored.clock.value, max(stamps, default=0.0))
    def testtool(self):
        self.assertEqual(calculate("2 + 3"), "5")
        self.assertEqual(calculate("total = 0\nfor value in range(4):\n    total += value\ntotal"), "6")
        self.assertEqual(calculate("values = []\nfor value in range(1, 10000):\n    if str(value ** 3).endswith('528'):\n        values.append(value)\nvalues[0]"), "162")
        self.assertTrue(calculate("import os").startswith("Error:"))
        self.assertTrue(calculate("pow(2, 1025)").startswith("Error:"))
        self.assertEqual(calculate("from math import comb\ncomb(25, 3)"), "2300")
        self.assertTrue(calculate("import math\nmath.atan2(3, 0)").startswith("1.5707"))
        self.assertTrue(calculate("def f(x):\n    return (3 * x - 2) / (x - 2)\nf(-2) + f(-1) + f(0)").startswith("4.6666"))
        self.assertTrue(calculate("def f(x):\n    return f(x)\nf(1)").startswith("Error:"))
        self.assertEqual(calculate("def f(n):\n    values = [i for i in range(1, n) if n % i == 0]\n    return sum(values)\nf(284)"), "220")
        self.assertEqual(calculate("for i in range(1, 1000000):\n    if i % 30 == 0:\n        if set(str(i)) <= {'0', '2'}:\n            print(i)\n            break"), "2220")
        self.assertTrue(calculate("from math import log\n(66200 / 20000) ** (1 / 3) - 1").startswith("0.4903"))
        self.assertEqual(calculate("a = 1\n b = 2\na + b"), "3")
        self.assertIn("budget exceeded", calculate("total = 0\nfor a in range(10000):\n    for b in range(10000):\n        total += 1\ntotal"))
        request = '<tool_call>{"name":"python","arguments":{"code":"2 + 3"}}</tool_call>'
        self.assertEqual(requests(request), [("python", {"code": "2 + 3"})])
        self.assertEqual(distinct(requests(request + request)), [("python", {"code": "2 + 3"})])
        used = set()
        self.assertEqual(invoke("python", {"code": "2 + 3"}, used), "5")
        self.assertTrue(invoke("python", {"code": "3 + 4"}, used).startswith("Error:"))
        self.assertEqual(invoke("python", {"code": "2 + 3"}, set(), {"search"}), "Error: tool was not retrieved.")
        with tempfile.TemporaryDirectory() as folder:
            restricted = Qwen(Path(folder).name, tools=True, guided=True, root=Path(folder).parent)
            self.assertEqual(restricted.available([Hit("search", "search", "find facts", "", 1.0, 1.0, 1.0, 1.0)]), ["search"])
            self.assertEqual(restricted.available(), [])
        completion = [{"role": "assistant", "content": "working", "tool_calls": [{"function": {"name": "python"}}]}, {"role": "tool", "name": "python", "content": "5"}, {"role": "assistant", "content": "<answer>\\boxed{5}</answer>"}]
        self.assertEqual(calls(completion), ["python"])
        self.assertIn("boxed", content(completion))
        self.assertEqual(candidate("The answer is 5"), "5")
        self.assertEqual(outcome("The answer is 5", "5"), 1.0)
        self.assertEqual(outcome("No final value", "5"), -0.5)
        self.assertGreater(similar("49", "50"), similar("5", "50"))
        self.assertEqual(shape("Answer: \\boxed{5}"), 0.03)
        self.assertEqual(format([completion]), [0.1])
        failed = [{"role": "tool", "name": "python", "content": "Error: rejected"}, {"role": "assistant", "content": "<answer>\\boxed{5}</answer>"}]
        self.assertEqual((failures(failed), reliability([failed])), (1, [-0.1]))
    def testsearl(self):
        with self.assertRaises(ValueError):
            endpoint("https://example.com/retrieve", "/retrieve")
        with self.assertRaises(ValueError):
            endpoint("http://127.0.0.1:5003/wrong", "/retrieve")
        self.assertIn("print(2 + 3)", printable("2 + 3"))
        backend = SearlTools()
        response = {"result": [[{"document": {"contents": "Einstein\nA physicist"}}]]}
        class Reply:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self):
                return json.dumps(response).encode("utf-8")
        class Opener:
            def open(self, request, timeout):
                self.request = request
                self.timeout = timeout
                return Reply()
        opener = Opener()
        with patch("core.searl.build_opener", return_value=opener):
            self.assertIn("A physicist", backend.search("Albert Einstein"))
        self.assertEqual(json.loads(opener.request.data)["topk"], 3)
        self.assertEqual(opener.timeout, 10)
        fake = SimpleNamespace(RunCodeRequest=lambda **kwargs: SimpleNamespace(**kwargs), run_code=lambda request: SimpleNamespace(json=lambda: json.dumps({"run_result": {"status": "Finished", "stdout": "5\n", "stderr": ""}})), set_endpoint=lambda value: None)
        with patch.dict("sys.modules", {"sandbox_fusion": fake}):
            self.assertIn("STDOUT:\n5", backend.python("2 + 3"))
        self.assertEqual(invoke("search", {"query": "X"}, set(), catalog={"search": lambda query: query}), "X")
        with tempfile.TemporaryDirectory() as folder:
            policy = Qwen(Path(folder).name, tools=True, root=Path(folder).parent, backend="searl")
            self.assertEqual(policy.available(), ["python", "search"])
            self.assertEqual(policy.rounds, 2)
    def testsearlbaselinepreflight(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            (root / "model").mkdir()
            (root / "data" / "math.jsonl").write_text(json.dumps({"dataset": "math500", "question": "2+3?", "answer": "5"}), encoding="utf-8")
            (root / "model" / "config.json").write_text(json.dumps({"model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 36}), encoding="utf-8")
            args = ["baseline.py", "data/math.jsonl", "result/math", "--name", "math500", "--model", "model", "--arms", "plain,tool", "--limit", "1"]
            with patch("sys.argv", args), patch("baseline.Source", lambda: Source(root)), patch("baseline.Corpus", lambda: Corpus(root)), redirect_stdout(io.StringIO()) as output:
                searl_baseline.main()
            self.assertEqual(json.loads(output.getvalue())["selected"], 1)
            self.assertFalse((root / "result").exists())
    def testsource(self):
        root = Path(__file__).resolve().parent
        files = list(root.rglob("*"))
        self.assertTrue(all(path.is_dir() or path.suffix == ".py" for path in files))
        for path in (item for item in files if item.is_file()):
            text = path.read_text(encoding="utf-8")
            self.assertFalse(any(not line.strip() for line in text.splitlines()))
            self.assertNotRegex(text, r"[\u4e00-\u9fff]")
if __name__ == "__main__":
    unittest.main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
