# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

"""Load and identify the baseline datasets."""
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path, PurePosixPath, PureWindowsPath
import csv
import json
import re
import unicodedata
BASE = Path(__file__).resolve().parents[2]
TRAIN = ("toolstar",)
TEST = ("gsm8k", "math500", "aime24", "hotpotqa", "2wiki", "musique", "bamboogle")
AUX = ("webwalker",)
ALIASES = {"toolstar": ("toolstar", "multitoolrl10k", "arporlreasoning10k", "drgrpomix", "grpomix", "arpo"), "gsm8k": ("gsm8k", "gsm"), "math500": ("math500",), "aime24": ("aime24", "aime2024"), "hotpotqa": ("hotpotqa", "hotpot"), "2wiki": ("2wiki", "2wikimultihopqa", "twowiki", "twowikimultihopqa"), "musique": ("musique", "musiq", "misque"), "bamboogle": ("bamboogle", "bamb"), "webwalker": ("webwalker",)}
@dataclass(slots=True)
class Record:
    key: str
    dataset: str
    split: str
    question: str
    answer: object
    ability: str
    raw: dict
    def dump(self) -> dict:
        return {"key": self.key, "dataset": self.dataset, "split": self.split, "question": self.question, "answer": self.answer, "ability": self.ability}
def token(value: object) -> str:
    """Normalize a dataset label before matching."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(re.findall(r"[a-z0-9]+", text))
def identify(value: object) -> str:
    """Map one normalized label to a canonical dataset."""
    clean = token(value)
    if not clean:
        return ""
    for name, aliases in ALIASES.items():
        if clean in aliases:
            return name
    matches = [(len(alias), name) for name, aliases in ALIASES.items() for alias in aliases if len(alias) >= 5 and alias in clean]
    return max(matches, default=(0, ""))[1]
def split(value: object, dataset: str) -> str:
    """Resolve an explicit split before using baseline defaults."""
    clean = token(value)
    if clean in ("train", "training"):
        return "train"
    if clean in ("valid", "validation", "val", "dev", "development"):
        return "valid"
    if clean in ("test", "testing", "eval", "evaluation"):
        return "test"
    return "train" if dataset in TRAIN else "test"
def nested(row: dict, *names: str):
    """Read one field from common nested schemas."""
    for name in names:
        value = row.get(name)
        if value not in (None, "", []):
            return value
    for parent in ("extra_info", "metadata", "reward_model"):
        value = row.get(parent)
        if isinstance(value, dict):
            for name in names:
                item = value.get(name)
                if item not in (None, "", []):
                    return item
    return None
def prompt(row: dict) -> str:
    """Extract a question without changing its content."""
    value = nested(row, "question", "input", "query", "problem")
    if value is None and isinstance(row.get("prompt"), list):
        messages = [item for item in row["prompt"] if isinstance(item, dict) and item.get("content")]
        users = [item for item in messages if token(item.get("role")) == "user"]
        value = (users or messages)[-1].get("content") if messages else None
    return str(value or "").strip()
def target(row: dict):
    """Extract one reference answer from common schemas."""
    value = nested(row, "ground_truth", "answer", "answers", "target", "output", "response")
    if isinstance(value, dict):
        for name in ("target", "answer", "value", "ground_truth"):
            if name in value:
                return value[name]
    return value
class Source:
    """Read supported files below one project root."""
    def __init__(self, root: str | Path = BASE):
        self.root = Path(root).resolve()
    def path(self, value: str | Path) -> Path:
        text = str(value)
        path = Path(text)
        windows = PureWindowsPath(text)
        posix = PurePosixPath(text)
        if path.is_absolute() or windows.is_absolute() or windows.drive or posix.is_absolute() or not path.parts or ".." in windows.parts or ".." in posix.parts:
            raise ValueError("dataset path must be relative")
        result = (self.root / path).resolve()
        if result == self.root or self.root not in result.parents:
            raise ValueError("dataset path must stay within the project")
        return result
    def read(self, value: str | Path) -> list[dict]:
        path = self.path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        suffix = path.suffix.casefold()
        if suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        elif suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            rows = data if isinstance(data, list) else data.get("data", data.get("rows", [data])) if isinstance(data, dict) else []
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        elif suffix == ".parquet":
            try:
                import pandas
                rows = pandas.read_parquet(path).to_dict("records")
            except ImportError as error:
                raise RuntimeError("parquet support requires pandas and pyarrow") from error
            except Exception as error:
                raise RuntimeError("parquet file could not be read") from error
        else:
            raise ValueError("dataset file must be json, jsonl, csv, or parquet")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("dataset rows must be objects")
        return rows
class Corpus:
    """Convert source rows into stable baseline records."""
    def __init__(self, root: str | Path = BASE):
        self.source = Source(root)
    def name(self, row: dict, default: str = "") -> str:
        values = [default, row.get("dataset"), row.get("dataset_name"), row.get("source"), row.get("benchmark")]
        for parent in ("extra_info", "metadata", "reward_model"):
            value = row.get(parent)
            if isinstance(value, dict):
                values.extend(value.get(name) for name in ("dataset", "dataset_name", "source", "benchmark"))
        values.append(row.get("data_source"))
        for value in values:
            name = identify(value)
            if name:
                return name
        return ""
    def load(self, path: str | Path, name: str = "", part: str = "") -> list[Record]:
        records = []
        for pos, row in enumerate(self.source.read(path)):
            dataset = self.name(row, name)
            if not dataset:
                raise ValueError(f"dataset is unknown at row {pos}")
            question = prompt(row)
            answer = target(row)
            if dataset == "gsm8k" and isinstance(answer, str) and "####" in answer:
                answer = answer.rsplit("####", 1)[-1].strip()
            if not question or answer in (None, "", []):
                raise ValueError(f"dataset row {pos} lacks a question or answer")
            section = split(part or nested(row, "split"), dataset)
            rawkey = nested(row, "id", "index", "key")
            digest = blake2b(f"{dataset}|{section}|{rawkey}|{question}".encode("utf-8"), digest_size=10).hexdigest()
            records.append(Record(digest, dataset, section, question, answer, token(nested(row, "ability")), row))
        return records
    def summary(self, records: list[Record]) -> dict:
        counts: dict[str, dict[str, int]] = {}
        for item in records:
            row = counts.setdefault(item.dataset, {"train": 0, "valid": 0, "test": 0})
            row[item.split] = row.get(item.split, 0) + 1
        return {"datasets": counts, "records": len(records), "main": list(TEST), "training": list(TRAIN), "auxiliary": list(AUX)}
def baseline() -> dict:
    """Describe the exact baseline data protocol."""
    return {"train": {"datasets": list(TRAIN), "samples": 10000}, "test": {"datasets": list(TEST)}, "auxiliary": {"datasets": list(AUX), "main_table": False}, "metrics": ["pass@1", "accuracy", "average rank"], "judge": "binary llm judge"}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
