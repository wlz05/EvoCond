# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Read-only release gate for a matched three-arm dynamic-tool experiment.'
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
ARMS = ("searl", "evocond", "teacher")
FORMAT = "evocond_searl_migration_v1"
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, help="Read-only JSON protocol description")
    return parser.parse_args()
def _path(value: str, base: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()
def _digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
def inspect(protocol: dict, *, base: Path) -> dict:
    blockers = []
    assets = {}
    if protocol.get("format") != FORMAT:
        blockers.append("protocol format is not evocond_searl_migration_v1")
    official = _path(protocol.get("official_repo", "."), base)
    if not (official / "envs" / "tool_manager" / "centralized" / "central.py").is_file():
        blockers.append("official SEARL tool manager is missing")
    entrypoint = official / "main_searl.sh"
    if not entrypoint.is_file():
        blockers.append("official SEARL training entrypoint is missing")
    elif re.search(r"\\\r?\n[ \t]*\r?\n", entrypoint.read_text(encoding="utf-8")):
        blockers.append("official main_searl.sh has blank lines after shell continuations")
    try:
        commit = subprocess.check_output(["git", "-c", f"safe.directory={official}",
                                          "-C", str(official), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    if not commit or commit != protocol.get("official_commit"):
        blockers.append("official SEARL checkout differs from pinned commit")
    assets["official_commit"] = commit
    model = _path(protocol.get("base_model", "."), base)
    config = model / "config.json"
    if not config.is_file():
        blockers.append("Qwen base config.json is missing")
    else:
        model_config = json.loads(config.read_text(encoding="utf-8"))
        if model_config.get("model_type") != "qwen3":
            blockers.append("base model is not identified as Qwen3")
        assets["base_config_sha256"] = _digest(config)
    datasets = protocol.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != {"train", "validation", "test"}:
        blockers.append("three frozen train/validation/test datasets are required")
    else:
        paths = {split: _path(value, base) for split, value in datasets.items()}
        if len(set(paths.values())) != 3:
            blockers.append("train/validation/test resolve to overlapping files")
        for split, path in paths.items():
            if not path.is_file():
                blockers.append(f"{split} dataset is missing")
            else:
                assets[f"{split}_sha256"] = _digest(path)
        if len(set(assets.get(f"{split}_sha256") for split in paths)) != 3:
            blockers.append("dataset files are identical or incomplete")
    audit_ref = protocol.get("split_identity_report")
    audit_path = _path(audit_ref, base) if isinstance(audit_ref, str) else None
    if audit_path is None or not audit_path.is_file():
        blockers.append("independent-question split identity audit is missing")
    else:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("format") != "evocond_searl_split_audit_v2" or audit.get("disjoint") is not True:
            blockers.append("independent-question split identity audit failed")
        for split in ("train", "validation", "test"):
            if audit.get("datasets", {}).get(split, {}).get("sha256") != assets.get(f"{split}_sha256"):
                blockers.append(f"{split} dataset differs from the split audit")
    sandbox = protocol.get("sandbox")
    if not isinstance(sandbox, dict) or (sandbox.get("isolation") not in {"container", "microvm"}
                                         or sandbox.get("isolation_verified") is not True
                                         or sandbox.get("generated_code_host_execution") is not False):
        blockers.append("isolated generated-code execution has not been verified")
    methods = protocol.get("methods")
    if not isinstance(methods, dict) or set(methods) != set(ARMS):
        blockers.append("all three method integrations are required")
    else:
        for arm in ARMS:
            method = methods[arm]
            if not isinstance(method, dict) or method.get("integrated") is not True:
                blockers.append(f"{arm} policy is not integrated with official SEARL rollouts")
                continue
            entrypoint = method.get("entrypoint")
            if not isinstance(entrypoint, str) or not _path(entrypoint, official).is_file():
                blockers.append(f"{arm} training entrypoint is missing")
            if method.get("same_base_and_split") is not True:
                blockers.append(f"{arm} has not confirmed matched base/split provenance")
    if protocol.get("fresh_test_sealed") is not True:
        blockers.append("fresh test split is not sealed")
    return {"ready": not blockers, "blockers": blockers, "assets": assets,
            "note": "A manifest assertion is not a sandbox inspection; verify the real deployment before a run."}
def main() -> None:
    args = arguments()
    path = Path(args.protocol).resolve(strict=True)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    result = inspect(protocol, base=path.parent)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ready"]:
        raise SystemExit(2)
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
