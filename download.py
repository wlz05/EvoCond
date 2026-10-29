# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import argparse
import json
import os
import subprocess
from pathlib import Path
MODEL = "Qwen/Qwen3-4B"
DATASET = "JiayuJeff/CostBench"
REPOS = {"costbench": "https://github.com/JiayuJeff/CostBench.git", "realtimegym": "https://github.com/wenyl22/RealtimeGym.git"}
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="download")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--skipmodel", action="store_true")
    parser.add_argument("--skiprepos", action="store_true")
    return parser.parse_args()
def clone(name: str, url: str) -> None:
    target = Path("assets") / "repos" / name
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)
def main() -> None:
    args = arguments()
    os.environ.setdefault("HF_HOME", "assets/cache")
    if args.mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from huggingface_hub import snapshot_download
    Path("assets").mkdir(exist_ok=True)
    if not args.skipmodel:
        snapshot_download(args.model, local_dir="assets/model")
    snapshot_download(DATASET, repo_type="dataset", local_dir="assets/data/costbench")
    if not args.skiprepos:
        for name, url in REPOS.items():
            clone(name, url)
    manifest = {"model": args.model, "dataset": DATASET, "repos": REPOS}
    Path("assets/manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=True, indent=2))
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
