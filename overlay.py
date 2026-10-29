# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Apply reviewed files to a pinned SEARL checkout, preserving prior originals.'
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from control.contracts import strict_json
PIN = "b535d6199602bd9d747fc260bd29c77764799358"
DIRECTORIES = {'envs', 'verl', 'tests', 'environments', 'evaluation', 'generator', 'prompt', 'rag_server', 'recipe', 'redis_server', 'scripts'}
ROOTFILES = {'main_searl.sh', 'main_eval.sh', 'main_grpo_baseline.sh', 'main_grpo_megatron.sh', 'main_grpo.sh', 'main_gspo_megatron.sh', 'install.sh', 'README.md', 'pyproject.toml', 'pytest.ini', 'setup.py'}
def digest(path):
    return sha256(path.read_bytes()).hexdigest()
def base_matches(path, known):
    if not path.exists():
        return None in known
    if not path.is_file():
        return False
    content = path.read_bytes()
    return sha256(content).hexdigest() in known or sha256(content.replace(b'\r\n', b'\n')).hexdigest() in known
def safe_path(root, name):
    pure = PurePosixPath(name)
    if not pure.parts or pure.is_absolute() or any(x in (".", "..") for x in pure.parts) or "\\" in name or ":" in name:
        raise ValueError("Unsafe overlay relative path")
    path = root.joinpath(*pure.parts)
    if not path.resolve().is_relative_to(root.resolve()) or any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise ValueError("Overlay target escapes checkout or contains a symlink")
    if pure.parts[0] not in DIRECTORIES and not (len(pure.parts) == 1 and name in ROOTFILES):
        raise ValueError("Overlay target is outside reviewed source directories")
    if path.suffix not in {'.py', '.sh', '.md', '.toml', '.ini', '.json', '.yaml', '.yml', '.cfg'}:
        raise ValueError('Overlay source extension is not reviewed')
    return path
def plan(root, overlay):
    manifest = strict_json((overlay / "manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("format") not in ("evocond_searl_overlay_v1", "evocond_searl_overlay_v2") or manifest.get("upstream_commit") != PIN:
        raise ValueError("Unknown overlay format or upstream commit")
    if manifest.get('deleted') and manifest['format'] != 'evocond_searl_overlay_v2':
        raise ValueError('Source removals require the explicit migration format')
    actual = subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root),
                                      "rev-parse", "HEAD"], text=True).strip()
    if actual != PIN:
        raise ValueError("Check out the pinned upstream commit before installing")
    operations = []
    if not manifest.get("files"):
        raise ValueError("Empty overlay")
    for name, info in manifest["files"].items():
        source, target = safe_path(overlay, name), safe_path(root, name)
        if not source.is_file() or digest(source) != info["sha256"]:
            raise ValueError(f"Overlay file checksum mismatch: {name}")
        current = digest(target) if target.is_file() else None
        if current == info["sha256"]:
            continue
        if target.exists() and not target.is_file():
            raise ValueError(f"Target is not a regular file: {name}")
        if not base_matches(target, {info.get('base_sha256')}):
            raise ValueError(f"Unrecognized local edits or incomplete base checkout: {name}")
        operations.append((name, source, target))
    for name, info in manifest.get('deleted', {}).items():
        target = safe_path(root, name)
        if name in manifest['files'] or not isinstance(info, dict):
            raise ValueError('Ambiguous source migration operation')
        if not target.exists():
            continue
        if not target.is_file() or not base_matches(target, {info['base_sha256'], info.get('original_sha256')}):
            raise ValueError(f'Unrecognized local edits in renamed source: {name}')
        operations.append((name, None, target))
    return operations
def install(root, overlay, run=False):
    operations = plan(root, overlay)
    if not run or not operations:
        return {"run": run, "files_to_install": len(operations), "backup": None}
    backup = root / (".evocond-backup-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    backup.mkdir(exist_ok=False)
    # Preserve every original before installing or retiring renamed paths.
    for name, source, target in operations:
        if target.exists():
            saved = backup.joinpath(*PurePosixPath(name).parts)
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
    for name, source, target in operations:
        if source is None:
            target.unlink()
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name+".evocond-install")
        with temporary.open("xb") as stream:
            stream.write(source.read_bytes())
        temporary.replace(target)
    return {"run": True, "files_to_install": len(operations), "backup": str(backup)}
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(install(Path(args.official_root).resolve(strict=True),
                             Path(args.overlay).resolve(strict=True), args.run)))
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
