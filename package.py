# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Build a source-only reviewed overlay; never include models or experiment runs.'
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED
from overlay import safe_path
PIN = "b535d6199602bd9d747fc260bd29c77764799358"
def build(workspace, output):
    official = workspace / "EvoCond/EvoCond/baselines/SEARL-official"
    method = workspace / "EvoCond/EvoCond"
    prefix = ["git", "-c", 'core.autocrlf=false', "-c", f"safe.directory={official.as_posix()}", "-C", str(official)]
    def git(*args):
        return subprocess.check_output(prefix+list(args))
    if git("rev-parse", "HEAD").decode().strip() != PIN:
        raise ValueError("Unexpected upstream version")
    names = git("diff", "--name-only", '-z', "HEAD").decode().rstrip('\0').split('\0')
    names += git("ls-files", "--others", "--exclude-standard", '-z').decode().rstrip('\0').split('\0')
    entries, manifest = {}, {"format": "evocond_searl_overlay_v2", "upstream_commit": PIN, "files": {}, 'deleted': {},
        'source_layout': 'single_lowercase_word_modules_with_preserved_python_special_files'}
    receipt = method/'configs/names.json'
    originals = json.loads(receipt.read_text(encoding='utf-8')).get('original_sha256', {}) if receipt.is_file() else {}
    for name in sorted(set(names)):
        if not name or any(part.startswith('.evocond-backup') or part in {'.git', '__pycache__', '.pytest_cache'} for part in Path(name).parts):
            continue
        relative = Path(name)
        source = safe_path(official, name)
        tracked = subprocess.run(prefix+["cat-file", "-e", f"HEAD:{name}"], capture_output=True).returncode == 0
        base = git("show", f"HEAD:{name}") if tracked else None
        if not source.exists():
            if base is None:
                raise ValueError('Missing source has no pinned upstream original')
            manifest['deleted'][name] = dict(base_sha256=sha256(base).hexdigest(),
                original_sha256=originals.get('baselines/SEARL-official/'+name))
            continue
        if not source.is_file() or source.is_symlink():
            raise ValueError('Only ordinary source files can be packaged')
        current = source.read_bytes()
        if source.suffix == '.sh' and b'\r' in current:
            raise ValueError('Linux launch scripts must use LF line endings before packaging')
        manifest["files"][name] = {"sha256": sha256(current).hexdigest(),
                                    "base_sha256": sha256(base).hexdigest() if base is not None else None}
        entries["official_overlay/"+relative.as_posix()] = current
    for source in sorted(method.rglob("*")):
        rel = source.relative_to(method)
        if (source.is_file() and source.suffix in (".py", ".json", ".yaml", ".yml") and
                not set(rel.parts) & {"__pycache__", "runs", "assets", ".cache", ".git", ".test-tmp", "baselines", "archives"} and
                not any(part.startswith("tmp") for part in rel.parts[:-1])):
            if source.is_symlink():
                raise ValueError("No source symlinks in package")
            entries["EvoCond/"+rel.as_posix()] = source.read_bytes()
    entries["docs/EvoCond_Procedure.md"] = (workspace / "EvoCond/procedure/EvoCond_Procedure.md").read_bytes()
    entries["docs/EvoCond.md"] = (workspace / "EvoCond/writing/EvoCond.md").read_bytes()
    for source in sorted((workspace/'EvoCond/procedure').glob('implementation-*.md')):
        entries['docs/'+source.name] = source.read_bytes()
    entries["official_overlay/manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode()
    inventory = {name: sha256(data).hexdigest() for name, data in entries.items()}
    entries["PACKAGE.json"] = json.dumps({"format": "evocond_source_package_v1", "upstream_commit": PIN,
        "files": inventory, "excludes": ["model weights", "datasets", "runs", "LoRA adapters"],
        "gpu_verified": False, "paper_reproduction_claim": False}, indent=2, sort_keys=True).encode()
    with ZipFile(output, "x", ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    with ZipFile(output) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(entries):
            raise RuntimeError("Package ZIP verification failed")
        for name, data in entries.items():
            if archive.read(name) != data:
                raise RuntimeError(f"Archive content differs: {name}")
    return {"path": str(output), "files": len(entries), "overlay_files": len(manifest["files"]), 'retired_source_paths': len(manifest['deleted']),
            "bytes": output.stat().st_size, "sha256": sha256(output.read_bytes()).hexdigest()}
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("../.."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.workspace.resolve(), args.output.resolve()), indent=2))

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
