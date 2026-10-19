# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'CPU-only pinned SEARL upstream asset preparation; never loads or trains models.'
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.request import Request, urlopen
GIB = 1024 ** 3
BLOCK = 1024 ** 2
DEFAULT_ASSETS = ("datasets", "evaluation", "retriever", "judge")
CATALOGUE = json.loads('\n[\n  {\n    "folder": "data/toolstar_rl10k",\n    "key": "toolstar_rl10k",\n    "repo": "dongguanting/Multi-Tool-RL-10K",\n    "category": "datasets",\n    "type": "datasets",\n    "revision": "c9574c31986db590a322d8924925a8b9684fb6c5",\n    "files": [\n      {\n        "name": "README.md",\n        "size": 318,\n        "hash": "gitblob",\n        "checksum": "992f67772f33f98311a93e9b6cb8aea5bc010b74"\n      },\n      {\n        "name": "grpo_mix_train.parquet",\n        "size": 1266096,\n        "hash": "sha256",\n        "checksum": "d59eeee5692d95b7730b4f6bbdce94e2480c7d19abec47f6d7b1cc4e40ef70ec"\n      }\n    ]\n  },\n  {\n    "folder": "data/arpo_rl10k",\n    "key": "arpo_rl10k",\n    "repo": "dongguanting/ARPO-RL-Reasoning-10K",\n    "category": "datasets",\n    "type": "datasets",\n    "revision": "bf575746c7e7d2b844ef8e2a2613de2ccfcee016",\n    "files": [\n      {\n        "name": "README.md",\n        "size": 4740,\n        "hash": "gitblob",\n        "checksum": "fcda356ba8b0b8dec1f25e982b1ae59addd7a909"\n      },\n      {\n        "name": "train_10k.parquet",\n        "size": 2793322,\n        "hash": "sha256",\n        "checksum": "d1268cd800b821d7fa2913766beb81db342e4b839cbb6ba50a6549d5d8d1851b"\n      }\n    ]\n  },\n  {\n    "key": "arpo_evaluation",\n    "type": "github",\n    "repo": "RUC-NLPIR/ARPO",\n    "revision": "4d19a746b631b93cb382bdaa20fdeff272ed7361",\n    "folder": "data/arpo_evaluation",\n    "category": "evaluation",\n    "files": [\n      {\n        "name": "2wiki/test.jsonl",\n        "source": "evaluation/data/2wiki/test.jsonl",\n        "size": 31675,\n        "hash": "gitblob",\n        "checksum": "3dde384d249d7e657e11cf3a7f0f3075a9da08c3"\n      },\n      {\n        "name": "aime24/test.jsonl",\n        "source": "evaluation/data/aime24/test.jsonl",\n        "size": 156944,\n        "hash": "gitblob",\n        "checksum": "bb72e176e8d30c09cd67037e17cfc41e6ff48f70"\n      },\n      {\n        "name": "bamboogle/test.jsonl",\n        "source": "evaluation/data/bamboogle/test.jsonl",\n        "size": 13878,\n        "hash": "gitblob",\n        "checksum": "04c8452493564e62a6e5dec5bbf5437856091f61"\n      },\n      {\n        "name": "gsm8k/test.jsonl",\n        "source": "evaluation/data/gsm8k/test.jsonl",\n        "size": 761421,\n        "hash": "gitblob",\n        "checksum": "3e6e62b831b5fcbf0585d0dce75ccc5e4ccb691c"\n      },\n      {\n        "name": "hotpotqa/test.jsonl",\n        "source": "evaluation/data/hotpotqa/test.jsonl",\n        "size": 28176,\n        "hash": "gitblob",\n        "checksum": "69501f9958f7b51d0dae27c5a885c204a0a8e8b4"\n      },\n      {\n        "name": "math500/test.jsonl",\n        "source": "evaluation/data/math500/test.jsonl",\n        "size": 119567,\n        "hash": "gitblob",\n        "checksum": "953000dc487bf7549c12d737f873952d3b10261f"\n      },\n      {\n        "name": "musique/test.jsonl",\n        "source": "evaluation/data/musique/test.jsonl",\n        "size": 32010,\n        "hash": "gitblob",\n        "checksum": "69e3c587f21dd75a01476f608c329212d0d69bb8"\n      },\n      {\n        "name": "webwalker/test.jsonl",\n        "source": "evaluation/data/webwalker/test.jsonl",\n        "size": 48314,\n        "hash": "gitblob",\n        "checksum": "d641d0869f5bade41442b317d730e6c293ba94dc"\n      }\n    ]\n  },\n  {\n    "folder": "models/e5-base-v2",\n    "key": "e5",\n    "repo": "intfloat/e5-base-v2",\n    "category": "retriever",\n    "type": "models",\n    "revision": "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",\n    "files": [\n      {\n        "name": "1_Pooling/config.json",\n        "size": 200,\n        "hash": "gitblob",\n        "checksum": "2a824fa43c359360ecddb16808b539c6be892fef"\n      },\n      {\n        "name": "README.md",\n        "size": 67561,\n        "hash": "gitblob",\n        "checksum": "8129f8744fe3605534961d7585b4826052818e49"\n      },\n      {\n        "name": "config.json",\n        "size": 650,\n        "hash": "gitblob",\n        "checksum": "24b4776266ab6abf7a9d6672f42b77f06ef77b87"\n      },\n      {\n        "name": "model.safetensors",\n        "size": 437955512,\n        "hash": "sha256",\n        "checksum": "d0d559c47d5f71b1d280b13b62a2657f3e3bc70c0786f9ab91a36545e6a8f693"\n      },\n      {\n        "name": "modules.json",\n        "size": 387,\n        "hash": "gitblob",\n        "checksum": "ac2039abdf6ff023b27c919bb9675cfe378cb10f"\n      },\n      {\n        "name": "sentence_bert_config.json",\n        "size": 57,\n        "hash": "gitblob",\n        "checksum": "4eca68d85ecd3034cf4174d8a4033a75344ea62d"\n      },\n      {\n        "name": "special_tokens_map.json",\n        "size": 125,\n        "hash": "gitblob",\n        "checksum": "a8b3208c2884c4efb86e49300fdd3dc877220cdf"\n      },\n      {\n        "name": "tokenizer.json",\n        "size": 711396,\n        "hash": "gitblob",\n        "checksum": "688882a79f44442ddc1f60d70334a7ff5df0fb47"\n      },\n      {\n        "name": "tokenizer_config.json",\n        "size": 314,\n        "hash": "gitblob",\n        "checksum": "8046d4d4aff4529a10e0ad47043bdabd6304fb58"\n      },\n      {\n        "name": "vocab.txt",\n        "size": 231508,\n        "hash": "gitblob",\n        "checksum": "fb140275c155a9c7c5a3b3e0e77a9e839594a938"\n      }\n    ]\n  },\n  {\n    "folder": "models/Qwen3-32B",\n    "key": "judge",\n    "repo": "Qwen/Qwen3-32B",\n    "category": "judge",\n    "type": "models",\n    "revision": "9216db5781bf21249d130ec9da846c4624c16137",\n    "files": [\n      {\n        "name": "LICENSE",\n        "size": 11343,\n        "hash": "gitblob",\n        "checksum": "6634c8cc3133b3848ec74b9f275acaaa1ea618ab"\n      },\n      {\n        "name": "README.md",\n        "size": 16636,\n        "hash": "gitblob",\n        "checksum": "d8d96d1482cb71e6b8fa6b8109fe3b449e7cc5d5"\n      },\n      {\n        "name": "config.json",\n        "size": 728,\n        "hash": "gitblob",\n        "checksum": "d66d65fbc7960c2b3c254293f74df73b47fef6d3"\n      },\n      {\n        "name": "generation_config.json",\n        "size": 239,\n        "hash": "gitblob",\n        "checksum": "20a8a9156fc8c3f25295ca067f61fdf120d517c5"\n      },\n      {\n        "name": "merges.txt",\n        "size": 1671853,\n        "hash": "gitblob",\n        "checksum": "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7"\n      },\n      {\n        "name": "model-00001-of-00017.safetensors",\n        "size": 3957109648,\n        "hash": "sha256",\n        "checksum": "52562b2ff97b61764260273e71bf5b4cf8a66f569399398f26dec0300fcf1316"\n      },\n      {\n        "name": "model-00002-of-00017.safetensors",\n        "size": 3900791760,\n        "hash": "sha256",\n        "checksum": "e26764b2c6878e3fb7198895fa833ec62838d84a19665e6abfbae43c6daf02b3"\n      },\n      {\n        "name": "model-00003-of-00017.safetensors",\n        "size": 3900791760,\n        "hash": "sha256",\n        "checksum": "6c5ba7bed9c52bc121e75cbe8a7be46936d0006cc80f42a6d5886ed40b4c2a62"\n      },\n      {\n        "name": "model-00004-of-00017.safetensors",\n        "size": 3900791800,\n        "hash": "sha256",\n        "checksum": "f736f6ac4d8c30866107fb1185a05b3c3cfce9717720082f466fa44e691bcec8"\n      },\n      {\n        "name": "model-00005-of-00017.safetensors",\n        "size": 3900791800,\n        "hash": "sha256",\n        "checksum": "a52ed375c083209c54d42ac510afeb1fbb5af4f193be2dc7d103f665a0f212d3"\n      },\n      {\n        "name": "model-00006-of-00017.safetensors",\n        "size": 3900791800,\n        "hash": "sha256",\n        "checksum": "37fae28990b0e4a70228549d040c0393e87bee3820d59e58e47844974d8dff5b"\n      },\n      {\n        "name": "model-00007-of-00017.safetensors",\n        "size": 3900791800,\n        "hash": "sha256",\n        "checksum": "37776006aeaba29eca8bc73b2b963fe3477e1c2e3f6a27cb9527be75b905e1bf"\n      },\n      {\n        "name": "model-00008-of-00017.safetensors",\n        "size": 3900791800,\n        "hash": "sha256",\n        "checksum": "73e74e9129674fe330948005075d70ab4fa0b92b68fb220c5a693f9cea553730"\n      },\n      {\n        "name": "model-00009-of-00017.safetensors",\n        "size": 3900791800,\n        "hash": "sha256",\n        "checksum": "a044b3602a01bd8ea62ff51badf9cc038ab1d73d97399480e6a55b4c86fa7fa6"\n      },\n      {\n        "name": "model-00010-of-00017.safetensors",\n        "size": 3900791800,\n        "hash": "sha256",\n        "checksum": "9966612ba7ecfc2cd2e592fb95224b86d743271ff88e172cc272a4b26382aa75"\n      },\n      {\n        "name": "model-00011-of-00017.safetensors",\n        "size": 3900791800,\n        "hash": "sha256",\n        "checksum": "e2a058a0ac7d4b992b731c29221ecfb4b76b8a48d9004d0e8a62ba44f699845c"\n      },\n      {\n        "name": "model-00012-of-00017.safetensors",\n        "size": 3900791800,\n        "hash": "sha256",\n        "checksum": "58a1aa89093fea07325f787072a468e3482a470ff4b7fe5ead5f749683907c40"\n      },\n      {\n        "name": "model-00013-of-00017.safetensors",\n        "size": 3900791800,\n        "hash": "sha256",\n        "checksum": "35f3381bab31a23370c37d922290aeecdf603418336058fb86fe42d8f51ac40c"\n      },\n      {\n        "name": "model-00014-of-00017.safetensors",\n        "size": 3900791800,\n        "hash": "sha256",\n        "checksum": "8713b062ddc178acf5917610b7f4b64eede833b2ea4aa37bd562dcf2f3a3339d"\n      },\n      {\n        "name": "model-00015-of-00017.safetensors",\n        "size": 3900791800,\n        "hash": "sha256",\n        "checksum": "bec439d23931821a236d8f62fa79deecf5551bd25602278aa2ae0ce432b378cf"\n      },\n      {\n        "name": "model-00016-of-00017.safetensors",\n        "size": 3900791800,\n        "hash": "sha256",\n        "checksum": "e569139fadd61fe7c8f9eb1c976d9a627cae48c57ddf228cfbd0593c59c64ff7"\n      },\n      {\n        "name": "model-00017-of-00017.safetensors",\n        "size": 3055341992,\n        "hash": "sha256",\n        "checksum": "1f47c318fcd7797c0f85b4233cb754438b10e795b8bc874889090c416a94bd38"\n      },\n      {\n        "name": "model.safetensors.index.json",\n        "size": 58330,\n        "hash": "gitblob",\n        "checksum": "2bc093f8069057e0b28f628398a05a273efd1297"\n      },\n      {\n        "name": "tokenizer.json",\n        "size": 11422654,\n        "hash": "sha256",\n        "checksum": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"\n      },\n      {\n        "name": "tokenizer_config.json",\n        "size": 9732,\n        "hash": "gitblob",\n        "checksum": "417d038a63fa3de29cfde265caedae14d1a58d92"\n      },\n      {\n        "name": "vocab.json",\n        "size": 2776833,\n        "hash": "gitblob",\n        "checksum": "4783fe10ac3adce15ac8f358ef5462739852c569"\n      }\n    ]\n  },\n  {\n    "folder": "search/wiki18",\n    "key": "wiki_index",\n    "repo": "PeterJinGo/wiki-18-e5-index",\n    "category": "wiki",\n    "type": "datasets",\n    "revision": "a4d31160a035f30764604f4827cd8f1d0315eb86",\n    "files": [\n      {\n        "name": "part_aa",\n        "size": 42949672960,\n        "hash": "sha256",\n        "checksum": "a8a6a246951da4bbc8771a223283ef61963882a32864d9044ec00abb90fc3023"\n      },\n      {\n        "name": "part_ab",\n        "size": 21609402413,\n        "hash": "sha256",\n        "checksum": "b6d9bc943626fe7cb44de4c849e9379e7f272ab216c0552acbcf2390cc033c11"\n      }\n    ]\n  },\n  {\n    "folder": "search/wiki18",\n    "key": "wiki_corpus",\n    "repo": "PeterJinGo/wiki-18-corpus",\n    "category": "wiki",\n    "type": "datasets",\n    "revision": "69c1c00ffe7c5554c68d8548355cb22e46aabc51",\n    "files": [\n      {\n        "name": "wiki-18.jsonl.gz",\n        "size": 5123307260,\n        "hash": "sha256",\n        "checksum": "7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db"\n      }\n    ]\n  }\n]\n')
def safe_path(root, relative):
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Unsafe asset path")
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Asset path escapes root through a symlink")
    return target
def selected_specs(categories):
    return [spec for spec in CATALOGUE if spec["category"] in categories]
def iter_files(specs):
    for spec in specs:
        for entry in spec["files"]:
            yield spec, entry
def partial_path(root, spec, entry):
    identity = json.dumps([spec["repo"], spec["revision"], entry["name"]]).encode()
    return safe_path(root, ".downloads/" + hashlib.sha256(identity).hexdigest() + ".part")
def verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    digest = hashlib.sha256() if entry["hash"] == "sha256" else hashlib.sha1(usedforsecurity=False)
    if entry["hash"] == "gitblob":
        digest.update(("blob " + str(entry["size"]) + "\0").encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest() == entry["checksum"]
def free_bytes(root):
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    return shutil.disk_usage(ancestor).free
def plan(root, specs, reserve_gib):
    records = []
    missing_bytes = 0
    for spec, entry in iter_files(specs):
        target = safe_path(root, spec["folder"] + "/" + entry["name"])
        partial = partial_path(root, spec, entry)
        if target.exists() and not target.is_file():
            raise ValueError("Non-file at asset target: " + str(target))
        if target.exists() and target.stat().st_size != entry["size"]:
            raise ValueError("Existing file differs in size; preserved: " + str(target))
        present = target.is_file()
        offset = partial.stat().st_size if partial.is_file() else 0
        if offset > entry["size"]:
            raise ValueError("Oversized partial download; preserved: " + str(partial))
        needed = 0 if present else entry["size"] - offset
        missing_bytes += needed
        records.append({"path": str(target), "size": entry["size"],
                        "state": "present_unverified" if present else "partial" if offset else "missing",
                        "remaining_bytes": needed})
    available = free_bytes(root)
    return {"root": str(root), "files": records,
            "remaining_GiB": round(missing_bytes / GIB, 3),
            "free_GiB": round(available / GIB, 3), "reserve_GiB": reserve_gib,
            "space_ok": available >= missing_bytes + int(reserve_gib * GIB),
            "gpu_required": False, "model_loaded": False, "training_started": False,
            "data_kind": "pinned upstream sources, NOT final SEARL processed splits",
            "wiki_note": "Optional wiki files are NOT merged or decompressed by this helper"}
def source_url(spec, entry, mirror):
    source = entry.get("source", entry["name"])
    if spec["type"] == "github":
        return "https://raw.githubusercontent.com/" + spec["repo"] + "/" + spec["revision"] + "/" + source
    prefix = "https://hf-mirror.com/" if mirror else "https://huggingface.co/"
    if spec["type"] == "datasets":
        prefix += "datasets/"
    return prefix + spec["repo"] + "/resolve/" + spec["revision"] + "/" + source + "?download=true"
@contextmanager
def exclusive_run(root):
    lock = safe_path(root, '.assets.lock')
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise ValueError("Another downloader may be running. Inspect before removing " + str(lock))
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()) + "\n")
        yield
    finally:
        lock.unlink()
def download_one(root, spec, entry, mirror=False, reserve_gib=10, opener=urlopen):
    target = safe_path(root, spec["folder"] + "/" + entry["name"])
    if target.exists():
        print("VERIFY " + str(target), flush=True)
        if not verify(target, entry):
            raise ValueError("Existing checksum mismatch; file preserved: " + str(target))
        return
    partial = partial_path(root, spec, entry)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial.parent.mkdir(parents=True, exist_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > entry["size"]:
        raise ValueError("Oversized partial; preserved: " + str(partial))
    if offset < entry["size"]:
        headers = {"User-Agent": "searl-assets/1", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = "bytes=" + str(offset) + "-"
        request = Request(source_url(spec, entry, mirror), headers=headers)
        with opener(request, timeout=120) as response:
            status = response.status
            if status == 206:
                value = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value)
                if not match or int(match[1]) != offset or int(match[3]) != entry["size"] or int(match[2]) != entry["size"] - 1:
                    raise ValueError("Unexpected Content-Range; partial preserved")
            elif status == 200:
                if offset:
                    print("Server ignored Range; restarting only this helper's partial file", flush=True)
                offset = 0
            else:
                raise OSError("Unexpected download status " + str(status))
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != entry["size"] - offset:
                raise ValueError("Response length does not match pinned file; partial preserved")
            reclaimable = partial.stat().st_size if offset == 0 and partial.exists() else 0
            if free_bytes(root) + reclaimable < entry["size"] - offset + int(reserve_gib * GIB):
                raise ValueError("Insufficient disk space; download not started")
            received = offset
            last_report = time.monotonic()
            with partial.open("ab" if offset else "wb") as stream:
                while True:
                    block = response.read(min(BLOCK, entry["size"] - received + 1))
                    if not block:
                        break
                    if received + len(block) > entry["size"]:
                        raise ValueError("Response exceeded pinned file size")
                    stream.write(block)
                    received += len(block)
                    if time.monotonic() - last_report >= 20:
                        print(entry["name"] + ": " + str(round(100 * received / entry["size"], 1)) + "%", flush=True)
                        last_report = time.monotonic()
                stream.flush()
                os.fsync(stream.fileno())
            if received != entry["size"]:
                raise OSError("Incomplete download; rerun the same command to resume")
    print("CHECKSUM " + entry["name"], flush=True)
    if not verify(partial, entry):
        raise ValueError("Checksum mismatch; partial preserved: " + str(partial))
    # Atomic, exclusive hard-link promotion avoids a second full-size copy.
    try:
        os.link(partial, target)
    except (OSError, NotImplementedError):
        raise ValueError("Hard-link promotion failed; verified partial preserved at " + str(partial))
    partial.unlink()
    print("READY " + str(target), flush=True)
def prepare(root, specs, mirror, reserve_gib):
    initial = plan(root, specs, reserve_gib)
    if not initial["space_ok"]:
        raise ValueError("Insufficient free space for the selected assets plus reserve")
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_run(root):
        for spec in specs:
            for entry in spec["files"]:
                print("FETCH " + spec["key"] + "/" + entry["name"], flush=True)
                for attempt in range(3):
                    try:
                        download_one(root, spec, entry, mirror, reserve_gib)
                        break
                    except (OSError, TimeoutError) as error:
                        if attempt == 2:
                            raise
                        print("Network/IO retry " + str(attempt + 1) + " (" + type(error).__name__ + ")", flush=True)
                        time.sleep(2 ** attempt)
            receipt = safe_path(root, "manifests/" + spec["key"] + ".json")
            payload = dict(spec, all_checksums_verified=True)
            receipt.parent.mkdir(parents=True, exist_ok=True)
            if receipt.exists():
                previous = json.loads(receipt.read_text(encoding="utf-8"))
                if previous.get("revision") != spec["revision"]:
                    raise ValueError("Existing receipt revision differs; preserved")
            temporary = receipt.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            temporary.replace(receipt)
        print("ASSETS_READY: selected files verified; NO training or sandbox execution started", flush=True)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path('.'))
    parser.add_argument("--assets", nargs="+", choices=(*DEFAULT_ASSETS, "wiki"), default=DEFAULT_ASSETS)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--mirror", action="store_true", help="Use hf-mirror.com; pinned hashes remain mandatory")
    parser.add_argument("--reserve-gib", type=float, default=10)
    parser.add_argument("--verify", action="store_true", help="Hash present files without downloading")
    args = parser.parse_args()
    if not math.isfinite(args.reserve_gib) or args.reserve_gib < 0:
        parser.error("--reserve-gib must be finite and nonnegative")
    root = args.root.expanduser().resolve()
    specs = selected_specs(args.assets)
    print(json.dumps(plan(root, specs, args.reserve_gib), indent=2), flush=True)
    if args.run:
        prepare(root, specs, args.mirror, args.reserve_gib)
    elif args.verify:
        failed = False
        for spec, entry in iter_files(specs):
            target = safe_path(root, spec["folder"] + "/" + entry["name"])
            ok = verify(target, entry)
            print(("VERIFIED " if ok else "MISSING_OR_MISMATCH ") + str(target), flush=True)
            failed |= not ok
        if failed:
            raise ValueError("Not all selected assets are verified")
    else:
        print("READ_ONLY: add --run to download; no torch, pip, GPU, model load or training")
if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        # Do not print CDN redirect URLs or credentials from transport errors.
        print("STOP: " + (str(error) if isinstance(error, ValueError) else type(error).__name__ + "; retry the same command or check network/disk"), file=sys.stderr)
        sys.exit(1)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
