# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations
import json
import platform
import shutil
from pathlib import Path
def main() -> None:
    import torch
    total, used, free = shutil.disk_usage(Path("."))
    cuda = torch.cuda.is_available()
    report = {"python": platform.python_version(), "torch": torch.__version__, "cuda": cuda, "device": torch.cuda.get_device_name(0) if cuda else "cpu", "vramgb": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2) if cuda else 0.0, "diskgb": round(total / 2 ** 30, 2), "freegb": round(free / 2 ** 30, 2), "cwd": Path(".").resolve().name}
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if not cuda or report["vramgb"] < 20.0:
        raise RuntimeError("A CUDA GPU with at least twenty gigabytes is required")
    if report["freegb"] < 60.0:
        raise RuntimeError("At least sixty gigabytes of free disk are required")
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
