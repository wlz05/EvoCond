# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Resolve the reviewed checkout in both a developer tree and deployed package.'
import os
from pathlib import Path
def official_root():
    configured = os.environ.get("SEARL_OFFICIAL_ROOT")
    root = Path(configured) if configured else Path(__file__).resolve().parents[1] / "baselines" / "SEARL-official"
    if not root.is_dir():
        raise RuntimeError("Set SEARL_OFFICIAL_ROOT to the pinned, overlay-installed official checkout before testing")
    return root.resolve()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
