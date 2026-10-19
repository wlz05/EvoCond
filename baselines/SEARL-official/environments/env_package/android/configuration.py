# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any

from environments.env_package.android.configurations import BaseEnvConfig

REPO_ROOT = Path(__file__).resolve().parents[4]

REPO_ROOT = Path(os.getenv("ANDROID_REPO_ROOT", REPO_ROOT))

DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_IMAGE_DIR = REPO_ROOT / "images"
DEFAULT_CACHE_DIR = REPO_ROOT / "cache"

@dataclass
class AndroidEnvConfig(BaseEnvConfig):

    prompt_format: str = "free_think"
    device_id: str = ""
    rollout_id: str = ""
    seed: int = 42
    down_sample_ratio: float = 1.0

    from pathlib import Path
    ROOT_DIR = Path(__file__).resolve().parents[4]
    data_path: str = str(ROOT_DIR / "data" / "task_metadata.jsonl")

    save_root: Path = DEFAULT_IMAGE_DIR
    cache_dir: Path = DEFAULT_CACHE_DIR

    base_url: str = os.getenv("ANDROID_BACKEND_URL", "http://10.197.146.67:8000")

    max_actions_per_step: int = 1

    use_orm: bool = True
    use_prm: bool = False
    reward_device: Dict[str, int] = field(
        default_factory=lambda: {"orm": 0, "prm": 1}
    )

    def resolve_dir(self) -> None:
        """Ensure directories exist before training starts."""
        self.save_root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def config_id(self) -> str:
        '\n        eg:\n        AndroidEnvConfig(env=androidWorld,prompt=free_think,seed=42,use_orm=True,use_prm=False)\n        '
        return (
            f"AndroidEnvConfig("
            f"env={self.env_name},"
            f"prompt={self.prompt_format},"
            f"seed={self.seed},"
            f"use_orm={self.use_orm},"
            f"use_prm={self.use_prm})"
        )

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
