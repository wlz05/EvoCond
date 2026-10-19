# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import subprocess
def reviewed(root, pin):
    root = Path(root).resolve()
    receipt = root/'source.json'
    if not receipt.is_file():
        version = subprocess.check_output(['git', '-c', f'safe.directory={root.as_posix()}', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        if version != pin:
            raise ValueError('Official checkout must use the reviewed pinned commit')
        return version
    value = json.loads(receipt.read_text(encoding='utf-8'))
    if value.get('format') != 'evocond_vendored_source_v1' or value.get('upstream_commit') != pin or not value.get('files'):
        raise ValueError('Vendored source provenance differs from the reviewed baseline')
    # Verify shipped source independently of the outer Git repository.
    for name, digest in value['files'].items():
        relative = PurePosixPath(name)
        path = root/name
        if relative.is_absolute() or '..' in relative.parts or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError('Invalid vendored source path')
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Vendored source changed: '+name)
    return pin

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
