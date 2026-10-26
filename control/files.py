# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Portable experiment paths and complete exclusive report publication.'
import os
from pathlib import Path, PureWindowsPath
import tempfile

def relative(root, name):
    if not isinstance(name, str) or not name or '\\' in name or PureWindowsPath(name).drive:
        raise ValueError('Use a nonempty portable relative path')
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Relative artifact paths cannot escape their declared root')
    base = Path(root).resolve()
    result = base/path
    if not result.resolve().is_relative_to(base):
        raise ValueError('Resolved artifact path leaves its declared root')
    return result

def publish(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.evocond-', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(text+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        # Linking publishes the complete artifact without replacing a prior result.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
