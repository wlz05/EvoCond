# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Verify and unpack a source package into a new directory without overwriting assets.'
import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import unicodedata
from zipfile import ZipFile
from control.contracts import strict_json
LIMIT = 256*1024*1024
def relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ':' in name or '\\' in name or path.as_posix() != name:
        raise ValueError('Invalid package path')
    if any(part in ('.', '..', '') for part in path.parts):
        raise ValueError('Package path escapes destination')
    return path
def verify(archive):
    with ZipFile(archive) as bundle:
        entries = bundle.infolist()
        names = [item.filename for item in entries]
        normalized = [unicodedata.normalize('NFKC', name).casefold() for name in names]
        if len(set(normalized)) != len(names) or sum(item.file_size for item in entries) > LIMIT:
            raise ValueError('Duplicate portable path or oversized source package')
        if any(item.is_dir() or (item.external_attr >> 16) & 0o170000 == 0o120000 for item in entries):
            raise ValueError('Package directories and symlinks are not supported')
        for name in names:
            relative(name)
        if 'PACKAGE.json' not in names:
            raise ValueError('Source inventory is missing')
        manifest = strict_json(bundle.read('PACKAGE.json').decode('utf-8-sig'), max_text=LIMIT)
        if manifest.get('format') != 'evocond_source_package_v1' or manifest.get('gpu_verified') is not False:
            raise ValueError('Unknown source package format or verification claim')
        inventory = manifest.get('files')
        if not isinstance(inventory, dict) or set(inventory) != set(names)-{'PACKAGE.json'}:
            raise ValueError('Source inventory does not exactly match archive entries')
        contents = {}
        for name in names:
            content = bundle.read(name)
            if name != 'PACKAGE.json' and sha256(content).hexdigest() != inventory[name]:
                raise ValueError(f'Package checksum mismatch: {name}')
            if name.endswith('.py'):
                stem = PurePosixPath(name).stem
                if stem not in ('__init__', '__main__') and re.fullmatch('[a-z]+', stem) is None:
                    raise ValueError(f'Invalid Python module filename: {name}')
                ast.parse(content.decode('utf-8-sig'), filename=name)
            if name.endswith('.sh') and b'\r' in content:
                raise ValueError('Linux source scripts must use LF line endings')
            contents[name] = content
        return contents
def unpack(archive, output, run=False):
    contents = verify(archive)
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError('Use a new destination; existing source and assets are never overwritten')
    result = dict(verified=True, extracted=run, files=len(contents), gpu_started=False,
                  archive_sha256=sha256(Path(archive).read_bytes()).hexdigest())
    if not run:
        return result
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.evocond-unpack-', dir=destination.parent))
    try:
        for name, content in contents.items():
            target = staging.joinpath(*relative(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(content)
        staging.rename(destination)
    finally:
        if staging.exists():
            if staging.resolve().parent != destination.parent or not staging.name.startswith('.evocond-unpack-'):
                raise ValueError('Refuse cleanup outside the validated staging directory')
            shutil.rmtree(staging)
    return result
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    print(json.dumps(unpack(args.archive, args.output, args.run)))
if __name__ == '__main__':
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
