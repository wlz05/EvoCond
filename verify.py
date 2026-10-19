# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'CPU test verifier; no models, datasets, remote services or GPU workloads.'
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
PROGRAM = "import json, sys, unittest\nsuite = unittest.defaultTestLoader.discover('tests', pattern=sys.argv[1], top_level_dir='.' if sys.argv[2] == 'package' else None)\nif not suite.countTestCases():\n    raise RuntimeError('No tests discovered; check the renamed test files')\nresult = unittest.TextTestRunner(stream=sys.stdout, verbosity=0).run(suite)\nprint('TEST_RECEIPT '+json.dumps(dict(tests=result.testsRun, failures=len(result.failures),\n                                  errors=len(result.errors), skipped=len(result.skipped))))\nsys.exit(0 if result.wasSuccessful() else 1)\n"
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-root', type=Path, required=True)
    args = parser.parse_args()
    method = Path(__file__).resolve().parent
    official = args.official_root.resolve()
    if not (official/'envs').is_dir() or not (official/'tests').is_dir():
        parser.error('Expected the installed official source and overlay tests')
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(method)+os.pathsep+environment.get('PYTHONPATH', '')
    reports = []
    for label, directory in (('method', method), ('official_overlay', official)):
        pattern = '*.py'
        mode = 'package' if label == 'official_overlay' else 'flat'
        result = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-c', PROGRAM, pattern, mode], cwd=directory,
                                env=environment, capture_output=True, text=True, encoding='utf-8')
        receipts = [line.removeprefix('TEST_RECEIPT ') for line in result.stdout.splitlines()
                    if line.startswith('TEST_RECEIPT ')]
        reports.append({'suite': label, 'exit_code': result.returncode,
                        'receipt': json.loads(receipts[0]) if len(receipts) == 1 else None})
        if result.returncode != 0 or len(receipts) != 1:
            print(result.stdout[-4000:])
            print(result.stderr[-4000:])
    print(json.dumps({'gpu_started': False, 'suites': reports}, indent=2))
    raise SystemExit(int(any(r['exit_code'] != 0 or r['receipt'] is None for r in reports)))
if __name__ == '__main__':
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
