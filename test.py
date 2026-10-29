# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

# Run renamed test modules without starting training or loading models.
import argparse
import os
from pathlib import Path
import tempfile
import unittest
from namespace import install
install()
def main():
    parser = argparse.ArgumentParser(description="Run maintained tests; use --pattern to select a named module.")
    parser.add_argument("--pattern", default="*.py")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    folder = Path(".test-tmp")
    folder.mkdir(exist_ok=True)
    tempfile.tempdir = str(folder.resolve())
    os.environ["TEMP"] = os.environ["TMP"] = tempfile.tempdir
    suite = unittest.defaultTestLoader.discover("tests", pattern=args.pattern)
    if not suite.countTestCases():
        raise RuntimeError("No tests discovered; test filenames no longer use the test_ prefix")
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    raise SystemExit(not result.wasSuccessful())
if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
