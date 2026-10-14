#!/usr/bin/env python3
"""Run a repository script with this isolated condition as its working directory."""
import os
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[2]
if len(sys.argv)<2:raise SystemExit('Repository-relative script required')
script=(root/sys.argv[1]).resolve()
if root not in script.parents:raise SystemExit('Script must be inside this condition')
os.chdir(root)
os.execv(sys.executable,[sys.executable,str(script),*sys.argv[2:]])
