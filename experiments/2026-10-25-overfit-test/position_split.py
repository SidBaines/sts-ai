"""Split teacher top-choice agreement by whether the teacher's top action was displayed as A.

A policy choosing by content agrees equally often in both cells; a position
shortcut agrees far more when the top action is shown first.
Usage: python position_split.py RUN_DIR STATES_JSONL [ITERATIONS_PER_PASS]
"""
import json
import sys
from pathlib import Path

run, states = Path(sys.argv[1]), Path(sys.argv[2])
per_pass = int(sys.argv[3]) if len(sys.argv) > 3 else 8
top = {}
for line in states.open():
    record = json.loads(line)
    top[record["id"]] = record["teacher_top"]
print("pass  top shown as A: agree/n   top shown elsewhere: agree/n   chose A")
cells = {}
for path in sorted((run / "generations").glob("iter_*.jsonl")):
    for line in path.open():
        g = json.loads(line)
        k = (g["iteration"] - 1) // per_pass + 1
        cell = cells.setdefault(k, [0, 0, 0, 0, 0, 0])
        at_a = g["order"][0] == top[g["id"]]
        agree = bool(g["valid"] and g["action_index"] == top[g["id"]])
        cell[0 if at_a else 2] += agree
        cell[1 if at_a else 3] += 1
        cell[4] += bool(g["valid"] and g["label"] == "A")
        cell[5] += 1
for k, c in sorted(cells.items()):
    print(f"{k:>4}  {c[0]:>4}/{c[1]:<4} ({c[0] / max(c[1], 1):.2f})"
          f"            {c[2]:>4}/{c[3]:<4} ({c[2] / max(c[3], 1):.2f})          {c[4] / c[5]:.2f}")
