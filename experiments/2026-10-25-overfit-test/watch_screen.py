"""Watch a screen started by launch_screen.py: one line per meaningful event.

Prints progress every N iterations, every evaluation, new screen.log lines,
power pauses and resumes, error lines in the per-rate logs, and warnings for
low validity. Exits on SCREEN_DONE or when the screen's process group is gone;
reports a stall after 20 minutes without new records. Suited to a Monitor or
`tail`-free polling; output is line-buffered with -u.
Usage: python3 -u watch_screen.py SCREEN_DIR SCREEN_PGID [EVERY=4] [resume]
("resume" skips records already present, for re-arming a watcher.)
"""
import json, os, re, subprocess, sys, time
from pathlib import Path
root = Path(sys.argv[1]); pgid = int(sys.argv[2]); every = int(sys.argv[3]) if len(sys.argv) > 3 else 4
seen = {}; screen_lines = 0; baby_lines = 0; last_change = time.time(); err_seen = set()
ERR = re.compile(r"Traceback|Error|error:|Killed|nan|NaN|out of memory", re.I)
RESUME = len(sys.argv) > 4 and sys.argv[4] == "resume"
def say(s):
    if not MUTE: print(s, flush=True)
MUTE = RESUME
def alive():
    out = subprocess.run(["ps", "-eo", "pgid"], capture_output=True, text=True).stdout.split()
    return str(pgid) in out
while True:
    changed = False
    lines = (root / "screen.log").read_text().splitlines() if (root / "screen.log").exists() else []
    for l in lines[screen_lines:]:
        say(f"screen: {l}"); changed = True
    screen_lines = len(lines)
    if any(l.startswith("SCREEN_DONE") for l in lines):
        say("DONE"); break
    b = (root / "babysitter.log").read_text().splitlines() if (root / "babysitter.log").exists() else []
    for l in b[baby_lines:]: say(f"power: {l}")
    baby_lines = len(b)
    for run in sorted(root.glob("lr_*")):
        if not run.is_dir(): continue
        for name in ("iterations.jsonl", "evaluations.jsonl"):
            f = run / name
            if not f.exists(): continue
            recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
            key = (run.name, name); n0 = seen.get(key, 0)
            for r in recs[n0:]:
                changed = True
                if name == "evaluations.jsonl":
                    a = r.get("train/all", {}); p = r.get("probe/all", {})
                    say(f"{run.name} EVAL it{r['iteration']}: train p={a.get('mean_teacher_probability',0):.3f} top1={a.get('teacher_top_agreement',0):.3f} valid={a.get('valid_fraction',0):.2f} A-rate={a.get('first_displayed_option_rate',0):.2f} medReason={a.get('median_reasoning_tokens')} | probe p={p.get('mean_teacher_probability',0):.3f} top1={p.get('teacher_top_agreement',0):.3f}")
                else:
                    i = r["iteration"]; a = r.get("all", {})
                    bad = a.get("valid_fraction", 1) < 0.9 or not (r.get("gradient_norm_before_clip") or 0) < 1e6
                    if i % every == 0 or i == 1 or bad:
                        say(f"{run.name} it{i}: reward={r['mean_reward']:.3f} p={a.get('mean_teacher_probability',0):.3f} valid={a.get('valid_fraction',0):.2f} A-rate={a.get('first_displayed_option_rate',0):.2f} medReason={a.get('median_reasoning_tokens')} gnorm={r.get('gradient_norm_before_clip',0):.3g} flat={r.get('zero_variance_group_fraction',0):.2f} {r['iteration_seconds']:.0f}s{' WARN' if bad else ''}")
            seen[key] = len(recs)
        log = root / f"{run.name}.log"
        if log.exists():
            for l in log.read_text().splitlines():
                if ERR.search(l) and l not in err_seen:
                    err_seen.add(l); say(f"{run.name} LOG: {l[:300]}")
    MUTE = False
    if changed: last_change = time.time()
    if not alive():
        say("FAILED: screen process group is gone without SCREEN_DONE"); break
    if time.time() - last_change > 1200:
        say("STALL: no new records for 20 minutes"); last_change = time.time()
    time.sleep(20)
