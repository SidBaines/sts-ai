#!/usr/bin/env python3
"""Read-only live dashboard for data/rl_runs (http://127.0.0.1:8791).

    dashboard.py serve [--port 8791] [--root DIR ...]    run in foreground
    dashboard.py start [--port 8791] [--root DIR ...]    launch detached (idempotent)
    dashboard.py restart [--port 8791] [--root DIR ...]  stop that port, then start

For a separate port and worktree:
    dashboard.py start --port 8795 --root /other/worktree/data/rl_runs

The default data/rl_runs and every extra --root are read together, including
each queue and runner heartbeat. Runs may be direct children or nested two
directories deep, named relative to their root (e.g. batch/recipe).

What it shows, most important first:
  1. Win rate on the 4,096-world win set (data/rl_runs/win_eval/*.json), with
     Wilson 95% intervals. The objective is an A0 Act 3 boss win.
  2. Fresh-world confirmations (data/rl_runs/fresh_eval/*.json).
  3. Jobs that are running, pending or paused now, with trainer health and ETA
     from the last three iteration times (including any pauses).
  4. Training runs: the key lineage (RUN_INFO) and anything active by default;
     every other run is "archived" behind a toggle.
  5. Charts for the selected runs: floors, wins, boss outcomes, behaviour,
     gradient cosine, entropy, KL, value loss, iteration/collect/update time,
     and collection versus evaluation win rates.

Labels for runs and eval files live in RUN_INFO and EVAL_LABELS below. Add an
entry when a run or eval matters. Unlisted runs are archived, and unlisted eval
files show their file name.

/api/state?since=<sig>&max_points=N omits the heavy fields when no log or eval
file has changed since <sig>, so an idle poll is ~1-2 KB.
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "rl_runs"
PORT = 8791

# Key runs, shown by default (in this order). Everything else is archived unless active.
RUN_INFO = {
    "large_batch_20261020/recipe": "Large batch: champion recipe at 3,072 games/iteration (potions visible out of combat during collection and eval)",
    "large_batch_20261020/recipe_hidden": "Large batch: same recipe with --hide-ooc-potions (deployment protocol)",
    "large_batch_20261020/hygiene_hidden": "Large batch: recipe + hide-ooc-potions + detached critic, normalised returns, strategic entropy 0.01",
    "ooc_r2_lam1": "Round 2: smith + elite distilled → PPO λ=1 (current best lineage)",
    "ooc_pfd_lam1": "Smith distilled → PPO λ=1 (keeps smithing, improves)",
    "ooc_pfd_lam099": "Smith distilled → PPO λ=0.99",
    "ooc_ppo_from_distilled": "Smith distilled → PPO λ=0.95 (unlearns smithing)",
    "ooc_main_s1": "E8 OOC PPO from original, seed 1 (λ=0.95)",
    "ooc_rep_s2": "E8 replicate, seed 2",
}
EVAL_LABELS = {
    "distilled": "Update 36 + smith rule distilled",
    "distilled_elite90": "Update 36 + smith distilled, + elite ≥90% overlay",
    "distilled_elite80": "Update 36 + smith distilled, + elite ≥80% overlay",
    "lam1_i12": "Smith distilled → PPO λ=1, 12 updates",
    "r2_lam1i12": "Round 2 start: smith + elite distilled",
    "r2_avoid_fh_cd": "Round 2 start, + avoid Fusion Hammer / Coffee Dripper overlay",
    "r2ppo_i4": "Round 2 → PPO λ=1, 4 updates",
    "r2ppo_i4_eliteact": "Round 2 → PPO λ=1, 4 updates, + per-act elite overlay",
    "r2ppo_i12": "Round 2 → PPO λ=1, 12 updates",
    "r2ppo_i16": "Round 2 → PPO λ=1, 16 updates",
    "r2_search_a23_s64_n2048": "Round 2 start + public search at Act 2/3 bosses (play-time search)",
    "orig": "Original checkpoint",
    "i36": "E8 OOC PPO, update 36",
}
KEEP_TRAIN = ("iteration", "collect_floor", "collect_wins", "games", "grad_cosine_half", "approx_kl", "entropy",
              "smith_rate", "map_elite", "card_skip_rate", "shop_potion", "fights_per_run", "iteration_seconds",
              "collect_win", "faults", "value_loss", "clip_fraction", "early_stop", "collector_restarts",
              "update_seconds", "collect_seconds", "ooc_choice_transitions", "return_scale", "steps")
KEEP_EVAL = ("iteration", "n", "mean_floor", "se", "wins", "act1_boss_pass", "act2_boss_pass", "act3_boss_reach",
             "act3_boss_pass", "paired", "act2_boss_reach", "faults", "collector_restarts")
ACTIVE = ("running", "pending", "paused")
_CACHE: dict[Path, tuple[tuple[int, int], object]] = {}


def _cached(path: Path, parse):
    st = path.stat()
    key = (st.st_mtime_ns, st.st_size)
    hit = _CACHE.get(path)
    if hit is not None and hit[0] == key:
        return key, hit[1]
    value = parse(path)
    _CACHE[path] = (key, value)
    return key, value


def _read_json(path: Path, default):
    """A file the runner may be rewriting in place; a torn read yields the default."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _parse_log(log: Path) -> dict:
    train, evals, config, done = [], [], {}, False
    for line in log.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = r.get("event")
        if event in ("start", "resume"):
            if r.get("config"):
                config = r["config"]  # a resume may change iterations (extended runs)
            done = False
        elif event == "train":
            train.append({k: r.get(k) for k in KEEP_TRAIN if k in r})
        elif event == "eval":
            evals.append({k: r.get(k) for k in KEEP_EVAL if k in r})
        elif event == "done":
            done = True
    return {"config": {k: config[k] for k in ("iterations", "games", "notes", "gae_lambda", "seed", "checkpoint",
                                             "hide_ooc_potions") if k in config},
            "train": train, "evals": evals, "done": done, "mtime": log.stat().st_mtime}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _parse_eval(path: Path) -> dict:
    s = _read_json(path, {}).get("summary") or {}
    n, wins = int(s.get("n") or 0), int(s.get("wins") or 0)
    lo, hi = wilson(wins, n)
    return {"name": path.stem, "label": EVAL_LABELS.get(path.stem, path.stem), "n": n, "wins": wins,
            "rate": wins / n if n else None, "lo": lo, "hi": hi, "mean_floor": s.get("mean_floor"),
            "se": s.get("se"), "act2_boss_pass": s.get("act2_boss_pass"), "act3_boss_reach": s.get("act3_boss_reach"),
            "act3_boss_pass": s.get("act3_boss_pass"), "mode": s.get("mode"),
            "mtime": path.stat().st_mtime}


def _eval_table(directory: Path, sig: list[str]) -> list[dict]:
    rows = []
    for path in sorted(directory.glob("*.json")) if directory.exists() else []:
        try:
            key, row = _cached(path, _parse_eval)
        except FileNotFoundError:
            continue
        sig.append(f"{path}:{key[0]}:{key[1]}")
        if row["n"]:
            rows.append(row)
    return sorted(rows, key=lambda r: -(r["rate"] or 0))


def _thin(rows: list, cap: int) -> list:
    if len(rows) <= cap:
        return rows
    last = len(rows) - 1
    return [rows[round(i * last / (cap - 1))] for i in range(cap)]


def _out_dir(cmd: list[str], root: Path | None = None) -> str | None:
    if "--out" not in cmd or cmd.index("--out") + 1 == len(cmd):
        return None
    path = Path(cmd[cmd.index("--out") + 1]).expanduser()
    try:
        return path.resolve().relative_to((root or RUNS).resolve()).as_posix()
    except ValueError:
        return path.name


def _progress(run: dict, now: float) -> dict:
    """Use iteration numbers so resumed logs need not contain the whole history."""
    train = run["train"]
    done = max((r.get("iteration", 0) for r in train), default=0)
    total = run["config"].get("iterations")
    times = [r["iteration_seconds"] for r in train[-3:]
             if isinstance(r.get("iteration_seconds"), (int, float))
             and math.isfinite(r["iteration_seconds"]) and r["iteration_seconds"] > 0]
    mean = sum(times) / len(times) if times else None
    remaining = max(0, total - done) * mean if total is not None and mean is not None else None
    return {"iterations_done": done, "mean_iteration_seconds": mean, "remaining_seconds": remaining,
            "eta": now + remaining if remaining and not run["done"] else None,
            "last_train": train[-1] if train else None, "last_eval": run["evals"][-1] if run["evals"] else None}


def state(max_points: int | None = None, since: str | None = None, extra_roots=()) -> dict:
    now = time.time()
    roots = list(dict.fromkeys(Path(p).expanduser().resolve() for p in (RUNS, *extra_roots)))
    sig: list[str] = []
    runs: dict[str, dict] = {}
    run_keys = {}
    queue, runners, win, fresh = [], [], [], []
    for root in roots:
        sig.append(str(root))
        queue.extend((root, q) for q in _read_json(root / "queue.json", {"runs": []}).get("runs", []))
        runners.append({"root": str(root), "status": _read_json(root / "runner_status.json", None)})
        logs = sorted([*root.glob("*/log.jsonl"), *root.glob("*/*/log.jsonl")])
        for log in logs:
            name = log.parent.relative_to(root).as_posix()
            if any(part.startswith("_") for part in log.parent.relative_to(root).parts):
                continue
            try:
                key, run = _cached(log, _parse_log)
            except FileNotFoundError:
                continue
            sig.append(f"{log}:{key[0]}:{key[1]}")
            run_key = name if name not in runs else f"{name} [{root}]"
            run_keys[root, name] = run_key
            runs[run_key] = dict(run, name=name, root=str(root))
        win.extend(dict(row, root=str(root)) for row in _eval_table(root / "win_eval", sig))
        fresh.extend(dict(row, root=str(root)) for row in _eval_table(root / "fresh_eval", sig))

    active, history = [], {}
    for root, q in queue:
        if q.get("status") not in ACTIVE:
            status = q.get("status", "?")
            history[status] = history.get(status, 0) + 1
            continue
        out = _out_dir(q.get("cmd", []), root)
        run_key = run_keys.get((root, out))
        run = runs.get(run_key)
        progress = _progress(run, now) if run else {}
        active.append({"name": q["name"], "root": str(root), "status": q["status"], "lane": q.get("lane", "main"),
                       "elapsed": now - q["started"] if q.get("started") else None, "run": run_key,
                       "out": out, "progress": [progress["iterations_done"], run["config"].get("iterations")] if run else None,
                       **progress})
    active_runs = {a["run"] for a in active if a["run"]}
    statuses = {a["run"]: a["status"] for a in active if a["run"]}
    sig.append("active:" + json.dumps([(a["root"], a["name"], a["run"], a["status"]) for a in active]))
    state_sig = hashlib.sha1("\n".join(sig).encode()).hexdigest()[:16]
    out = {"now": now, "runner": runners[0]["status"], "runners": runners,
           "active": active, "history": history, "sig": state_sig,
           "runs": None, "win": None, "fresh": None}
    if since == state_sig:
        return out
    priority = {name: i for i, name in enumerate(RUN_INFO)}
    order = sorted(runs, key=lambda n: (priority.get(runs[n]["name"], len(priority)), n))
    shown = {}
    for name in order:
        run = runs[name]
        entry = {"name": run["name"], "root": run["root"],
                 "label": RUN_INFO.get(run["name"]) or (run["config"].get("notes") or "")[:90],
                 "key": run["name"] in RUN_INFO, "active": name in active_runs, "done": run["done"],
                 "status": statuses.get(name, "done" if run["done"] else "stopped"), "config": run["config"],
                 "iterations": run["config"].get("iterations"), "n_train": len(run["train"]),
                 "train": run["train"], "evals": run["evals"], **_progress(run, now)}
        if max_points is not None:
            entry["train"] = _thin(entry["train"], max_points)
        shown[name] = entry
    out.update(runs=shown, win=sorted(win, key=lambda r: -(r["rate"] or 0)),
               fresh=sorted(fresh, key=lambda r: -(r["rate"] or 0)))
    return out


def state_payload(path: str, extra_roots=()) -> bytes:
    query = parse_qs(urlsplit(path).query)
    since = query.get("since", [""])[0] or None
    try:
        max_points = max(2, int(query["max_points"][0])) if "max_points" in query else None
    except ValueError:
        max_points = None
    return json.dumps(state(max_points=max_points, since=since, extra_roots=extra_roots), separators=(",", ":")).encode()


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>StS RL dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--card:#fff;--line:#e4e1da;--acc:#2f6fdb;--good:#1a8f3c;--warn:#c47f00;--bad:#c0392b;--bar:#d8e4fb}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#ecebe7;--mut:#9b9a95;--card:#1f1f1d;--line:#34332f;--acc:#7aa7ff;--bar:#22324f}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,sans-serif;margin:0;padding:16px;max-width:1500px}
h1{font-size:20px;margin:0 0 2px}.mut{color:var(--mut)}.small{font-size:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin-top:12px}
.card h2{font-size:15px;margin:0 0 8px}.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px;margin-top:12px}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--mut);font-weight:500}td.num{text-align:right;font-variant-numeric:tabular-nums}
td.desc{white-space:normal;min-width:200px;max-width:280px}
.pill{padding:1px 7px;border-radius:9px;font-size:12px;border:1px solid var(--line)}
.running{color:var(--good)}.paused{color:var(--warn)}.pending{color:var(--mut)}
.ci{position:relative;height:10px;width:180px;background:transparent}.ci .r{position:absolute;top:3px;height:4px;background:var(--bar);border-radius:2px}
.ci .p{position:absolute;top:0;width:2px;height:10px;background:var(--acc)}
.best td{font-weight:600}label{margin-right:10px;white-space:nowrap}.tw{overflow-x:auto}
button{font:inherit;font-size:12px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;padding:2px 8px;cursor:pointer}
</style></head><body>
<h1>Slay the Spire RL</h1>
<div class="mut small">Objective: complete the game, i.e. an Ascension 0 Act 3 boss win. Write-up: research/README.md · results: research/results.md</div>
<div class="mut small" id="hdr">loading…</div>
<div class="card"><h2>Win rate, 4,096-world win set (TRAIN seeds from 6e9), Wilson 95% CI</h2>
<div class="tw"><table id="win"></table></div>
<div class="mut small" style="margin-top:6px">Paired comparisons and McNemar p-values: <code>scripts/rl/compare_evals.py A.json B.json</code>. Win counts at 1,024 worlds (training-run evals below) swing by ±7, so use this table for win claims.</div></div>
<div class="row">
<div class="card"><h2>Active now</h2><div class="tw"><table id="active"></table></div><div class="mut small" id="hist"></div></div>
<div class="card"><h2>Fresh-world confirmations (1,024 untouched worlds)</h2><div class="tw"><table id="fresh"></table></div></div>
</div>
<div class="card"><h2>Training runs <button id="arch">show archived</button></h2><div class="tw"><table id="runs"></table></div>
<div class="mut small" id="protocol"></div>
<div class="mut small">ETA uses the last three completed iteration times, including pauses; future eval time is excluded.</div></div>
<div class="grid">
<div class="card"><h2>Paired gain vs original (floors, ±1.96 SE; 1,024 standard worlds)</h2><canvas id="c_paired"></canvas></div>
<div class="card"><h2>Wins per 1,024-world eval (noisy; see the win set above)</h2><canvas id="c_wins"></canvas></div>
<div class="card"><h2>Training-time floor (sampled collection policy)</h2><canvas id="c_train"></canvas></div>
<div class="card"><h2>Sampled behaviour: smith (solid), elite path (dashed), card skip (dotted)</h2><canvas id="c_beh"></canvas></div>
<div class="card"><h2>Act 2 boss pass (solid), Act 3 boss reach (dashed), per eval</h2><canvas id="c_boss"></canvas></div>
<div class="card"><h2>Learning health: half-batch gradient cosine</h2><canvas id="c_cos"></canvas></div>
<div class="card"><h2>Entropy</h2><canvas id="c_entropy"></canvas></div>
<div class="card"><h2>KL per update (approx_kl)</h2><canvas id="c_kl"></canvas></div>
<div class="card"><h2>Value loss</h2><canvas id="c_value"></canvas></div>
<div class="card"><h2>Iteration minutes (solid), collect (dashed), update (dotted)</h2><canvas id="c_time"></canvas></div>
<div class="card"><h2>Win rate: collection (solid), eval (dashed)</h2><canvas id="c_rate"></canvas></div>
</div>
<script>
const POLL_MS=30000, MAX_POINTS=400;
const COLORS=['#2f6fdb','#d9480f','#2b8a3e','#ae3ec9','#e67700','#0b7285','#c2255c','#5c940d','#495057','#1971c2'];
const charts={}; let runs={}, sig='', showArchived=false, timer=null, inflight=false, lastOk=null;
const prev={};
function set(id,html){if(prev[id]!==html){document.getElementById(id).innerHTML=html;prev[id]=html}}
function fmt(v,d=2){return v==null||v!==v?'–':Number(v).toFixed(d)}
function pct(v,d=1){return v==null?'–':(100*v).toFixed(d)+'%'}
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function dur(s){if(s==null)return'–';s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?`${h}h${String(m).padStart(2,'0')}m`:`${m}m`}
function rootCell(root){const label=root.replace(/\/data\/rl_runs\/?$/,'').split('/').pop();return `<td class="mut small" title="${esc(root)}">${esc(label)}</td>`}
function progressText(r){return r.iterations_done==null?'–':`${r.iterations_done} / ${r.iterations??r.progress?.[1]??'?'} iterations`}
function timing(r){if(r.mean_iteration_seconds==null)return'–';let text=`${fmt(r.mean_iteration_seconds,0)} s/iteration (last 3)`;
 if(r.status==='running'&&r.eta)text+=`<br>~${dur(r.remaining_seconds)} left · finish ${new Date(r.eta*1000).toLocaleString()}`;
 else if(['pending','paused'].includes(r.status)&&r.remaining_seconds)text+=`<br>~${dur(r.remaining_seconds)} left once resumed`;
 return text}
function faults(r){return (r.faults?` · faults ${r.faults}`:'')+(r.collector_restarts?` · collector_restarts ${r.collector_restarts}`:'')}
function trainText(t){return t?`floor ${fmt(t.collect_floor)} · wins ${t.collect_wins??'–'}/${t.games??'?'}<br>cosine ${fmt(t.grad_cosine_half,3)} · KL ${fmt(t.approx_kl,4)} · entropy ${fmt(t.entropy,3)}${t.early_stop?' · early stop':''}${faults(t)}`:'–'}
function evalText(e){return e?`u${e.iteration}: ${e.wins??'–'}/${e.n??'?'} wins · floor ${fmt(e.mean_floor)}${faults(e)}`:'–'}
function visible(){return Object.keys(runs).filter(n=>showArchived||runs[n].key||runs[n].active)}
function color(n){return COLORS[Object.keys(runs).indexOf(n)%COLORS.length]}
function mk(id,ds,opts={}){const c=charts[id];
 if(c){const hidden=new Set(c.data.datasets.filter((d,i)=>!c.isDatasetVisible(i)).map(d=>d.label));ds.forEach(d=>{if(hidden.has(d.label))d.hidden=true});c.data.datasets=ds;c.update('none');return}
 charts[id]=new Chart(document.getElementById(id),{type:'line',data:{datasets:ds},options:Object.assign({animation:false,parsing:false,
  plugins:{legend:{labels:{boxWidth:10,font:{size:11}}}},scales:{x:{type:'linear',title:{display:true,text:'update'}}}},opts)})}
function pts(a,x,y,scale=1){return a.filter(r=>r[y]!=null&&Number.isFinite(r[y])).map(r=>({x:r[x],y:r[y]*scale}))}
function rates(a,numerator,denominator){return a.filter(r=>r[numerator]!=null&&r[denominator]>0).map(r=>({x:r.iteration,y:100*r[numerator]/r[denominator]}))}
function winTable(rows,id,maxHi){
 const top=rows.length?Math.max(...rows.map(r=>r.hi)):1;const W=180;
 let h='<tr><th>agent</th><th class="num">wins / n</th><th class="num">win rate</th><th>95% CI</th><th class="num">mean floor</th><th class="num">Act 2 boss pass</th><th class="num">Act 3 boss reach</th><th class="num">Act 3 boss pass</th><th class="mut">file</th></tr>';
 rows.forEach((r,i)=>{const sc=v=>Math.round(W*v/top);
  h+=`<tr class="${i==0?'best':''}"><td>${r.label}</td><td class="num">${r.wins} / ${r.n}</td><td class="num">${pct(r.rate,2)}</td>
  <td><div class="ci"><div class="r" style="left:${sc(r.lo)}px;width:${Math.max(2,sc(r.hi)-sc(r.lo))}px"></div><div class="p" style="left:${sc(r.rate)}px"></div></div><span class="mut small">${pct(r.lo)}–${pct(r.hi)}</span></td>
  <td class="num">${fmt(r.mean_floor)}</td><td class="num">${pct(r.act2_boss_pass)}</td><td class="num">${pct(r.act3_boss_reach)}</td><td class="num">${pct(r.act3_boss_pass)}</td><td class="mut small">${r.name}${r.mode&&r.mode!=='greedy'?' · '+r.mode:''}</td></tr>`});
 set(id,rows.length?h:'<tr><td class="mut">no results yet</td></tr>')}
async function load(){
 const res=await fetch(`/api/state?max_points=${MAX_POINTS}&since=${encodeURIComponent(sig)}`,{cache:'no-store',signal:AbortSignal.timeout(20000)});
 if(!res.ok)throw new Error(`HTTP ${res.status}`);
 const s=await res.json();lastOk=s.now;
 document.getElementById('hdr').textContent=s.runners.map(({root,status:rn})=>`${root}: `+(rn?`heartbeat ${fmt(s.now-rn.heartbeat,0)}s ago · ${rn.on_battery==null?'power unknown':rn.on_battery?'ON BATTERY: jobs paused':'AC power'}`:'queue runner not started')).join(' | ')+` · updated ${new Date(s.now*1000).toLocaleTimeString()}`;
 let a='<tr><th>job</th><th>root / queue</th><th>status</th><th>lane</th><th class="num">elapsed</th><th>progress</th><th>timing / estimated finish</th><th>latest train</th><th>latest eval</th></tr>';
 s.active.forEach(j=>{
  a+=`<tr><td title="${esc(j.out)}">${esc(j.name)}</td>${rootCell(j.root)}<td><span class="pill ${j.status}">${j.status}</span></td><td class="mut">${esc(j.lane)}</td><td class="num">${dur(j.elapsed)}</td><td>${progressText(j)}</td><td>${timing(j)}</td><td>${trainText(j.last_train)}</td><td class="mut">${evalText(j.last_eval)}</td></tr>`});
 set('active',s.active.length?a:'<tr><td class="mut">nothing running</td></tr>');
 set('hist','History: '+Object.entries(s.history).map(([k,v])=>`${v} ${k}`).join(', '));
 if(s.runs!==null){runs=s.runs;sig=s.sig;winTable(s.win,'win');winTable(s.fresh,'fresh');draw()}
 s.active.forEach(j=>{if(runs[j.run])runs[j.run].eta=j.eta});renderRuns();
}
function renderRuns(){
 const names=visible();const n=Object.keys(runs).length;
 document.getElementById('arch').textContent=showArchived?`hide archived`:`show archived (${n-Object.keys(runs).filter(k=>runs[k].key||runs[k].active).length})`;
 let h='<tr><th>run</th><th>root / queue</th><th>description</th><th>status</th><th>progress</th><th>timing / estimated finish</th><th>latest train</th><th>latest eval</th><th class="num">paired vs orig</th></tr>';
 names.forEach(nm=>{const r=runs[nm];const p=r.last_eval?.paired;const st=r.status;
  h+=`<tr><td style="color:${color(nm)}">■ ${esc(r.name)}</td>${rootCell(r.root)}<td class="mut desc">${esc(r.label)}</td><td><span class="pill ${st}">${st}</span></td><td>${progressText(r)}</td><td>${timing(r)}</td><td>${trainText(r.last_train)}</td><td class="mut">${evalText(r.last_eval)}</td>
  <td class="num">${p?(p.diff>=0?'+':'')+fmt(p.diff)+' ± '+fmt(p.se):'–'}</td></tr>`});
 set('runs',h);
 const protocols=new Set(Object.values(runs).map(r=>!!r.config.hide_ooc_potions));
 set('protocol',protocols.size>1?'Eval numbers with out-of-combat potions visible are not comparable with those using --hide-ooc-potions (different protocols).':'')}
document.getElementById('arch').addEventListener('click',()=>{showArchived=!showArchived;renderRuns();draw()});
function draw(){
 const names=visible();const ds=f=>names.flatMap(n=>f(n,color(n),runs[n]));
 mk('c_paired',ds((n,c,r)=>{const e=r.evals.filter(x=>x.paired);return[
  {label:n,data:e.map(x=>({x:x.iteration,y:x.paired.diff})),borderColor:c,backgroundColor:c},
  {label:n+' hi',data:e.map(x=>({x:x.iteration,y:x.paired.diff+1.96*x.paired.se})),borderColor:c,borderDash:[2,3],pointRadius:0,borderWidth:1},
  {label:n+' lo',data:e.map(x=>({x:x.iteration,y:x.paired.diff-1.96*x.paired.se})),borderColor:c,borderDash:[2,3],pointRadius:0,borderWidth:1}]}),
  {plugins:{legend:{labels:{filter:l=>!/ (hi|lo)$/.test(l.text),boxWidth:10}}}});
 mk('c_wins',ds((n,c,r)=>[{label:n,data:pts(r.evals,'iteration','wins'),borderColor:c,backgroundColor:c}]));
 mk('c_train',ds((n,c,r)=>[{label:n,data:pts(r.train,'iteration','collect_floor'),borderColor:c,pointRadius:1}]));
 mk('c_beh',ds((n,c,r)=>[{label:n+' smith',data:pts(r.train,'iteration','smith_rate'),borderColor:c,pointRadius:1},
  {label:n+' elite',data:pts(r.train,'iteration','map_elite'),borderColor:c,borderDash:[5,4],pointRadius:1},
  {label:n+' card skip',data:pts(r.train,'iteration','card_skip_rate'),borderColor:c,borderDash:[2,3],pointRadius:1}]));
 mk('c_boss',ds((n,c,r)=>[{label:n+' A2 pass',data:pts(r.evals,'iteration','act2_boss_pass'),borderColor:c},
  {label:n+' A3 reach',data:pts(r.evals,'iteration','act3_boss_reach'),borderColor:c,borderDash:[5,4]}]));
 mk('c_cos',ds((n,c,r)=>[{label:n,data:pts(r.train,'iteration','grad_cosine_half'),borderColor:c,pointRadius:1}]));
 for(const [id,field] of [['c_entropy','entropy'],['c_kl','approx_kl'],['c_value','value_loss']])
  mk(id,ds((n,c,r)=>[{label:n,data:pts(r.train,'iteration',field),borderColor:c,pointRadius:1}]));
 mk('c_time',ds((n,c,r)=>[
  {label:n+' total',data:pts(r.train,'iteration','iteration_seconds',1/60),borderColor:c,pointRadius:1},
  {label:n+' collect',data:pts(r.train,'iteration','collect_seconds',1/60),borderColor:c,borderDash:[5,4],pointRadius:1},
  {label:n+' update',data:pts(r.train,'iteration','update_seconds',1/60),borderColor:c,borderDash:[2,3],pointRadius:1}]));
 mk('c_rate',ds((n,c,r)=>[
  {label:n+' collect %',data:rates(r.train,'collect_wins','games'),borderColor:c,pointRadius:1},
  {label:n+' eval %',data:rates(r.evals,'wins','n'),borderColor:c,borderDash:[5,4]}]));
}
async function poll(){
 if(document.hidden||inflight)return;inflight=true;
 try{await load()}catch(e){document.getElementById('hdr').textContent=`update failed (${e.message||e}); retrying`+(lastOk?` · last good ${new Date(lastOk*1000).toLocaleTimeString()}`:'')}
 finally{inflight=false}
 clearTimeout(timer);timer=document.hidden?null:setTimeout(poll,POLL_MS)}
document.addEventListener('visibilitychange',()=>{if(document.hidden){clearTimeout(timer);timer=null}else poll()});
poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    extra_roots = ()

    def do_GET(self):
        if self.path.startswith("/api/state"):
            body, kind = state_payload(self.path, self.extra_roots), "application/json"
        elif self.path in ("/", "/index.html"):
            body, kind = PAGE.encode(), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


def _server_pids(port: int) -> list[int]:
    """PIDs of `dashboard.py serve --port <port>` processes (never matches this process)."""
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()
    pids = []
    for line in out[1:]:
        pid, _, args = line.strip().partition(" ")
        parts = args.split()
        serves = any(Path(p).name == "dashboard.py" and parts[i + 1:i + 2] == ["serve"]
                     for i, p in enumerate(parts))
        ports = [parts[i + 1] for i, p in enumerate(parts[:-1]) if p == "--port"]
        ports.extend(p.split("=", 1)[1] for p in parts if p.startswith("--port="))
        if serves and str(port) in ports and int(pid) != os.getpid():
            pids.append(int(pid))
    return pids


def _start(port: int, extra_roots=()) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, __file__, "serve", "--port", str(port)]
    for root in extra_roots:
        cmd.extend(["--root", str(Path(root).expanduser().resolve())])
    with (RUNS / "dashboard.log").open("a") as log:
        subprocess.Popen(cmd, cwd=ROOT, start_new_session=True, stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT)
    print(f"dashboard started at http://127.0.0.1:{port}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="serve", choices=("serve", "start", "restart"))
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--root", type=Path, action="append", default=[], help="additional data/rl_runs root (repeatable)")
    args = parser.parse_args(argv)
    command, port = args.command, args.port
    if command == "start":
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=3):
                pass
            print(f"dashboard already up at http://127.0.0.1:{port}")
            return
        except Exception:
            pass
        if _server_pids(port):
            print(f"dashboard already starting at http://127.0.0.1:{port}")
            return
        _start(port, args.root)
        return
    if command == "restart":
        for pid in _server_pids(port):
            os.kill(pid, signal.SIGTERM)
        time.sleep(1)
        _start(port, args.root)
        return
    Handler.extra_roots = tuple(args.root)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
