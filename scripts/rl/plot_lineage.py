#!/usr/bin/env python3
"""Reconstruct the out-of-combat PPO training lineage and plot it as one HTML page.

Every run is placed where its parent checkpoint sits in the parent's history, so
the x-axis counts out-of-combat PPO iterations (or training games) from the root
instead of restarting at zero. Parents are matched by SHA-256 of checkpoint
bytes: a launch record's ``parent_checkpoint.sha256`` when there is one,
otherwise the hash of the file named in the start config. Edges the files do not
show (head splices made with combine_heads.py, runs branched by copying
``latest.pt`` and resuming, champion labels) come from ``--extra-edges``.

Writes ``<out>/lineage.json`` (nodes, edges, cached file hashes) and
``<out>/training_lineage.html``. Re-running reuses cached hashes, so it is cheap
to refresh while a run is still training. Read-only on every run folder.

Example (from the repository root; see the experiment folder for the full list):
  .venv/bin/python scripts/rl/plot_lineage.py <worktree>/data/rl_runs ... \\
      --extra-edges experiments/2026-10-22-training-lineage/extra_edges.json \\
      --exclude repro_verification --out data/lineage
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "node_modules", "records", "record_objects", "external", "__pycache__"}
ITER_RE = re.compile(r"^iter_(\d+)\.pt$")
PLOTLY_URL = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"


# ---------------------------------------------------------------- discovery

def read_events(path: Path) -> list[dict]:
    events = []
    with open(path) as handle:
        for line in handle:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a running job may be part-way through writing a line
    return events


def first_config(events: list[dict]) -> dict:
    for event in events:
        if isinstance(event.get("config"), dict):
            return event["config"]
    return {}


def resolve(raw: str | None, run_dir: Path) -> Path | None:
    """Config paths are absolute or relative to the checkout the job ran in."""
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path if path.exists() else None
    for ancestor in run_dir.parents:
        if (ancestor / path).exists():
            return ancestor / path
    return None


def run_id(run_dir: Path) -> str:
    text = str(run_dir)
    for marker in ("/rl_runs/", "/data/"):
        if marker in text:
            return text.split(marker, 1)[1]
    return run_dir.name


def parse_run(log: Path, events: list[dict], kind: str) -> dict:
    run_dir = log.parent
    starts = [e for e in events if e.get("event") == "start"]
    resumes = [e for e in events if e.get("event") == "resume"]
    config = first_config(events)
    kinds = [e.get("event") for e in events]
    # a run that resumes before training anything was branched from a copied latest.pt
    branched = "resume" in kinds and ("train" not in kinds or kinds.index("resume") < kinds.index("train"))
    base_iter = int(resumes[0]["iteration"]) if branched else 0
    trains, evals = {}, {}
    for e in events:  # later events win, so a resumed iteration is counted once
        if e.get("event") == "train" and "iteration" in e:
            trains[int(e["iteration"])] = {"games": e.get("games") or 0, "collect_wins": e.get("collect_wins"),
                                           "entropy": e.get("entropy")}
        if kind == "ppo" and e.get("event") == "eval" and e.get("wins") is not None and e.get("n"):
            evals[int(e["iteration"])] = {"wins": e["wins"], "n": e["n"], "mean_floor": e.get("mean_floor"),
                                          "world_start": e.get("world_start", config.get("eval_world_start"))}
    parent_sha, evidence = None, None
    record = ((starts[0].get("run_record") or {}).get("path")) if starts else None
    if record and Path(record).exists():
        parent_sha = (json.loads(Path(record).read_text()).get("parent_checkpoint") or {}).get("sha256")
        evidence = "launch record parent hash" if parent_sha else None
    return {
        "id": run_id(run_dir), "kind": kind, "dir": run_dir, "config": config,
        "branched": branched, "base_iter": base_iter,
        "resume_iters": sorted({int(e["iteration"]) for e in resumes}),
        "start_time": events[0].get("t") if events else None,
        "trains": trains, "evals": evals, "done": any(e.get("event") == "done" for e in events),
        "parent_raw": config.get("checkpoint"), "parent_path": resolve(config.get("checkpoint"), run_dir),
        "parent_sha": parent_sha, "parent_evidence": evidence,
    }


def discover(roots: list[Path], excludes: list[str]) -> tuple[list[dict], list[dict], list[dict]]:
    runs, skipped, promotions, seen = [], [], [], set()
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            promotions += read_promotion_records(Path(dirpath), filenames)
            if "log.jsonl" not in filenames:
                continue
            log = Path(dirpath) / "log.jsonl"
            if log.resolve() in seen:
                continue
            seen.add(log.resolve())
            if any(pattern in str(log) for pattern in excludes):
                skipped.append({"path": str(log.parent), "reason": "excluded by --exclude"})
                continue
            events = read_events(log)
            config = first_config(events)
            if "credit" in config:
                kind = "ppo"
            elif "beta" in config and "head" in config:
                if config.get("head") != "ooc":
                    skipped.append({"path": str(log.parent), "reason": f"{config.get('head')}-head distillation"})
                    continue
                kind = "distill"
            else:
                continue  # some other tool's log; not part of the OOC lineage
            run = parse_run(log, events, kind)
            if kind == "ppo" and not run["trains"] and any(i > run["base_iter"] for i in run["evals"]):
                skipped.append({"path": str(log.parent), "reason": "evaluation-only re-score, no training"})
                continue
            runs.append(run)
    ids = [r["id"] for r in runs]
    for run in runs:  # disambiguate the same relative name in two checkouts
        if ids.count(run["id"]) > 1:
            run["id"] = f"{run['id']} ({run['dir'].parts[-4]})"
    return runs, skipped, promotions


def read_promotion_records(directory: Path, filenames: list[str]) -> list[dict]:
    """inputs/champion_<hash>.json files written when Sid promotes a checkpoint."""
    records = []
    for name in filenames:
        if name.startswith("champion_") and name.endswith(".json"):
            try:
                data = json.loads((directory / name).read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and data.get("sha256") and data.get("promoted"):
                records.append({**data, "file": str(directory / name)})
    return records


# ---------------------------------------------------------------- lineage

class Hasher:
    def __init__(self, cache: dict):
        self.cache, self.computed = cache, 0

    def __call__(self, path: Path) -> str:
        stat, key = path.stat(), str(path.resolve())
        hit = self.cache.get(key)
        if hit and hit["size"] == stat.st_size and hit["mtime_ns"] == stat.st_mtime_ns:
            return hit["sha256"]
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        self.cache[key] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}
        self.computed += 1
        return digest.hexdigest()


def protocol(config: dict, evals: dict, extras: dict) -> str:
    """'filtered': potion filter on; 'visible': OOC potion actions offered and not hidden (roughly half the win
    rate); 'absent': an older simulator that had no OOC potion actions (they arrived in environment 0.1.5)."""
    if config.get("hide_ooc_potions") is True:
        return "filtered"
    worlds = {str(e["world_start"]) for e in evals.values()} or {str(config.get("eval_world_start"))}
    absent = set(extras.get("eval_worlds_without_potion_actions", []))
    return "absent" if worlds <= absent else "visible"


def reconstruct(roots: list[Path], extras: dict | None = None, excludes: list[str] | None = None,
                hash_cache: dict | None = None) -> dict:
    extras, hash_cache = extras or {}, {} if hash_cache is None else hash_cache
    hasher, warnings = Hasher(hash_cache), []
    runs, skipped, promotions = discover(roots, excludes or [])
    labels = extras.get("labels", {})
    nodes: dict[str, dict] = {}
    index: dict[str, tuple[str, int | None]] = {}  # sha256 -> (node id, iteration)
    by_id = {}

    for run in runs:
        nid = f"{run['kind']}:{run['id']}"
        by_id[run["id"]] = nid
        cfg = run["config"]
        nodes[nid] = {
            "id": nid, "kind": run["kind"], "run": run["id"], "label": labels.get(run["id"], run["id"]),
            "path": str(run["dir"]), "start_time": run["start_time"], "base_iter": run["base_iter"],
            "last_iter": max(run["trains"], default=run["base_iter"]), "done": run["done"],
            "resume_iters": [i for i in run["resume_iters"] if i != run["base_iter"] or not run["branched"]],
            "trains": run["trains"], "evals": run["evals"], "notes": cfg.get("notes"),
            "games_per_iter": cfg.get("games"), "seed": cfg.get("seed"),
            "protocol": protocol(cfg, run["evals"], extras), "parent": None,
        }
        if run["kind"] == "ppo":
            for file in sorted(run["dir"].iterdir()):
                match = ITER_RE.match(file.name)
                if match:
                    index.setdefault(hasher(file), (nid, int(match.group(1))))
        elif (run["dir"] / "distilled.pt").exists():
            index.setdefault(hasher(run["dir"] / "distilled.pt"), (nid, None))

    splices = []
    for splice in extras.get("splices", []):
        nid = f"splice:{splice['id']}"
        output = Path(splice["output"])
        if not output.exists():
            warnings.append(f"splice {splice['id']}: output {output} is missing")
            continue
        sha = hasher(output)
        nodes[nid] = {"id": nid, "kind": "splice", "label": splice.get("label", splice["id"]), "path": str(output),
                      "sha256": sha, "combat": splice.get("combat_label"), "combat_path": splice.get("combat"),
                      "start_time": output.stat().st_mtime, "parent": None, "notes": splice.get("note")}
        index.setdefault(sha, (nid, None))
        splices.append((nid, Path(splice["ooc"])))

    def locate(sha: str | None, path: Path | None, raw: str | None) -> tuple[str, int | None]:
        if sha in index:
            return index[sha]
        name = Path(raw).name if raw else "unknown"
        nid = f"external:{sha[:12] if sha else name}"
        if nid not in nodes:
            external = extras.get("externals", {}).get(name, {})
            nodes[nid] = {"id": nid, "kind": "external", "label": external.get("label", name),
                          "path": str(path or raw), "sha256": sha, "parent": None, "start_time": None,
                          "combat": external.get("combat_label")}
            index[sha or nid] = (nid, None)
        return nodes[nid]["id"], None

    edges = []

    def link(child: str, parent: tuple[str, int | None], kind: str, origin: str, evidence: str) -> None:
        nodes[child]["parent"] = {"node": parent[0], "iteration": parent[1]}
        edges.append({"type": kind, "source": parent[0], "source_iteration": parent[1], "target": child,
                      "origin": origin, "evidence": evidence})

    branches = {b["run"]: b for b in extras.get("branches", [])}
    for run in runs:
        nid = by_id[run["id"]]
        if run["id"] in branches:
            branch = branches[run["id"]]
            parent = by_id.get(branch["parent_run"])
            if parent is None:
                warnings.append(f"branch {run['id']}: parent run {branch['parent_run']} not found")
                continue
            if not run["branched"] or run["base_iter"] != branch["iteration"]:
                warnings.append(f"branch {run['id']}: log resumes at {run['base_iter']}, extra edge says "
                                f"{branch['iteration']}")
            link(nid, (parent, branch["iteration"]), "resume", "extra_edges",
                 branch.get("note", "copied latest.pt and resumed"))
            continue
        if run["branched"]:
            warnings.append(f"{run['id']}: resumed from a copied checkpoint at iteration {run['base_iter']} "
                            "and has no branch entry in the extra edges; parent unknown")
            continue
        sha, evidence = run["parent_sha"], run["parent_evidence"]
        if sha is None and run["parent_path"] is not None:
            sha, evidence = hasher(run["parent_path"]), "hash of the start config's checkpoint file"
        if sha is None:
            warnings.append(f"{run['id']}: parent {run['parent_raw']} not found on disk and no launch record")
        parent = locate(sha, run["parent_path"], run["parent_raw"])
        if parent[0].startswith("external:") and sha and run["parent_raw"]:
            evidence = (evidence or "") + "; hash matches no discovered checkpoint"
        link(nid, parent, "ppo" if run["kind"] == "ppo" else "distill", "files", evidence or "")

    for nid, ooc in splices:
        if not ooc.exists():
            warnings.append(f"{nid}: OOC source {ooc} is missing")
            continue
        sha = hasher(ooc)
        link(nid, locate(sha, ooc, str(ooc)), "splice", "extra_edges",
             f"OOC head from {ooc.name} (matched by hash); combat head from {nodes[nid]['combat_path']}")

    # positions: a node starts where its parent checkpoint sits
    memo: dict[str, tuple[float, float]] = {}

    def start(nid: str, depth: int = 0) -> tuple[float, float]:
        if nid not in memo:
            parent = nodes[nid]["parent"]
            memo[nid] = (0.0, 0.0) if parent is None or depth > 200 else at(parent["node"], parent["iteration"], depth + 1)
        return memo[nid]

    def at(nid: str, iteration: int | None, depth: int = 0) -> tuple[float, float]:
        x0, g0 = start(nid, depth)
        node = nodes[nid]
        if node["kind"] != "ppo" or iteration is None:
            return x0, g0
        games = sum(t["games"] for i, t in node["trains"].items() if node["base_iter"] < i <= iteration)
        return x0 + iteration - node["base_iter"], g0 + games

    def combat(nid: str, depth: int = 0) -> str:
        node = nodes[nid]
        if node.get("combat") or node["parent"] is None or depth > 200:
            return node.get("combat") or extras.get("default_combat_label", "original combat head")
        return combat(node["parent"]["node"], depth + 1)

    champions = []
    for record in extras.get("champions", []):
        sha = record.get("sha256")
        if not sha and record.get("path") and Path(record["path"]).exists():
            sha = hasher(Path(record["path"]))
        hit = index.get(sha) if sha else None
        if hit is None:
            warnings.append(f"champion {record.get('label')}: hash {str(sha)[:12]} matches no discovered checkpoint")
            continue
        champions.append({**record, "sha256": sha, "node": hit[0], "iteration": hit[1]})
    for record in promotions:
        if any(c["sha256"] == record["sha256"] for c in champions):
            continue
        hit = index.get(record["sha256"])
        if hit is None:
            warnings.append(f"promotion record {record['file']}: hash matches no discovered checkpoint")
            continue
        champions.append({"label": f"champion_{record['sha256'][:12]}", "promoted": record.get("promoted"),
                          "sha256": record["sha256"], "node": hit[0], "iteration": hit[1],
                          "source": record.get("source")})

    for nid, node in nodes.items():
        node["x0"], node["g0"] = start(nid)
        end = node.get("last_iter") if node["kind"] == "ppo" else None
        node["x_end"], node["g_end"] = at(nid, end)
        node["combat"] = combat(nid)
        if node["kind"] == "ppo":
            node["eval_points"] = [{"iteration": i, "x": at(nid, i)[0], "g": at(nid, i)[1], **e,
                                    "rate": e["wins"] / e["n"]} for i, e in sorted(node["evals"].items())]
            node["resumes"] = [{"iteration": i, "x": at(nid, i)[0], "g": at(nid, i)[1]} for i in node["resume_iters"]]
    for champion in champions:
        champion["x"], champion["g"] = at(champion["node"], champion["iteration"])
    for node in nodes.values():  # keep the JSON small: per-iteration training stats are not plotted
        if node["kind"] == "ppo":
            node["iterations"] = len(node.pop("trains"))
            node["evals"] = node.pop("eval_points")
            del node["resume_iters"]
        else:
            for key in ("trains", "evals", "resume_iters"):
                node.pop(key, None)

    # the current champion's ancestors are drawn heavier
    if champions:
        latest = max(champions, key=lambda c: c.get("promoted") or "")
        nid = latest["node"]
        while nid:
            nodes[nid]["champion_path"] = True
            parent = nodes[nid]["parent"]
            nid = parent["node"] if parent else None

    assign_lanes(nodes)
    return {"generated": time.strftime("%Y-%m-%d %H:%M"), "roots": [str(r) for r in roots],
            "nodes": list(nodes.values()), "edges": edges, "champions": champions, "warnings": warnings,
            "skipped": skipped, "eval_world_labels": extras.get("eval_world_labels", {}),
            "hashes_computed": hasher.computed, "file_hashes": hash_cache}


def assign_lanes(nodes: dict[str, dict]) -> None:
    """Depth-first order, children by start position then time, so a branch sits just below its parent."""
    children: dict[str | None, list[str]] = {}
    for nid, node in nodes.items():
        children.setdefault(node["parent"]["node"] if node["parent"] else None, []).append(nid)
    order = lambda nid: (nodes[nid].get("x0", 0), nodes[nid].get("start_time") or 0)
    lane, stack = 0, sorted(children.get(None, []), key=order, reverse=True)
    while stack:
        nid = stack.pop()
        nodes[nid]["lane"] = lane
        lane += 1
        stack.extend(sorted(children.get(nid, []), key=order, reverse=True))


# ---------------------------------------------------------------- HTML

def write_html(lineage: dict, path: Path) -> None:
    data = {k: v for k, v in lineage.items() if k != "file_hashes"}
    payload = json.dumps(data, default=str).replace("</", "<\\/")
    path.write_text(HTML.replace("__PLOTLY__", PLOTLY_URL).replace("__DATA__", payload))


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Training lineage</title>
<script src="__PLOTLY__"></script>
<style>
:root { color-scheme: light; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8984; --grid:#e6e5e0;
  --panel:#f3f2ee; --c0:#2a78d6; --c1:#1baf7a; --c2:#eb6834; --c3:#6250d6; --c4:#e87ba4;
  --ppo:#52514e; --distill:#eda100; --splice:#6250d6; --star:#0b0b0b; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --surface:#1a1a19;
  --ink:#f0efec; --ink2:#c3c2b7; --muted:#8f8e86; --grid:#33322f; --panel:#242422; --c0:#3987e5; --c1:#199e70;
  --c2:#d95926; --c3:#9085e9; --c4:#d55181; --ppo:#c3c2b7; --distill:#c98500; --splice:#9085e9; --star:#f0efec; } }
:root[data-theme="dark"] { color-scheme: dark; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#8f8e86;
  --grid:#33322f; --panel:#242422; --c0:#3987e5; --c1:#199e70; --c2:#d95926; --c3:#9085e9; --c4:#d55181;
  --ppo:#c3c2b7; --distill:#c98500; --splice:#9085e9; --star:#f0efec; }
body { margin:0; background:var(--surface); color:var(--ink); font:15px/1.5 system-ui, -apple-system, sans-serif; }
main { max-width:1400px; margin:0 auto; padding:20px 16px 40px; }
h1 { font-size:22px; margin:0 0 6px; } h2 { font-size:17px; margin:28px 0 4px; }
p { margin:6px 0; color:var(--ink2); max-width:110ch; }
.controls { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:14px 0 6px; }
.controls button { border:1px solid var(--grid); background:var(--panel); color:var(--ink); padding:5px 12px;
  border-radius:6px; cursor:pointer; font:inherit; }
.controls button[aria-pressed="true"] { border-color:var(--ink2); font-weight:600; }
.key { display:flex; flex-wrap:wrap; gap:6px 18px; font-size:13px; color:var(--ink2); margin:6px 0; }
.key span { white-space:nowrap; } .sw { display:inline-block; width:22px; height:0; vertical-align:middle;
  margin-right:5px; border-top:3px solid; } .warn { background:var(--panel); border-left:3px solid var(--c2);
  padding:8px 12px; margin:10px 0; max-width:110ch; color:var(--ink); font-size:14px; }
#chart { height:560px; } .small { font-size:13px; color:var(--muted); }
details { margin-top:18px; font-size:13px; color:var(--ink2); } li { margin:2px 0; }
</style></head><body><main>
<h1>Out-of-combat training lineage</h1>
<p>Every line is one training run of the <b>out-of-combat (OOC) policy</b>: the part of the agent that chooses map
paths, card rewards, rest sites and shop purchases, trained with PPO (proximal policy optimisation) while the
combat policy stays frozen. A run starts where its parent checkpoint sits in the parent's history, so the x-axis
counts PPO iterations since the root checkpoint instead of restarting at zero. Parents were matched by the
SHA-256 hash of checkpoint bytes.</p>
<div class="controls"><span class="small">x-axis:</span>
<button id="mode-x" aria-pressed="true">OOC PPO iterations</button>
<button id="mode-g" aria-pressed="false">Training games</button></div>
<div class="key" id="key"></div>
<div class="warn"><b>Compare heights only within one evaluation protocol.</b> Dashed lines were evaluated with
out-of-combat potion actions visible (no potion filter, environment 0.1.5); they win roughly half as often as the
same agent under the potion filter (solid). Dotted lines are runs from 13 to 14 October on older simulator versions,
which had no out-of-combat potion actions, and on different worlds (marker shapes): the 14 October champion scores
about 5% there and 3.4% on the 20 October protocol. A jump at a branch point usually means the protocol changed
there, not the policy.</div>
<div id="chart"></div>
<p class="small">y = the trainer's own evaluation: wins divided by games evaluated, greedy (argmax) play.
Hover a point for the run, iteration, wins, games evaluated, mean floor, cumulative training games and parent.
Click legend entries to hide runs; double-click to isolate one.</p>
<h2>Which model came from where</h2>
<p>One row per run, on the same x-axis as the chart above. A vertical connector joins a run to the parent
checkpoint it started from. Grey connectors are PPO starts, dotted grey are runs branched by copying a run's latest
checkpoint and resuming, amber are distillations of a rule into the OOC head, and violet are splices that pair one
checkpoint's OOC head with another's combat head (a splice adds no OOC training, so it keeps its OOC parent's
x-position). Stars mark promoted champions; small ticks on a bar mark where a run was stopped and resumed.</p>
<div id="tree"></div>
<details id="notes"><summary>Sources, unresolved links and skipped logs</summary><div id="notes-body"></div></details>
</main>
<script>
const L = __DATA__;
const nodes = Object.fromEntries(L.nodes.map(n => [n.id, n]));
let mode = 'x';
const DASH = {filtered: 'solid', visible: 'dash', absent: 'dot'};
const PROTOCOL = {filtered: 'potion filter on', visible: 'potion actions visible', absent: 'older simulator without potion actions'};
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const combats = [...new Set(L.nodes.slice().sort((a, b) => a.x0 - b.x0 || a.lane - b.lane).map(n => n.combat))];
const combatColour = c => css('--c' + Math.min(combats.indexOf(c), 4));
const worlds = [...new Set(L.nodes.flatMap(n => (n.evals || []).map(e => String(e.world_start))))].sort((a, b) => a - b);
const SYMBOLS = ['square', 'triangle-up', 'circle', 'x', 'pentagon'];
const worldSymbol = w => SYMBOLS[worlds.indexOf(String(w)) % SYMBOLS.length];
const worldLabel = w => L.eval_world_labels[String(w)] || ('worlds from ' + Number(w).toLocaleString());
const date = t => t ? new Date(t * 1000).toLocaleString('en-GB', {day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit'}) : '';
const X = o => mode === 'x' ? o.x : o.g;
const parentText = n => {
  if (!n.parent) return 'none (root)';
  const p = nodes[n.parent.node];
  return p.label + (n.parent.iteration != null ? ' @ iteration ' + n.parent.iteration : '');
};
const xTitle = () => mode === 'x' ? 'Cumulative OOC PPO iterations from the root' : 'Cumulative OOC training games from the root';
function layout(extra) {
  const ink = css('--ink2'), grid = css('--grid');
  return Object.assign({paper_bgcolor: css('--surface'), plot_bgcolor: css('--surface'),
    font: {family: 'system-ui, sans-serif', size: 12, color: ink}, margin: {l: 60, r: 20, t: 10, b: 50},
    hoverlabel: {bgcolor: css('--panel'), font: {color: css('--ink')}},
    xaxis: {title: xTitle(), gridcolor: grid, zeroline: false, linecolor: grid},
  }, extra);
}
function key() {
  const k = document.getElementById('key');
  k.innerHTML = combats.map(c => `<span><i class="sw" style="border-color:${combatColour(c)}"></i>${c}</span>`).join('') +
    `<span><i class="sw" style="border-color:var(--ink2)"></i>potion filter (study protocol)</span>` +
    `<span><i class="sw" style="border-top-style:dashed;border-color:var(--ink2)"></i>potion actions visible (about half the win rate)</span>` +
    `<span><i class="sw" style="border-top-style:dotted;border-color:var(--ink2)"></i>older simulator, before potion actions existed</span>` +
    worlds.map(w => `<span>${{square: '■', 'triangle-up': '▲', circle: '●', x: '✕', pentagon: '⬟'}[worldSymbol(w)]} ${worldLabel(w)}</span>`).join('') +
    `<span>○ first point of a branch</span><span>◆ combat head swapped</span><span>★ champion</span><span>thick lines: ancestors of the current champion</span>`;
}
function chart() {
  const traces = [];
  const runs = L.nodes.filter(n => n.kind === 'ppo' && n.evals.length).sort((a, b) => combats.indexOf(a.combat) - combats.indexOf(b.combat) || a.lane - b.lane);
  for (const n of runs) {
    const heavy = !!n.champion_path;
    traces.push({type: 'scatter', mode: 'lines+markers', name: n.label, legendgroup: n.combat,
      legendgrouptitle: {text: n.combat}, x: n.evals.map(X), y: n.evals.map(e => e.rate),
      line: {color: combatColour(n.combat), width: heavy ? 3 : 1.5, dash: DASH[n.protocol]},
      opacity: heavy ? 1 : 0.7,
      marker: {size: 8, color: combatColour(n.combat), line: {color: css('--surface'), width: 1.5},
        symbol: n.evals.map((e, i) => worldSymbol(e.world_start) + (i === 0 && e.iteration === n.base_iter && n.parent ? '-open' : ''))},
      customdata: n.evals.map(e => [e.iteration, e.wins, e.n, e.mean_floor == null ? 'n/a' : e.mean_floor.toFixed(2),
        Math.round(e.g).toLocaleString(), parentText(n), worldLabel(e.world_start), PROTOCOL[n.protocol],
        n.run, (100 * e.rate).toFixed(1), Math.round(e.x)]),
      hovertemplate: '<b>' + n.label + '</b><br>run %{customdata[8]}<br>iteration %{customdata[0]} (x = %{customdata[10]})<br>' +
        'wins %{customdata[1]} of %{customdata[2]} = %{customdata[9]}%<br>mean floor %{customdata[3]}<br>' +
        'cumulative training games %{customdata[4]}<br>parent: %{customdata[5]}<br>eval: %{customdata[6]}<br>%{customdata[7]}<extra></extra>'});
  }
  // connect a branch to its parent's evaluation at the branch point when both use the same protocol
  const cx = [], cy = [];
  for (const n of runs) {
    if (!n.parent || n.evals[0].iteration !== n.base_iter) continue;
    let p = nodes[n.parent.node], it = n.parent.iteration;
    while (p && p.kind !== 'ppo' && p.parent) { it = p.parent.iteration; p = nodes[p.parent.node]; }
    const pe = p && p.evals && p.evals.find(e => e.iteration === it);
    if (!pe || p.protocol !== n.protocol || String(pe.world_start) !== String(n.evals[0].world_start)) continue;
    cx.push(X(pe), X(n.evals[0]), null); cy.push(pe.rate, n.evals[0].rate, null);
  }
  traces.push({type: 'scatter', mode: 'lines', x: cx, y: cy, name: 'branch connector', showlegend: false,
    hoverinfo: 'skip', line: {color: css('--muted'), width: 1, dash: 'dot'}});
  const spliced = runs.filter(n => n.parent && nodes[n.parent.node].kind === 'splice' && n.evals[0].iteration === n.base_iter);
  traces.push({type: 'scatter', mode: 'markers', name: 'combat head swapped', showlegend: false,
    x: spliced.map(n => X(n.evals[0])), y: spliced.map(n => n.evals[0].rate),
    marker: {symbol: 'diamond-open', size: 18, color: css('--splice'), line: {width: 2}},
    text: spliced.map(n => 'Combat head swapped: ' + nodes[n.parent.node].label), hovertemplate: '%{text}<extra></extra>'});
  const stars = L.champions.map(c => {
    const n = nodes[c.node];
    let e = n.evals && n.evals.find(e => e.iteration === c.iteration);
    if (!e) { const child = runs.find(r => r.parent && r.parent.node === c.node && r.evals[0].iteration === r.base_iter); e = child && child.evals[0]; }
    return e ? {c, e} : null;
  }).filter(Boolean);
  traces.push({type: 'scatter', mode: 'markers+text', name: 'champion', showlegend: false,
    x: stars.map(s => X(s.e)), y: stars.map(s => s.e.rate), text: stars.map(s => s.c.short || s.c.label),
    textposition: 'top center', textfont: {size: 11, color: css('--ink')},
    marker: {symbol: 'star', size: 17, color: css('--star'), line: {color: css('--surface'), width: 1}},
    customdata: stars.map(s => [s.c.label, s.c.promoted || '']), hovertemplate: '<b>%{customdata[0]}</b><br>promoted %{customdata[1]}<extra></extra>'});
  Plotly.react('chart', traces, layout({yaxis: {title: "Trainer's evaluation win rate", tickformat: '.0%', gridcolor: css('--grid'), zeroline: false, rangemode: 'tozero'},
    legend: {font: {size: 11}, groupclick: 'toggleitem', itemclick: 'toggle'}, hovermode: 'closest'}), {responsive: true, displaylogo: false});
}
function tree() {
  const ns = L.nodes.slice().sort((a, b) => a.lane - b.lane);
  const traces = [];
  const edgeStyle = {ppo: ['--ppo', 'solid', 'PPO start'], resume: ['--ppo', 'dot', 'branched by copy and resume'],
    distill: ['--distill', 'solid', 'distillation'], splice: ['--splice', 'solid', 'head splice']};
  for (const [type, [colour, dash, name]] of Object.entries(edgeStyle)) {
    const xs = [], ys = [];
    for (const e of L.edges.filter(e => e.type === type)) {
      const t = nodes[e.target], s = nodes[e.source];
      xs.push(X({x: t.x0, g: t.g0}), X({x: t.x0, g: t.g0}), null); ys.push(s.lane, t.lane, null);
    }
    traces.push({type: 'scatter', mode: 'lines', x: xs, y: ys, name, line: {color: css(colour), width: 2, dash}, hoverinfo: 'skip'});
  }
  for (const n of ns) {
    const colour = combatColour(n.combat);
    const hover = `<b>${n.label}</b><br>${n.kind === 'ppo' ? 'PPO run ' + n.run : n.kind === 'distill' ? 'distillation ' + n.run : n.kind}` +
      `<br>parent: ${parentText(n)}` + (n.kind === 'ppo' ? `<br>iterations ${n.base_iter + 1}–${n.last_iter} (${n.iterations} trained), ${n.games_per_iter} games each, seed ${n.seed}` +
      `<br>x ${Math.round(n.x0)} → ${Math.round(n.x_end)}; ${PROTOCOL[n.protocol]}${n.done ? '' : '; <b>not finished</b>'}` : `<br>x ${Math.round(n.x0)}`) +
      `<br>combat head: ${n.combat}` + (n.start_time ? `<br>started ${date(n.start_time)}` : '') +
      (n.notes ? `<br><i>${String(n.notes).replace(/(.{1,90})(\s|$)/g, '$1<br>')}</i>` : '') + `<br><span style="font-size:10px">${n.path}</span>`;
    if (n.kind === 'ppo') {
      traces.push({type: 'scatter', mode: 'lines', x: [X({x: n.x0, g: n.g0}), X({x: n.x_end, g: n.g_end})], y: [n.lane, n.lane],
        line: {color: colour, width: n.champion_path ? 9 : 5}, opacity: n.champion_path ? 1 : 0.75, showlegend: false, hoverinfo: 'text', text: hover, hoveron: 'points+fills'});
      traces.push({type: 'scatter', mode: 'markers', x: [X({x: n.x0, g: n.g0}), X({x: n.x_end, g: n.g_end})], y: [n.lane, n.lane],
        marker: {size: 10, color: 'rgba(0,0,0,0)'}, showlegend: false, hoverinfo: 'text', text: [hover, hover]});
    } else {
      const symbol = {distill: 'square', splice: 'diamond', external: 'circle'}[n.kind];
      traces.push({type: 'scatter', mode: 'markers', x: [X({x: n.x0, g: n.g0})], y: [n.lane], showlegend: false,
        marker: {symbol, size: 13, color: n.kind === 'splice' ? css('--splice') : n.kind === 'distill' ? css('--distill') : colour,
          line: {color: css('--surface'), width: 1.5}}, hoverinfo: 'text', text: [hover]});
    }
  }
  const labelText = n => '\u2003' + n.label + (n.kind === 'ppo' ? `  ·  ${n.iterations} it.` : '') + (n.start_time ? `  ·  ${date(n.start_time).split(',')[0]}` : '');
  traces.push({type: 'scatter', mode: 'text', x: ns.map(n => X({x: n.x_end, g: n.g_end})), y: ns.map(n => n.lane),
    text: ns.map(labelText), textposition: 'middle right', textfont: {size: 11, color: css('--ink')}, showlegend: false, hoverinfo: 'skip'});
  const rs = ns.flatMap(n => (n.resumes || []).map(r => [n, r]));
  traces.push({type: 'scatter', mode: 'markers', x: rs.map(([n, r]) => X(r)), y: rs.map(([n]) => n.lane), name: 'stopped and resumed',
    marker: {symbol: 'line-ns-open', size: 16, color: css('--ink'), line: {width: 2}}, hovertemplate: 'resumed at iteration %{text}<extra></extra>', text: rs.map(([, r]) => r.iteration)});
  traces.push({type: 'scatter', mode: 'markers', name: 'champion', x: L.champions.map(X), y: L.champions.map(c => nodes[c.node].lane),
    marker: {symbol: 'star', size: 17, color: css('--star'), line: {color: css('--surface'), width: 1}},
    customdata: L.champions.map(c => [c.label, c.promoted || '', c.sha256.slice(0, 12)]),
    hovertemplate: '<b>%{customdata[0]}</b><br>promoted %{customdata[1]}<br>sha256 %{customdata[2]}<extra></extra>'});
  const maxX = Math.max(...ns.map(n => X({x: n.x_end, g: n.g_end})));
  Plotly.react('tree', traces, layout({height: 60 + 24 * ns.length, margin: {l: 20, r: 20, t: 10, b: 50},
    xaxis: {title: xTitle(), gridcolor: css('--grid'), zeroline: false, range: [-0.02 * maxX, maxX * 1.32]},
    yaxis: {autorange: 'reversed', showticklabels: false, showgrid: false, zeroline: false},
    legend: {orientation: 'h', y: 1.0, yanchor: 'bottom', font: {size: 11}}, hovermode: 'closest'}), {responsive: true, displaylogo: false});
}
function notes() {
  const esc = s => String(s).replace(/[&<>]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;'}[c]));
  const from = L.edges.map(e => `<li>${e.type}: ${esc(nodes[e.source].label)}${e.source_iteration != null ? ' @ ' + e.source_iteration : ''} → ${esc(nodes[e.target].label)} — <i>${e.origin === 'files' ? 'from files' : 'from extra_edges.json'}</i>: ${esc(e.evidence)}</li>`);
  document.getElementById('notes-body').innerHTML = `<p>Generated ${L.generated} from ${L.roots.map(esc).join(', ')}.</p>` +
    `<b>Edges</b><ul>${from.join('')}</ul><b>Warnings</b><ul>${(L.warnings.length ? L.warnings : ['none']).map(w => '<li>' + esc(w) + '</li>').join('')}</ul>` +
    `<b>Skipped logs</b><ul>${L.skipped.map(s => '<li>' + esc(s.path) + ': ' + esc(s.reason) + '</li>').join('')}</ul>`;
}
function render() { key(); chart(); tree(); }
for (const [id, m] of [['mode-x', 'x'], ['mode-g', 'g']]) document.getElementById(id).onclick = () => {
  mode = m; document.getElementById('mode-x').setAttribute('aria-pressed', m === 'x'); document.getElementById('mode-g').setAttribute('aria-pressed', m === 'g'); render(); };
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', render);
render(); notes();
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path, help="directories to search for run folders (log.jsonl)")
    parser.add_argument("--extra-edges", type=Path, help="JSON with splices, copy-and-resume branches, champions, labels")
    parser.add_argument("--exclude", action="append", default=[], help="skip logs whose path contains this text")
    parser.add_argument("--out", type=Path, default=Path("data/lineage"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache_file = args.out / "lineage.json"
    cache = json.loads(cache_file.read_text()).get("file_hashes", {}) if cache_file.exists() else {}
    extras = json.loads(args.extra_edges.read_text()) if args.extra_edges else {}
    lineage = reconstruct(args.roots, extras, args.exclude, cache)
    cache_file.write_text(json.dumps(lineage, indent=1, default=str))
    write_html(lineage, args.out / "training_lineage.html")
    print(f"{len(lineage['nodes'])} nodes, {len(lineage['edges'])} edges, {lineage['hashes_computed']} files hashed")
    for warning in lineage["warnings"]:
        print("warning:", warning, file=sys.stderr)
    print("wrote", cache_file, "and", args.out / "training_lineage.html")


if __name__ == "__main__":
    main()
