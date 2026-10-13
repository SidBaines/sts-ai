#!/usr/bin/env python3
"""Read-only live dashboard for data/rl_runs (http://127.0.0.1:8791).

    dashboard.py serve [--port 8791]     run in foreground
    dashboard.py start                   launch detached (idempotent)
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "rl_runs"
PORT = 8791
KEEP_TRAIN = ("iteration", "t", "collect_floor", "collect_floor_se", "collect_act1_clear", "collect_act2_clear",
              "collect_wins", "games", "grad_cosine_half", "approx_kl", "entropy", "value_loss", "clip_fraction",
              "steps", "steps_total", "collect_win", "smith_rate", "card_skip_rate", "map_monster", "map_elite", "map_event",
              "map_rest", "shop_remove", "shop_potion", "fights_per_run", "iteration_seconds", "adv_raw_std",
              "disp_ratio_state_encoder", "disp_ratio_scorer", "disp_ratio_critic", "disp_ratio_action_encoder",
              "panel_score", "teacher_agreement")


def state() -> dict:
    queue = json.loads((RUNS / "queue.json").read_text()) if (RUNS / "queue.json").exists() else {"runs": []}
    status = json.loads((RUNS / "runner_status.json").read_text()) if (RUNS / "runner_status.json").exists() else None
    runs = {}
    for directory in sorted(p for p in RUNS.iterdir() if p.is_dir() and not p.name.startswith("_")) if RUNS.exists() else []:
        log = directory / "log.jsonl"
        if not log.exists():
            continue
        train, evals, panels, config, done = [], [], [], {}, False
        for line in log.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "start":
                config = r.get("config", {})
            elif r.get("event") == "train":
                train.append({k: r.get(k) for k in KEEP_TRAIN if k in r})
            elif r.get("event") == "eval":
                evals.append({k: v for k, v in r.items() if k != "event"})
            elif r.get("event") == "panel":
                panels.append({k: v for k, v in r.items() if k != "event"})
            elif r.get("event") == "done":
                done = True
        runs[directory.name] = {"config": config, "train": train, "evals": evals, "panels": panels, "done": done,
                                "mtime": log.stat().st_mtime}
    baseline = None
    base_file = RUNS / "baselines" / "orig_argmax_1024.json"
    if base_file.exists():
        baseline = json.loads(base_file.read_text())["summary"]
    return {"now": time.time(), "runner": status,
            "queue": [{k: r.get(k) for k in ("name", "status", "started", "ended", "returncode")} for r in queue["runs"]],
            "runs": runs, "baseline": baseline}


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>StS RL runs</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--card:#fff;--line:#e4e1da;--acc:#2f6fdb}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#ecebe7;--mut:#9b9a95;--card:#1f1f1d;--line:#34332f;--acc:#7aa7ff}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,sans-serif;margin:0;padding:16px}
h1{font-size:20px;margin:0 0 4px}.mut{color:var(--mut)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px;margin-top:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h2{font-size:14px;margin:0 0 8px}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--mut);font-weight:500}.pill{padding:1px 7px;border-radius:9px;font-size:12px;border:1px solid var(--line)}
.running{color:#1a8f3c}.paused{color:#c47f00}.failed{color:#c0392b}.done{color:var(--mut)}
label{margin-right:10px;white-space:nowrap}.tw{overflow-x:auto}
</style></head><body>
<h1>Slay the Spire RL — live runs</h1>
<div class="mut" id="hdr">loading…</div>
<div class="card" style="margin-top:12px"><h2>Runs</h2><div class="tw"><table id="runs"></table></div>
<div style="margin-top:8px" id="sel"></div></div>
<div class="grid">
<div class="card"><h2>Greedy eval: mean floor (1,024 TRAIN-eval worlds)</h2><canvas id="c_eval"></canvas></div>
<div class="card"><h2>Paired gain vs original (floors, ±1.96 SE)</h2><canvas id="c_paired"></canvas></div>
<div class="card"><h2>Boss pass rate (Act 1 solid, Act 2 dashed)</h2><canvas id="c_boss"></canvas></div>
<div class="card"><h2>Training-time floor (collection policy)</h2><canvas id="c_train"></canvas></div>
<div class="card"><h2>Combat runs: held-out fight win rate (Act 2 boss solid, Act 2 elite dashed, Act 3 boss dotted)</h2><canvas id="c_panel"></canvas></div>
<div class="card"><h2>Gradient signal: half-batch cosine (OOC actor)</h2><canvas id="c_cos"></canvas></div>
<div class="card"><h2>Displacement / Adam random-walk line (scorer)</h2><canvas id="c_disp"></canvas></div>
<div class="card"><h2>Behaviour: smith rate (solid), card skip (dashed)</h2><canvas id="c_beh"></canvas></div>
<div class="card"><h2>Behaviour: map monster (solid), event (dashed)</h2><canvas id="c_map"></canvas></div>
</div>
<script>
const COLORS=['#2f6fdb','#d9480f','#2b8a3e','#ae3ec9','#e67700','#0b7285','#c2255c','#5c940d','#495057','#1971c2'];
let charts={}, selected=null;
function color(i){return COLORS[i%COLORS.length]}
function mk(id,ds,opts={}){const ctx=document.getElementById(id);if(charts[id])charts[id].destroy();
 charts[id]=new Chart(ctx,{type:'line',data:{datasets:ds},options:Object.assign({animation:false,parsing:false,
 plugins:{legend:{labels:{boxWidth:10,font:{size:11}}}},scales:{x:{type:'linear',title:{display:true,text:'update'}}}},opts)});}
function pts(arr,xk,yk){return arr.filter(r=>r[yk]!=null&&r[yk]===r[yk]).map(r=>({x:r[xk],y:r[yk]}))}
function fmt(v,d=2){return v==null?'–':Number(v).toFixed(d)}
async function load(){
 const s=await (await fetch('/api/state')).json();
 const rn=s.runner; const age=rn?Math.round(s.now-rn.heartbeat):null;
 document.getElementById('hdr').textContent=(rn?`runner pid ${rn.pid}, heartbeat ${age}s ago, ${rn.on_battery?'ON BATTERY (paused)':'AC power'}, active: ${rn.active.join(', ')||'none'}`:'runner not started')+
   (s.baseline?` · original baseline ${fmt(s.baseline.mean_floor)} ± ${fmt(s.baseline.se)} floors, Act1 boss ${fmt(100*s.baseline.act1_boss_pass,1)}%, Act2 boss ${fmt(100*s.baseline.act2_boss_pass,1)}%, wins ${s.baseline.wins}/${s.baseline.n}`:'')+
   ` · updated ${new Date(s.now*1000).toLocaleTimeString()}`;
 const names=Object.keys(s.runs); const qstat={}; s.queue.forEach(q=>qstat[q.name]=q.status);
 if(selected===null){selected=new Set(names)}
 names.forEach(n=>{if(!(n in (window._seen||{})))selected.add(n)}); window._seen=Object.fromEntries(names.map(n=>[n,1]));
 let html='<tr><th>run</th><th>status</th><th>updates</th><th>last eval</th><th>paired Δ vs orig</th><th>Act1/Act2 boss</th><th>wins</th><th>notes</th></tr>';
 names.forEach((n,i)=>{const r=s.runs[n];const e=r.evals[r.evals.length-1]||{};const p=e.paired;
  const st=qstat[n]||(r.done?'done':'?');
  html+=`<tr><td style="color:${color(i)}">■ ${n}</td><td><span class="pill ${st}">${st}</span></td><td>${r.train.length}/${r.config.iterations||'?'}</td>
  <td>${fmt(e.mean_floor)} ± ${fmt(e.se)}</td><td>${p?`${p.diff>=0?'+':''}${fmt(p.diff)} ± ${fmt(p.se)}`:'–'}</td>
  <td>${fmt(100*(e.act1_boss_pass||0),1)}% / ${fmt(100*(e.act2_boss_pass||0),1)}%</td><td>${e.wins??'–'}</td><td class="mut">${(r.config.notes||'').slice(0,90)}</td></tr>`});
 s.queue.filter(q=>!(q.name in s.runs)).forEach(q=>{html+=`<tr><td class="mut">${q.name}</td><td><span class="pill">${q.status}</span></td><td colspan=6></td></tr>`});
 document.getElementById('runs').innerHTML=html;
 document.getElementById('sel').innerHTML=names.map((n,i)=>`<label><input type="checkbox" data-n="${n}" ${selected.has(n)?'checked':''}> <span style="color:${color(i)}">${n}</span></label>`).join('');
 document.querySelectorAll('#sel input').forEach(cb=>cb.onchange=()=>{cb.checked?selected.add(cb.dataset.n):selected.delete(cb.dataset.n);draw(s)});
 draw(s);
}
function draw(s){
 const names=Object.keys(s.runs);const ds=f=>names.map((n,i)=>[n,i]).filter(([n])=>selected.has(n)).flatMap(([n,i])=>f(n,i,s.runs[n]));
 const base=s.baseline?[{label:'original',data:[{x:0,y:s.baseline.mean_floor},{x:60,y:s.baseline.mean_floor}],borderColor:'#888',borderDash:[4,4],pointRadius:0}]:[];
 mk('c_eval',[...base,...ds((n,i,r)=>[{label:n,data:pts(r.evals,'iteration','mean_floor'),borderColor:color(i),backgroundColor:color(i)}])]);
 mk('c_paired',ds((n,i,r)=>{const e=r.evals.filter(x=>x.paired);return[
  {label:n,data:e.map(x=>({x:x.iteration,y:x.paired.diff})),borderColor:color(i),backgroundColor:color(i)},
  {label:n+' hi',data:e.map(x=>({x:x.iteration,y:x.paired.diff+1.96*x.paired.se})),borderColor:color(i),borderDash:[2,3],pointRadius:0,borderWidth:1},
  {label:n+' lo',data:e.map(x=>({x:x.iteration,y:x.paired.diff-1.96*x.paired.se})),borderColor:color(i),borderDash:[2,3],pointRadius:0,borderWidth:1}]}),
  {plugins:{legend:{labels:{filter:l=>!/ (hi|lo)$/.test(l.text),boxWidth:10}}}});
 mk('c_boss',ds((n,i,r)=>[{label:n+' A1',data:pts(r.evals,'iteration','act1_boss_pass'),borderColor:color(i)},
  {label:n+' A2',data:pts(r.evals,'iteration','act2_boss_pass'),borderColor:color(i),borderDash:[5,4]}]));
 mk('c_train',ds((n,i,r)=>[{label:n,data:pts(r.train,'iteration','collect_floor'),borderColor:color(i),pointRadius:1}]));
 mk('c_panel',ds((n,i,r)=>(r.panels||[]).length?[{label:n+' BOSS2',data:pts(r.panels,'iteration','BOSS2'),borderColor:color(i),pointRadius:1},
  {label:n+' ELITE2',data:pts(r.panels,'iteration','ELITE2'),borderColor:color(i),borderDash:[5,4],pointRadius:1},
  {label:n+' BOSS3',data:pts(r.panels,'iteration','BOSS3'),borderColor:color(i),borderDash:[1,3],pointRadius:1}]:[]));
 mk('c_cos',ds((n,i,r)=>[{label:n,data:pts(r.train,'iteration','grad_cosine_half'),borderColor:color(i),pointRadius:1}]));
 mk('c_disp',ds((n,i,r)=>[{label:n,data:pts(r.train,'iteration','disp_ratio_scorer'),borderColor:color(i),pointRadius:1}]));
 mk('c_beh',ds((n,i,r)=>[{label:n+' smith',data:pts(r.train,'iteration','smith_rate'),borderColor:color(i),pointRadius:1},
  {label:n+' skip',data:pts(r.train,'iteration','card_skip_rate'),borderColor:color(i),borderDash:[5,4],pointRadius:1}]));
 mk('c_map',ds((n,i,r)=>[{label:n+' monster',data:pts(r.train,'iteration','map_monster'),borderColor:color(i),pointRadius:1},
  {label:n+' event',data:pts(r.train,'iteration','map_event'),borderColor:color(i),borderDash:[5,4],pointRadius:1}]));
}
load();setInterval(load,30000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/state"):
            body, kind = json.dumps(state()).encode(), "application/json"
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


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "serve"
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else PORT
    if command == "start":
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=3)
            print(f"dashboard already up at http://127.0.0.1:{port}")
            return
        except Exception:
            pass
        RUNS.mkdir(parents=True, exist_ok=True)
        subprocess.Popen([sys.executable, __file__, "serve", "--port", str(port)], cwd=ROOT, start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=(RUNS / "dashboard.log").open("a"), stderr=subprocess.STDOUT)
        print(f"dashboard started at http://127.0.0.1:{port}")
        return
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
