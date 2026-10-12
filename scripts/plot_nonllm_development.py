#!/usr/bin/env python3
"""Reproducible development-only plots and source tables, without model imports.

SVG needs only the standard library. --png additionally needs optional Pillow;
the Codex bundled Python provides it without changing the environment core.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo

BLUE, ORANGE, TEAL = "#2563a6", "#d07b22", "#218a83"
INK, MUTED, GRID = "#172b40", "#607286", "#e5ebf1"
SUCCESS = {"battle_victory", "act_clear", "victory", "PLAYER_VICTORY"}
FAILURE = {"battle_defeat", "defeat", "PLAYER_LOSS"}


def episode_summary(row):
    episodes = row.get("episodes", [])
    faults = row.get("faults", [])
    for episode in episodes + faults:
        if episode.get("split") not in (None, "dev"):
            raise ValueError("development plotting refuses non-development episodes")
    valid = [e for e in episodes if not e.get("fault", False)]
    decisive = [e for e in valid if e.get("terminated") and not e.get("truncated") and
                str(e.get("task_outcome") or e.get("outcome", "")).rsplit(".", 1)[-1] in SUCCESS | FAILURE]
    wins = sum(str(e.get("task_outcome") or e.get("outcome", "")).rsplit(".", 1)[-1] in SUCCESS for e in decisive)
    censored = sum(bool(e.get("truncated")) for e in valid)
    hp_values = [float(e['hp_cost']) for e in valid if e.get('hp_cost') is not None]
    task_ids = row.get('task_ids') or [e.get('task_id') for e in episodes]
    return {"wins": wins, "valid_episodes": len(valid), "decisive": len(decisive),
            "faults": len(faults) + sum(bool(e.get("fault")) for e in episodes),
            "truncated": censored, "unresolved": len(valid) - len(decisive),
            "win_rate_decisive": wins / len(decisive) if decisive else None,
            "mean_hp_cost": sum(hp_values) / len(hp_values) if hp_values else None,
            "worlds": len({e.get("world_seed") for e in valid}),
            "cohort_id": hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()[:12]
                if task_ids and all(task_ids) else None}


def phase_label(name):
    if re.fullmatch(r"combat_\d{4}", name):
        return "combat training rounds"
    if "hardening" in name:
        return "battle data expansion"
    if "pilot" in name:
        return "PPO hyperparameter pilots"
    return "initial combat evaluation"


def collect_data(campaign, *, timezone="Europe/London"):
    sources = []
    def read(path):
        raw = path.read_bytes()
        sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, ZoneInfo("UTC")).isoformat()})
        result = json.loads(raw)
        if result.get("config", {}).get("split") not in (None, "dev"):
            raise ValueError(f"development plotting refuses split in {path}")
        return result
    state = read(campaign / "campaign.json")
    started = float(state["started_at"])
    commands = {row["name"]: row for row in state.get("commands", [])}
    combat = []
    for path in sorted(campaign.glob("combat_*_dev_combined.metrics.json")):
        metric = read(path)
        name = path.name.removesuffix("_dev_combined.metrics.json")
        rounds = metric.get("rounds", [])
        if not rounds:
            continue
        stamp = max([float(commands[key].get("finished", commands[key].get("started", 0)))
                     for key in (name + "_dev_act1", name + "_dev_hard", name + "_dev") if key in commands]
                    or [path.stat().st_mtime])
        primary = episode_summary(rounds[0])
        hard = episode_summary(rounds[1]) if len(rounds) > 1 else None
        score = metric.get("selection_score")
        combat.append({"name": name, "phase": phase_label(name), "timestamp": stamp,
                       "local_time": datetime.fromtimestamp(stamp, ZoneInfo(timezone)).isoformat(),
                       "hours_since_start": (stamp - started) / 3600,
                       "selection_score": score if score is not None and math.isfinite(score) else None,
                       "primary_score": metric.get("primary_score"), "hard_score": metric.get("hard_score"),
                       "act1": primary, "later_acts": hard,
                       "comparison_cohort": ((primary["cohort_id"] + ":" + (hard["cohort_id"] if hard else "none"))
                                             if primary['cohort_id'] and (hard is None or hard['cohort_id']) else 'unknown:' + path.name),
                       "source": str(path.resolve())})
    architecture = []
    for row in state.get("architecture_panel", []):
        name = Path(row["checkpoint"]).stem
        architecture.append({"name": name, "score": row["score"], "checkpoint": row["checkpoint"]})
    architecture.sort(key=lambda row: row["score"], reverse=True)
    health = []
    health_studies = []
    for path in sorted(campaign.glob("health_ablation_*/selection.json")):
        selection = read(path)
        health_studies.append({"source": str(path.resolve()), "selected": selection.get("selected_condition"),
                               "timestamp": path.stat().st_mtime,
                               "criterion": selection.get("criterion"), "reasons": selection.get("exclusion_reasons", {})})
        for condition in ("reference", "hp0.25", "hp1.0"):
            for cohort in ("act1_48", "expanded_hard32", "composite_act1_24"):
                metrics = path.parent / f"{condition}_{cohort}.metrics.json"
                if not metrics.exists():
                    continue
                row = read(metrics)
                rounds = row.get("rounds", [])
                aggregate = {"episodes": [e for r in rounds for e in r.get("episodes", [])],
                             "faults": [f for r in rounds for f in r.get("faults", [])],
                             "task_ids": [task for r in rounds for task in r.get("task_ids", [])]}
                health.append({"study": path.parent.name, "condition": condition, "cohort": cohort,
                               "selected": condition == selection.get("selected_condition"),
                               **episode_summary(aggregate), "source": str(metrics.resolve())})
    return {"scope": "development only; no final metrics read", "timezone": timezone,
            "started_at": started, "combat": sorted(combat, key=lambda r: r["timestamp"], reverse=True),
            "architecture": architecture, "health": health,
            "health_studies": sorted(health_studies, key=lambda row: row['timestamp']),
            "sources": sources,
            "notes": ["Selection score is a heuristic combining success, floor and HP; it is not a win percentage.",
                      "Only identical task cohort IDs can support comparable connected progress curves.",
                      "Faults and truncated outcomes are separate from decisive losses.",
                      "Repeated development selection does not establish final-test performance."]}


class Canvas:
    def __init__(self, width, height, *, png=False):
        self.width, self.height = width, height
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                    '<rect width="100%" height="100%" fill="white"/>']
        self.image = None
        if png:
            from PIL import Image, ImageDraw, ImageFont
            self.ImageFont = ImageFont
            self.image = Image.new("RGB", (width * 2, height * 2), "white")
            self.draw = ImageDraw.Draw(self.image)
            self.fonts = {}
    def line(self, x1, y1, x2, y2, color=GRID, width=1):
        self.svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>')
        if self.image:
            self.draw.line((x1*2,y1*2,x2*2,y2*2), fill=color, width=max(1,int(width*2)))
    def rect(self, x, y, width, height, color):
        self.svg.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{color}"/>')
        if self.image:
            self.draw.rectangle((x*2,y*2,(x+width)*2,(y+height)*2), fill=color)
    def circle(self, x, y, radius, color):
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}"/>')
        if self.image:
            self.draw.ellipse(((x-radius)*2,(y-radius)*2,(x+radius)*2,(y+radius)*2),fill=color)
    def text(self, x, y, value, size=18, color=INK, anchor="start", bold=False):
        value = str(value)
        self.svg.append(f'<text x="{x}" y="{y}" dominant-baseline="hanging" text-anchor="{anchor}" font-family="Helvetica,Arial,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escape(value)}</text>')
        if self.image:
            key = (size,bold)
            if key not in self.fonts:
                candidates = [Path('/System/Library/Fonts/Supplemental') / ('Arial Bold.ttf' if bold else 'Arial.ttf'),
                              Path('/System/Library/Fonts/Helvetica.ttc'),
                              Path('/usr/share/fonts/truetype/dejavu') / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')]
                font_path = next((p for p in candidates if p.exists()), None)
                self.fonts[key] = self.ImageFont.truetype(str(font_path),size*2) if font_path else self.ImageFont.load_default(size=size*2)
            font = self.fonts[key]
            text_width = self.draw.textlength(value,font=font)
            left = x*2 - (text_width/2 if anchor=="middle" else text_width if anchor=="end" else 0)
            self.draw.text((left,y*2),value,font=font,fill=color,anchor="lt")
    def save(self, path):
        path.with_suffix('.svg').write_text('\n'.join(self.svg + ['</svg>'])+'\n')
        if self.image:
            self.image.save(path.with_suffix('.png'))


def combat_chart(data, destination, png):
    rows = sorted([r for r in data['combat'] if r['phase']=='combat training rounds' and r['selection_score'] is not None],key=lambda r:r['timestamp'])
    if not rows:
        return []
    c = Canvas(1400,810,png=png)
    c.text(70,28,'Combat training: latest development trend',32,bold=True)
    c.text(70,77,'Connected points share the same battle cohorts. Scores are selection heuristics, not win percentages.',18,MUTED)
    x0,x1,y0,y1 = 100,1240,150,540
    xs = [r['hours_since_start'] for r in rows]
    xmin,xmax = min(xs),max(xs)
    if xmax==xmin: xmax=xmin+.1
    numeric = [r[key] for r in rows for key in ('selection_score','primary_score','hard_score') if r.get(key) is not None and math.isfinite(r[key])]
    low = math.floor((min(numeric)-3)/5)*5; high=math.ceil((max(numeric)+3)/5)*5
    def px(x): return x0+(x-xmin)/(xmax-xmin)*(x1-x0)
    def py(y): return y1-(y-low)/(high-low)*(y1-y0)
    for score in range(int(low),int(high)+1,5):
        c.line(x0,py(score),x1,py(score)); c.text(x0-14,py(score)-9,score,16,MUTED,'end')
    for i in range(6):
        hour=xmin+(xmax-xmin)*i/5
        c.text(px(hour),y1+15,f'{hour:.2f}h',16,MUTED,'middle')
    for key,color,label in [('primary_score',BLUE,'Act 1 panel'),('hard_score',ORANGE,'Later-act panel'),('selection_score',TEAL,'Equal-panel selection score')]:
        for previous,current in zip(rows,rows[1:]):
            if previous['comparison_cohort']==current['comparison_cohort'] and all(
                    r.get(key) is not None and math.isfinite(r[key]) for r in (previous,current)):
                c.line(px(previous['hours_since_start']),py(previous[key]),px(current['hours_since_start']),py(current[key]),color,2)
        for row in rows:
            if row.get(key) is not None and math.isfinite(row[key]):
                c.circle(px(row['hours_since_start']),py(row[key]),4,color)
        legend_x={'primary_score':100,'hard_score':400,'selection_score':730}[key]
        c.circle(legend_x,y1+70,5,color);c.text(legend_x+14,y1+59,label,18,color)
    last=rows[-1]
    c.text(100,640,f'Latest completed checkpoint: {last["name"]}  |  score {last["selection_score"]:.2f}  |  {last["local_time"][11:16]} local',23,bold=True)
    def result(label,row):
        return f'{label}: {row["wins"]}/{row["decisive"]} decisive wins; faults {row["faults"]}; truncated {row["truncated"]}; {row["worlds"]} worlds'
    c.text(100,682,result('Act 1',last['act1']),18,MUTED)
    if last['later_acts']: c.text(100,712,result('Later acts',last['later_acts']),18,MUTED)
    c.text(100,755,'Repeated development measurements guide selection; this chart does not show final-test results.',17,MUTED)
    path=destination/'combat_development';c.save(path)
    return [str(path.with_suffix('.svg'))]+([str(path.with_suffix('.png'))] if png else [])


def architecture_chart(data,destination,png):
    rows=data['architecture']
    if not rows: return []
    c=Canvas(1400,225+len(rows)*36,png=png)
    c.text(70,28,'Model architecture and learning-rate selection',32,bold=True)
    c.text(70,77,'Behavioral-cloning checkpoints ranked by development selection score; higher is better.',18,MUTED)
    x0,x1=430,1210
    for tick in range(0,101,20):
        x=x0+(x1-x0)*tick/100;c.line(x,118,x,130+len(rows)*36);c.text(x,140+len(rows)*36,tick,15,MUTED,'middle')
    for i,row in enumerate(rows):
        y=125+i*36
        label=row['name'].removeprefix('bc_').replace('_lr','  lr ').replace('_rep','  seed ').replace('_',' ')
        c.text(70,y+3,label,17,INK,bold=i==0)
        width=(x1-x0)*max(0,min(100,row['score']))/100
        c.rect(x0,y,width,24,BLUE if i==0 else '#a8bdd3')
        c.text(x0+width+12,y+3,f'{row["score"]:.2f}',17,INK,bold=i==0)
    c.text(70,177+len(rows)*36,'Selection score is not a win percentage; source CSV preserves all candidate names and checkpoint paths.',16,MUTED)
    path=destination/'architecture_development';c.save(path)
    return [str(path.with_suffix('.svg'))]+([str(path.with_suffix('.png'))] if png else [])


def health_chart(data,destination,png):
    if not data['health_studies']: return []
    study=data['health_studies'][-1]
    name=Path(study['source']).parent.name
    rows=[r for r in data['health'] if r['study']==name]
    c=Canvas(1400,770,png=png)
    c.text(70,28,'HP reward: local battle gains did not carry into full Act 1 runs',30,bold=True)
    c.text(70,75,f'Development-only reward ablation. Selected condition: {study["selected"]}. Same model start and fixed OOC partner.',18,MUTED)
    labels={'act1_48':'Act 1 battles','expanded_hard32':'Later-act battles','composite_act1_24':'Composite Act 1 runs'}
    for index,cohort in enumerate(labels):
        left=90+index*440; ytop,ybase=180,540
        c.text(left+170,125,labels[cohort],24,INK,'middle',True)
        for percent in (0,25,50,75,100):
            y=ybase-(ybase-ytop)*percent/100
            c.line(left,y,left+355,y);c.text(left-10,y-8,f'{percent}%',14,MUTED,'end')
        for j,(condition,color) in enumerate([('reference',BLUE),('hp0.25',ORANGE),('hp1.0',TEAL)]):
            row=next((r for r in rows if r['condition']==condition and r['cohort']==cohort),None)
            if row is None: continue
            rate=row['win_rate_decisive'] or 0
            x=left+25+j*108; height=(ybase-ytop)*rate
            c.rect(x,ybase-height,78,height,color)
            c.text(x+39,ybase-height-28,f'{row["wins"]}/{row["decisive"]}',19,color,'middle',True)
            c.text(x+39,ybase+15,condition.replace('hp','HP '),17,INK,'middle')
    for i,(condition,color) in enumerate([('reference',BLUE),('hp0.25',ORANGE),('hp1.0',TEAL)]):
        subset=[r for r in rows if r['condition']==condition]
        c.text(90+i*440,615,f'{condition}: faults {sum(r["faults"] for r in subset)}, truncated {sum(r["truncated"] for r in subset)}',18,MUTED)
    c.text(90,673,'Both HP conditions were rejected because composite clears regressed. Small development panels; no significance claim.',17,MUTED)
    c.text(90,710,'Counts include only decisive completed episodes. Faults and censoring remain separate in the source tables.',17,MUTED)
    path=destination/'health_development';c.save(path)
    return [str(path.with_suffix('.svg'))]+([str(path.with_suffix('.png'))] if png else [])


def csv_write(path,rows):
    if not rows: return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,default=Path('data/nonllm_v1/campaign'))
    parser.add_argument('--output',type=Path,default=Path('data/nonllm_v1/report'))
    parser.add_argument('--timezone',default='Europe/London')
    parser.add_argument('--png',action='store_true',help='add PNG using optional Pillow; SVG always generated')
    args=parser.parse_args()
    if args.png:
        try: import PIL
        except ImportError: parser.error('--png needs optional Pillow; omit it for standard-library SVG output')
    data=collect_data(args.campaign,timezone=args.timezone)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'development_source.json').write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')
    combat=[]
    for row in data['combat']:
        base={k:v for k,v in row.items() if k not in ('act1','later_acts')}
        for panel in ('act1','later_acts'):
            base.update({panel+'_'+k:v for k,v in (row[panel] or {}).items()})
        combat.append(base)
    # Normalize optional later-act fields to a common CSV header.
    fields=list(dict.fromkeys(key for row in combat for key in row))
    csv_write(args.output/'combat_development.csv',[{key:row.get(key) for key in fields} for row in combat])
    csv_write(args.output/'architecture_development.csv',data['architecture'])
    csv_write(args.output/'health_development.csv',data['health'])
    artifacts=[]
    for chart in (combat_chart,architecture_chart,health_chart):
        artifacts.extend(chart(data,args.output,args.png))
    manifest={'scope':data['scope'],'latest_combat':data['combat'][0]['name'] if data['combat'] else None,
              'artifacts':artifacts,'source_files':len(data['sources']),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'png_backend':'Pillow '+PIL.__version__ if args.png else None,
              'source_json':'development_source.json','csv_files':['combat_development.csv','architecture_development.csv','health_development.csv']}
    (args.output/'plot_manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    print(json.dumps(manifest,sort_keys=True))


if __name__=='__main__': main()
