#!/usr/bin/env python3
"""Read-only localhost fresh-study dashboard. No external assets or actions."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

HTML = r'''<!doctype html><meta charset="utf-8"><title>Fresh study</title>
<style>body{font:15px system-ui;max-width:1100px;margin:30px auto;background:#f4f6fa;color:#17213a}canvas{background:white;width:100%;height:230px}pre{white-space:pre-wrap;overflow-wrap:anywhere}section{background:white;padding:14px;margin:16px 0}.legend span{margin-right:25px}</style>
<h1>Fresh study • read-only</h1><p>Supervised validation and greedy evaluation are separate from stochastic training rollouts.</p>
<div class="legend"><span style="color:#2766c4">● FIT diagnostic CE</span><span style="color:#d75c20">● Supervised VALIDATION CE</span><span style="color:#178467">● Fixed93e9 probe KL</span><span style="color:#777">● FIT diagnostic KL</span></div>
<div id="state"></div>
<script>
function line(ctx,rows,fn,color,max){ctx.strokeStyle=color;ctx.beginPath();rows.forEach((r,i)=>{let y=fn(r);if(y==null)return;let x=40+i*970/Math.max(1,rows.length-1);let v=210-y*190/max;i?ctx.lineTo(x,v):ctx.moveTo(x,v)});ctx.stroke()}
function draw(canvas,rows){let c=canvas.getContext('2d');let max=Math.max(.001,...rows.map(r=>r.validation?.loss||0),...rows.map(r=>r.fit_diagnostic?.loss||0),...rows.map(r=>r.fixed93e9_probes?.pooled?.kl_previous_to_current||0),...rows.map(r=>r.movement?.kl_previous_to_current||0));c.fillText('CE / movement KL; horizontal axis: logged supervised update blocks',40,15);line(c,rows,r=>r.fit_diagnostic?.loss,'#2766c4',max);line(c,rows,r=>r.validation?.loss,'#d75c20',max);line(c,rows,r=>r.fixed93e9_probes?.pooled?.kl_previous_to_current,'#178467',max);line(c,rows,r=>r.movement?.kl_previous_to_current,'#777',max)}
async function poll(){if(document.hidden)return;let s=await(await fetch('/api/state',{cache:'no-store'})).json();let el=document.getElementById('state');el.replaceChildren();for(let r of s.runs){let box=document.createElement('section');let h=document.createElement('h2');h.textContent=r.name;box.append(h);let canvas=document.createElement('canvas');canvas.width=1050;canvas.height=230;box.append(canvas);draw(canvas,r.metrics);let p=document.createElement('pre');p.textContent=JSON.stringify({status:r.status,config:r.config,coverage:r.coverage,summary:r.summary,receipt:r.receipt,latest_metrics:r.metrics.at(-1)},null,2);box.append(p);el.append(box)}let reports=document.createElement('pre');reports.textContent='Paired outcome comparisons / intervals\n'+JSON.stringify(s.comparisons,null,2);el.append(reports)}
poll();setInterval(()=>poll().catch(console.error),5000);
</script>'''


def safe_json(path, root):
    if not path.is_file() or not path.resolve().is_relative_to(root):
        return None
    try:
        return json.loads(path.read_text())
    except (OSError,ValueError):
        return None


def study_state(directory):
    root=Path(directory).resolve()
    # Only root + immediate declared study children; never follow links outside.
    runs=[]
    candidates=[root]+sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink())
    for path in candidates:
        config=safe_json(path/'config.json',root)
        status=safe_json(path/'status.json',root)
        summary=safe_json(path/'summary.json',root)
        if not any((config,status,summary)):
            continue
        metrics=[]
        log=path/'metrics.jsonl'
        if log.is_file() and log.resolve().is_relative_to(root):
            from collections import deque
            with log.open() as stream:
                for line in deque(stream,maxlen=1000):
                    try:
                        metrics.append(json.loads(line))
                    except ValueError:
                        pass
        runs.append(dict(name=path.name,status=status,config=config,summary=summary,metrics=metrics,
                         coverage=safe_json(path/'coverage.json',root),receipt=safe_json(path/'receipt.json',root)))
    comparisons = {p.name:safe_json(p,root) for p in sorted((root/'comparisons').glob('*.json')) if not p.is_symlink()}
    return {'runs':runs,'comparisons':comparisons,'read_only':True,'network':'localhost only'}


def make_server(directory, port=0):
    root=Path(directory).resolve(strict=True)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path=urlsplit(self.path).path
            if path=='/api/state':
                body=json.dumps(study_state(root),allow_nan=False).encode()
                content='application/json'
            elif path=='/':
                body=HTML.encode()
                content='text/html; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type',content)
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'")
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self,*args):
            pass
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-dir',required=True)
    p.add_argument('--port',type=int,default=8379)
    a=p.parse_args(argv)
    server=make_server(a.study_dir,a.port)
    print('http://127.0.0.1:'+str(server.server_port),flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()

if __name__=='__main__':
    main()
