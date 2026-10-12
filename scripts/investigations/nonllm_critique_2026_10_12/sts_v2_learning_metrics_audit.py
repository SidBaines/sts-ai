from __future__ import annotations
import collections,hashlib,json,math,statistics
from pathlib import Path
BASE=Path('/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=BASE/'data/long_rl_v2/campaign/journal.json';journal=json.loads(p.read_text());jobs=[r for r in journal['jobs'] if r.get('kind')=='train' and r.get('status')=='complete']
 groups=collections.defaultdict(list);chunks=[];inputs=[]
 for j in jobs:
  m=json.loads(Path(j['metrics']).read_text());r=m['rounds'][0];u=r['update'];task_path=j['command'][j['command'].index('--tasks')+1];side=json.loads(Path(task_path+'.manifest.json').read_text());metas={x['task_id']:x for x in side['tasks']};chunks.append((j['round'],r,u,m));inputs.append({'round':j['round'],'metrics':j['metrics'],'metrics_sha256':sha(j['metrics']),'task_manifest':task_path+'.manifest.json','task_manifest_sha256':sha(task_path+'.manifest.json')})
  for e in r['episodes']:
   meta=metas[e['task_id']];groups[(meta['start_act'],meta['start_source'])].append(e)
 groupstats={str(k):{'valid_episodes':len(es),'transitions':sum(e['steps'] for e in es),'median_steps':statistics.median(e['steps'] for e in es),'wins':sum(e['outcome']=='victory' for e in es),'mean_return':statistics.mean(e['reward'] for e in es)} for k,es in groups.items()}
 windows={}
 for name,subset in [('first20',chunks[:20]),('last20',chunks[-20:]),('all',chunks)]:
  fresh=[e for i,r,u,m in subset for e in r['episodes'] if e['start_state']['floor']==0]
  windows[name]={'rounds':[i for i,r,u,m in subset],'update_metric_medians':{k:statistics.median(u[k] for i,r,u,m in subset) for k in ('approx_kl','clip_fraction','entropy','policy_loss','value_loss')},'early_stops':sum(u['early_stop'] for i,r,u,m in subset),'mean_effective_epochs':statistics.mean(u['steps']/math.ceil(r['transitions']/u['config']['minibatch_size']) for i,r,u,m in subset),'collection_seconds':sum(r['elapsed_seconds'] for i,r,u,m in subset),'update_seconds':sum(u['elapsed_seconds'] for i,r,u,m in subset),'child_seconds':sum(m['elapsed_seconds'] for i,r,u,m in subset),'fresh_valid_episodes':len(fresh),'fresh_mean_floor':statistics.mean(e['floor'] for e in fresh),'fresh_act1_boundary_floor17_count':sum(e['floor']>=17 for e in fresh),'fresh_act2_boundary_floor34_count':sum(e['floor']>=34 for e in fresh),'fresh_wins':sum(e['outcome']=='victory' for e in fresh)}
 out={'scope':'read-only frozen completed v2 jobs; descriptive unpaired rotating TRAIN windows, no causal comparison','journal':str(p),'journal_sha256':sha(p),'chunks':len(chunks),'valid_transitions':sum(r['transitions'] for i,r,u,m in chunks),'optimizer_steps':sum(u['steps'] for i,r,u,m in chunks),'groups':groupstats,'windows':windows,'inputs':inputs}
 Path('/private/tmp/sts_v2_learning_metrics_audit.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k!='inputs'},indent=2))
if __name__=='__main__':main()
