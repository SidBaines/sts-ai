"""Find real TRAIN map candidate collisions with different route options."""
from __future__ import annotations
from functools import lru_cache
import json
import re
import argparse
from pathlib import Path
from sts_policy.encoding import encode_observation

parser=argparse.ArgumentParser()
parser.add_argument('--teacher',default='/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act1_200_native011_seed10000/teacher.jsonl')
parser.add_argument('--max-train-rows',type=int,default=30000)
parser.add_argument('--output',default='/private/tmp/nonllm_map_alias_results.json')
args=parser.parse_args()
teacher=Path(args.teacher)
counts={'train_rows':0,'map_rows':0,'identical_map_candidate_pairs':0,'different_route_statistics_pairs':0}
examples=[]
with teacher.open() as handle:
    for line in handle:
        row=json.loads(line)
        if row.get('split')!='train':continue
        counts['train_rows']+=1
        obs=row['observation']
        actions=obs['legal_actions']
        if len(actions)>1 and all('map node x=' in a['description'] for a in actions):
            counts['map_rows']+=1
            graph=obs['state']['map'];by={(n['x'],n['y']):n for n in graph['nodes']}
            @lru_cache(maxsize=None)
            def routes(coordinate):
                n=by[coordinate]
                branches=[routes((x,n['y']+1)) for x in n['edges'] if (x,n['y']+1) in by]
                e=int(n['room'] in ('E','ELITE'));r=int(n['room'] in ('R','REST'))
                return {'min_elites':e+min((v['min_elites'] for v in branches),default=0),
                        'max_elites':e+max((v['max_elites'] for v in branches),default=0),
                        'min_rests':r+min((v['min_rests'] for v in branches),default=0),
                        'max_rests':r+max((v['max_rests'] for v in branches),default=0),
                        'path_count':sum(v['path_count'] for v in branches) if branches else 1}
            encoded=encode_observation(obs);seen={}
            for a,features in zip(actions,encoded.action_features):
                x=int(re.search(r'map node x=(\d+)',a['description'])[1])
                coordinate=(x,graph['cur_y']+1)
                stats=routes(coordinate)
                key=features.tobytes()
                for other_a,other_stats in seen.get(key,[]):
                    counts['identical_map_candidate_pairs']+=1
                    if stats!=other_stats:
                        counts['different_route_statistics_pairs']+=1
                        if len(examples)<5 or any(stats[k]!=other_stats[k] for k in ('min_elites','max_elites','min_rests','max_rests')):
                            examples.append({'world_seed':row['world_seed'],'step':row['step'],
                                'first_action':other_a,'second_action':a,'first_route_stats':other_stats,
                                'second_route_stats':stats,'public_observation':obs})
                seen.setdefault(key,[]).append((a,stats))
        if counts['train_rows']>=args.max_train_rows:break
result={'counts':counts,'examples':examples,'teacher':str(teacher),'scope':'Read only, existing TRAIN artifacts. Path statistics are descriptive, no teacher or future observations.'}
Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'counts':counts,'examples':[{k:v for k,v in e.items() if k!='public_observation'} for e in examples]},indent=2))
