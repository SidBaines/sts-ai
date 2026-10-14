#!/usr/bin/env python3
"""Compare canonical native query statistics in two fresh interpreter processes."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))

def main():
    from sts_rl.recording import begin_run,file_identity
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True);parser.add_argument('--checkpoint',required=True)
    args=parser.parse_args();os.chdir(ROOT);out=Path(args.out)
    if (out/'acceptance.json').exists():parser.error('Fresh output required')
    record=begin_run(out,vars(args),args.checkpoint)
    code=r"""
import json,hashlib
from sts_env._backend.lightspeed import LightspeedHybridEnv
from tests.integration.test_public_combat_teacher import _advance_to_first_combat
results=[]
for seed in [3,7,11,19]:
 env=LightspeedHybridEnv(world_seed=seed,max_act=1,combat_control='llm',combat_observation='combat_public_v3')
 _advance_to_first_combat(env)
 before=env.describe_state()
 r=env.search_best_combat_action(100,search_seed=173,draw_order_seed=23)
 assert env.describe_state()==before and r['provenance']['hidden_pile_order_canonicalized']
 results.append({'seed':seed,'bits':int(r['bits']),'description':r['description'],'selection':r['selection_method'],
                 'edges':[(int(e['bits']),int(e['visits']),e['evaluation_sum']) for e in r['root_edges']],
                 'public_text_sha256':hashlib.sha256(before.encode()).hexdigest()})
print(json.dumps(results,sort_keys=True))
"""
    env={**os.environ,'PYTHONPATH':str(ROOT/'src')+os.pathsep+str(ROOT)}
    outputs=[]
    for i in range(2):
        p=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=120)
        (out/f'process_{i}.stdout.json').write_text(p.stdout);(out/f'process_{i}.stderr.log').write_text(p.stderr)
        if p.returncode:raise RuntimeError(f'Fresh process {i} failed {p.returncode}')
        outputs.append(json.loads(p.stdout))
    if outputs[0]!=outputs[1]:raise RuntimeError('Fresh native query statistics differ')
    (out/'acceptance.json').write_text(json.dumps({'run_record':record,'worlds':4,'fresh_processes':2,
           'exact_query_statistics_and_public_text':True,'live_states_unchanged':True,
           'outputs':[file_identity(out/f'process_{i}.stdout.json') for i in range(2)]},indent=2)+'\n')
    print('Fresh-process canonical queries and non-mutation pass on four worlds',flush=True)

if __name__=='__main__':main()
