#!/usr/bin/env python3
"""Bounded recorded acceptance for the isolated canonical-sampler candidate."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))

def main():
    from sts_rl.recording import begin_run,file_identity
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--build-record',default='data/canonical_20261014/build')
    args=parser.parse_args();os.chdir(ROOT);out=Path(args.out)
    if (out/'acceptance.json').exists():parser.error('Fresh acceptance output required')
    record=begin_run(out,vars(args),args.checkpoint)
    fixture=next((ROOT/args.build_record).glob('sts_hidden_pile_fixture*.so'))
    env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'STS_REQUIRE_SIMULATOR':'1',
         'STS_SEARCH_TEST_FIXTURE':str(fixture),'STS_REQUIRE_SEARCH_TEST_FIXTURE':'1'}
    command=([sys.executable,'-m','unittest','discover','-s','tests','-t','.'] if args.full else
             [sys.executable,'-m','unittest','tests.integration.test_canonical_search_sampling'])
    out.mkdir(parents=True,exist_ok=True)
    result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=300)
    (out/'tests.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError(f'Acceptance suite failed {result.returncode}; inspect tests.log')
    commands=[command]
    if args.full:
        smoke=[sys.executable,'scripts/run_rollout.py','--agent','random','--seed','3','--max-act','1',
               '--battle-simulations','50','--output',str(out/'smoke.jsonl')]
        result=subprocess.run(smoke,env=env,capture_output=True,text=True,timeout=120)
        (out/'smoke.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('Bounded smoke failed')
        commands.append(smoke)
    (out/'acceptance.json').write_text(json.dumps({'run_record':record,'full':args.full,'commands':commands,
             'test_log':file_identity(out/'tests.log'),'passed':True},indent=2)+'\n')
    print((out/'tests.log').read_text()[-1500:]);print('Acceptance passed',args.full,flush=True)

if __name__=='__main__':main()
