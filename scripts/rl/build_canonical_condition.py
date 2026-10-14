#!/usr/bin/env python3
"""Prepare isolated core/RL builds and the native regression fixture, offline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))

def run(command,**kwargs):subprocess.run(command,check=True,timeout=600,**kwargs)

def main():
    from sts_rl.recording import begin_run,file_identity
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--revision',type=int,default=1)
    args=parser.parse_args();os.chdir(ROOT);out=Path(args.out)
    suffix='' if args.revision==1 else '_v'+str(args.revision)
    if (out/'summary.json').exists():parser.error('Fresh preparation output required')
    record=begin_run(out,vars(args),args.checkpoint,inputs=[ROOT/'tests/native/hidden_pile_fixture.cpp'],preparation=True)
    source=ROOT/'external/sts_lightspeed';core=source/('build'+suffix)
    env={**os.environ,'PYTHON':str(ROOT/'.venv-native/bin/python'),'STS_OFFLINE_DEPS':'1',
         'STS_BUILD_JOBS':'2','STS_LIGHTSPEED_SOURCE':str(source),'STS_LIGHTSPEED_BUILD':str(core)}
    run(['bash',str(ROOT/'scripts/build_lightspeed.sh')],env=env)
    rl=ROOT/('external/sts_lightspeed_rl'+suffix)
    if rl.exists():raise RuntimeError('Fresh RL source required; do not overwrite an existing build')
    run(['git','clone','--no-hardlinks','--no-checkout',str(source),str(rl)])
    run(['git','-C',str(rl),'checkout','--detach','7476a81954020087da31d41d16fddf475746ec2d'])
    run(['git','-C',str(rl),'apply',str(ROOT/'patches/sts_lightspeed_python_api.patch')])
    for dep in ['json','pybind11']:
        target=rl/dep
        if target.exists():target.rmdir()
        run(['git','clone','--no-hardlinks',str(source/dep),str(target)])
    run(['git','-C',str(rl),'add','bindings','include','src'])
    run(['git','-C',str(rl),'-c','user.name=sts_rl','-c','user.email=rl@local','commit','-m','Versioned canonical sampler environment patch'])
    run(['git','-C',str(rl),'apply',str(ROOT/'patches/sts_lightspeed_rl_panel.patch')])
    run(['bash',str(ROOT/'scripts/rl/build_native_rl.sh')],env={**env,'BUILD_DIR':'build014_canonical','STS_RL_SOURCE':str(rl)})
    fixture=out/('sts_hidden_pile_fixture'+sysconfig.get_config_var('EXT_SUFFIX'));out.mkdir(parents=True,exist_ok=True)
    compiler='/Library/Developer/CommandLineTools/usr/bin/c++'
    command=[compiler,'-std=gnu++17','-O2','-arch','arm64','-isysroot','/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk',
             '-bundle','-undefined','dynamic_lookup','-I'+str(source/'include'),'-I'+str(source/'pybind11/include'),
             '-I'+sysconfig.get_paths()['include'],str(ROOT/'tests/native/hidden_pile_fixture.cpp'),'-o',str(fixture)]
    run(command)
    report={'run_record':record,'core_manifest':json.loads((core/'build_manifest.json').read_text()),
            'rl_manifest':json.loads((rl/'build014_canonical/build_manifest.json').read_text()),
            'test_fixture':file_identity(fixture),'fixture_compile_command':command}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print('Canonical condition built; acceptance tests still required',flush=True)

if __name__=='__main__':main()
