#!/usr/bin/env python3
"""Compile a bounded standalone search-invariance probe from immutable native objects."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))


def main():
    from sts_rl.recording import begin_run,file_identity
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--worlds',type=int,default=16)
    parser.add_argument('--samples',type=int,default=24)
    parser.add_argument('--simulations',type=int,default=100)
    args=parser.parse_args();out=Path(args.out)
    if (out/'summary.json').exists():parser.error('Fresh output required')
    root=Path(__file__).resolve().parents[2];native=root/'external/sts_lightspeed_rl'
    objects=sorted((native/'build013_clean/CMakeFiles/slaythespire.dir/src').rglob('*.o'))
    if not objects:raise RuntimeError('Missing accepted native object files')
    source=root/'scripts/rl/probe_hidden_order.cpp'
    record=begin_run(out,vars(args),args.checkpoint,inputs=[source,*objects])
    executable=out/'probe_hidden_order';out.mkdir(parents=True,exist_ok=True)
    command=['/Library/Developer/CommandLineTools/usr/bin/c++','-std=gnu++17','-O3','-flto',
             '-arch','arm64','-isysroot','/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk',
             '-I'+str(native/'include'),'-I'+str(native/'json/single_include/nlohmann'),
             str(source),*[str(p) for p in objects],'-o',str(executable)]
    subprocess.run(command,check=True,timeout=120)
    process=subprocess.run([str(executable),str(args.worlds),str(args.samples),str(args.simulations)],
                           capture_output=True,text=True,timeout=180)
    (out/'stdout.jsonl').write_text(process.stdout);(out/'stderr.log').write_text(process.stderr)
    if process.returncode:raise RuntimeError(f'Probe exited {process.returncode}; inspect retained logs')
    rows=[json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    if len(rows)!=args.worlds:raise RuntimeError('Incomplete bounded probe')
    report={'schema':'sts_hidden_pile_invariance_probe_v1','run_record':record,
            'executable':file_identity(executable),'worlds':len(rows),'vote_histograms_changed':sum(r['votes_differ'] for r in rows),
            'pluralities_changed':sum(r['plurality_differ'] for r in rows),
            'canonical_histograms_all_equal':all(r['canonical_votes_equal'] for r in rows),'records':rows,
            'scope':'Constructed Ironclad/Champ starts. Copies differ only by reversal of non-public pile order; hand/enemies and all other fields copied unchanged. Standalone algorithm mirrors reseeding/shuffling/root selection; plurality tie-break differs from Python and is secondary. Not a performance evaluation or proof of systematic clairvoyance. Existing native source and builds untouched.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['worlds','vote_histograms_changed','pluralities_changed','canonical_histograms_all_equal']}))


if __name__=='__main__':main()
