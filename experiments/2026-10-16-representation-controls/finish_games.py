#!/usr/bin/env python3
"""Finish the representation-controls full-game arms after the 19 October handoff.

Each shard command is the study's own game command (as last launched by its
controller): same script, checkpoints, worlds, worker count and timeout. Only
the interrupted old1919 arm gets a new output folder; its stopped attempt is
kept and excluded. Checkpoints come from the study's global_freeze_v1.json.

    finish_games.py plan          print the shard and merge commands
    finish_games.py enqueue       add the shards to the study's queue and start it
    finish_games.py retry         queue a fresh run of each shard in shard_retries, in a new folder
    finish_games.py merge-watch   queue each arm's merge once its 8 shards are complete

A shard listed in shard_retries (for example after a battery pause made one
world time out) is re-run whole into the named folder, and merges use that
folder. Arms listed in no_merge are analysed from their shards instead.

merge-watch stops when every arm's merge is queued, or when STOP exists in the
run's finish_games_20261019/ folder.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

CONFIG = json.loads((Path(__file__).with_name('finish_games.json')).read_text())
ROOT = Path(CONFIG['root'])
BASE = ROOT / CONFIG['base']
CONTROL = BASE / 'finish_games_20261019'
PY = ROOT / '.venv/bin/python'
QUEUE = [str(PY), '-B', str(ROOT / 'scripts/rl/queue_runner.py')]


def prefix():
    return ['/usr/bin/env', f'STS_LIGHTSPEED_BUILD={CONFIG["native_build"]}', 'PYTHONDONTWRITEBYTECODE=1',
            'OPENBLAS_NUM_THREADS=1', 'VECLIB_MAXIMUM_THREADS=1', str(PY)]


def checkpoints():
    freeze = json.loads((BASE / 'global_freeze_v1.json').read_text())
    return {arm: freeze['conditions'][sel]['checkpoint']['path']
            for arm, sel in freeze['game_arms'].items() if sel}


def arm_dir(arm):
    return BASE / 'games' / CONFIG['out_dir'].get(arm, arm)


def shard_dir(arm, i):
    return arm_dir(arm) / CONFIG.get('shard_retries', {}).get(f'{arm}/{i:02d}', f'shard_{i:02d}')


def shard_command(arm, i, checkpoint, folder):
    return prefix() + [str(ROOT / 'scripts/rl/controlled_eval.py'), '--mode', 'eval',
        '--checkpoint', CONFIG['champion'], '--no-teacher-labels', '--no-bank', '--self-consistency-every', '0',
        '--world-start', str(CONFIG['world_start'] + CONFIG['shard_worlds'] * i), '--n', str(CONFIG['shard_worlds']),
        '--workers', '1', '--timeout', '30', '--out', str(folder), '--combat', checkpoint]


def merge_command(arm):
    shards = [str(shard_dir(arm, i)) for i in range(CONFIG['shards'])]
    return prefix() + [str(ROOT / 'scripts/rl/fresh_merge.py'), '--inputs', *shards,
        '--world-start', str(CONFIG['world_start']), '--n', str(CONFIG['shard_worlds'] * CONFIG['shards']),
        '--out', str(arm_dir(arm) / 'merged')]


def shard_jobs():
    """Arms in priority order, shards spread round-robin over the lanes."""
    cps, jobs = checkpoints(), []
    for arm in CONFIG['arm_order']:
        for i in range(CONFIG['shards']):
            k = len(jobs)
            jobs.append((f'{CONFIG["queue_prefix"]}_game_{arm}_{i:02d}', f'game{k % CONFIG["lanes"]}',
                         shard_command(arm, i, cps[arm], arm_dir(arm) / f'shard_{i:02d}')))
    return jobs


def queued_names():
    runs = json.loads((ROOT / 'data/rl_runs/queue.json').read_text())['runs']
    return {r['name'] for r in runs}


def add(name, lane, command):
    subprocess.run(QUEUE + ['add', name, '--lane', lane, '--'] + command, check=True)


def shard_complete(arm, i):
    path = shard_dir(arm, i) / 'completion.json'
    if not path.exists():
        return None
    done = json.loads(path.read_text())
    return done.get('status') == 'completed' and done.get('cohort_complete') is True


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'plan'
    if mode == 'plan':
        for name, lane, cmd in shard_jobs():
            print(name, lane, ' '.join(cmd))
        for arm in CONFIG['arm_order']:
            print(f'{CONFIG["queue_prefix"]}_merge_{arm}', 'merge', ' '.join(merge_command(arm)))
    elif mode == 'enqueue':
        for arm in CONFIG['out_dir']:
            if arm_dir(arm).exists():
                sys.exit(f'{arm_dir(arm)} already exists; refusing to reuse an output folder')
        CONTROL.mkdir(exist_ok=True)
        jobs = shard_jobs()
        (CONTROL / 'plan.json').write_text(json.dumps(
            {'config': CONFIG, 'checkpoints': checkpoints(), 'shards': jobs,
             'merges': {arm: merge_command(arm) for arm in CONFIG['arm_order']}}, indent=1))
        existing = queued_names()
        for name, lane, cmd in jobs:
            if name not in existing:
                add(name, lane, cmd)
        subprocess.run(QUEUE + ['start'], check=True)
    elif mode == 'retry':
        cps, lanes, existing = checkpoints(), {name: lane for name, lane, _ in shard_jobs()}, queued_names()
        for key, folder in CONFIG['shard_retries'].items():
            arm, i = key.split('/')[0], int(key.split('/')[1])
            original = f'{CONFIG["queue_prefix"]}_game_{arm}_{i:02d}'
            name = f'{original}_{folder.rsplit("_", 1)[-1]}'
            if name in existing:
                continue
            if shard_dir(arm, i).exists():
                sys.exit(f'{shard_dir(arm, i)} already exists; refusing to reuse an output folder')
            add(name, lanes[original], shard_command(arm, i, cps[arm], shard_dir(arm, i)))
    elif mode == 'merge-watch':
        queued, warned = set(), set()
        while True:
            if (CONTROL / 'STOP').exists():
                print('STOP present; exiting', flush=True)
                return
            names = queued_names()
            for arm in CONFIG['arm_order']:
                name = f'{CONFIG["queue_prefix"]}_merge_{arm}'
                if arm in CONFIG.get('no_merge', {}):
                    queued.add(name)
                    continue
                if name in queued or name in names:
                    queued.add(name)
                    continue
                states = [shard_complete(arm, i) for i in range(CONFIG['shards'])]
                if all(s is True for s in states):
                    add(name, 'merge', merge_command(arm))
                    queued.add(name)
                    print(time.strftime('%H:%M:%S'), 'queued', name, flush=True)
                elif any(s is False for s in states) and arm not in warned:
                    warned.add(arm)
                    print(time.strftime('%H:%M:%S'), arm, 'has an incomplete shard; not merging', flush=True)
            if len(queued) == len(CONFIG['arm_order']):
                print('all merges queued', flush=True)
                return
            time.sleep(60)
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
