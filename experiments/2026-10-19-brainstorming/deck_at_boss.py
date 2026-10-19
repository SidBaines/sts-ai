"""Replay recorded games and dump the public state when the Act 2 boss fight starts.

Usage: STS_LIGHTSPEED_BUILD=<build> python deck_at_boss.py <games/arm folder> <worlds> <out.json>
Run with the frozen study runtime; it reads the per-world shard records and replays each
action prefix through the same worker the evaluations used."""
import gzip, json, sys, glob, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from sts_policy.curriculum import TaskSpec
from sts_rl.common import make_workers

def main():
    arm_dir, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    files = sorted(glob.glob(arm_dir + '/shard_*/worlds/*.json.gz'))
    worker = make_workers(1, dead_action_filter=True, slim=False)[0]
    results = []
    t0 = time.time()
    try:
        for f in files:
            if len(results) >= n: break
            rec = json.load(gzip.open(f))
            if rec['status'] != 'completed' or not rec['bosses'].get('2', {}).get('reach'): continue
            task = dict(rec['task']); task['prefix'] = []; task['ooc_previews'] = []; task.pop('expected_public_hash', None)
            task.pop('task_version', None); task.pop('split', None)
            spec = TaskSpec.from_dict(task)
            obs, _ = worker.submit('reset', spec).result()
            snap = None; term = cut = False; k = 0; prev_phase = obs['phase']
            for a in rec['action_prefix']:
                st = obs['state']
                if snap is None and obs['phase'] == 'combat' and st['act'] == 2 and str(st.get('room')).endswith('BOSS'):
                    snap = {'seed': rec['world_seed'], 'step': k, 'win': rec['win'], 'floor_reached': rec['floor'],
                            'pass2': rec['bosses']['2']['pass'], 'state': st}
                    break
                obs, _, term, cut, _ = worker.submit('step', a).result()
                k += 1
                if term or cut: break
            if snap is None:
                results.append({'seed': rec['world_seed'], 'missing': True, 'final_floor': obs['state']['floor'], 'recorded_floor': rec['floor']})
            else:
                results.append(snap)
            if len(results) % 10 == 0: print(len(results), 'done', round(time.time() - t0), 's', flush=True)
    finally:
        worker.close()
    Path(out).write_text(json.dumps(results))
    print('wrote', out, len(results))


if __name__ == '__main__':
    main()
