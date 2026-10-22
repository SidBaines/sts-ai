#!/usr/bin/env python3
"""Joint loop orchestrator: teacher labelling and gated combat updates beside a PPO run.

    joint_loop.py CONFIG.json

The out-of-combat PPO trainer runs separately (`train_ooc.py --combat-follow FOLLOW`).
Each cycle of this loop:
  1. takes the trainer's newest saved checkpoint as the player (the start checkpoint
     before the first save) and labels `worlds_per_cycle` new training worlds with the
     search teacher (`fresh_collect.py`, eight shards on the queue's teacher lanes);
  2. fits one combat student per seed on every corpus so far (`fresh_train.py
     --mixed-sources`, the 15 October recipe);
  3. gates each student: the player with the student spliced in (`combine_heads.py`)
     and the player as it is both play the gate worlds (`fast_eval.py`); the best
     candidate is adopted only if it wins at least as many games and its mean floor
     is at least as high, by atomically replacing FOLLOW, which the trainer picks up
     at its next iteration.
Every decision is appended to OUT/decisions.jsonl. A file OUT/STOP ends the loop
between steps. Heavy jobs go through the queue, so they pause on battery.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class Loop:
    def __init__(self, cfg):
        self.cfg = cfg
        self.out = Path(cfg['out'])
        self.out.mkdir(parents=True, exist_ok=True)
        self.code = Path(cfg['code'])
        self.py = cfg['python']
        self.build = cfg['build']

    def log(self, **event):
        event['time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with (self.out / 'decisions.jsonl').open('a') as f:
            f.write(json.dumps(event) + '\n')
        print(json.dumps(event), flush=True)

    def stopped(self):
        return (self.out / 'STOP').exists()

    def queue(self, name, lane, cmd):
        runner = str(Path(self.cfg['queue_root']) / 'scripts/rl/queue_runner.py')
        subprocess.run([self.py, runner, 'add', name, '--lane', lane, '--',
                        'env', f'STS_LIGHTSPEED_BUILD={self.build}', *cmd], check=True, capture_output=True)
        subprocess.run([self.py, runner, 'start'], check=True, capture_output=True)

    def job_failed(self, name):
        q = json.loads((Path(self.cfg['queue_root']) / 'data/rl_runs/queue.json').read_text())
        return any(r['name'] == name and r['status'] in ('failed', 'exited', 'cancelled') for r in q['runs'])

    def wait(self, paths, jobs):
        while not all(Path(p).exists() for p in paths):
            failed = [j for j in jobs if self.job_failed(j)]
            if failed:
                raise RuntimeError(f'queue jobs failed: {failed}')
            time.sleep(60)
        time.sleep(5)

    def player(self):
        saved = sorted(Path(self.cfg['trainer_out']).glob('iter_*.pt'))
        return saved[-1] if saved else Path(self.cfg['start'])

    def label(self, cycle, player):
        corpus = self.out / f'corpus_{cycle:02d}'
        n, shards = self.cfg['worlds_per_cycle'], 8
        start = self.cfg['label_world_start'] + cycle * self.cfg['label_world_stride']
        copy = self.out / 'players' / f'player_{cycle:02d}.pt'
        copy.parent.mkdir(exist_ok=True)
        shutil.copy2(player, copy)
        jobs, done = [], []
        for i in range(shards):
            name = f"{self.cfg['name']}-c{cycle:02d}-label{i}"
            self.queue(name, f"{self.cfg['lane_prefix']}label{i}", [self.py, str(self.code / 'scripts/rl/fresh_collect.py'), '--mode', 'collect',
                       '--checkpoint', str(copy), '--out', str(corpus / f'shard_{i:02d}'),
                       '--world-start', str(start + i * (n // shards)), '--n', str(n // shards), '--workers', '1',
                       '--timeout', '30', '--teacher-probabilities', json.dumps(self.cfg['teacher_probabilities']),
                       '--samples', '24', '--simulations', '1000', '--self-consistency-every', '10'])
            jobs.append(name)
            done.append(corpus / f'shard_{i:02d}' / 'completion.json')
        self.log(event='labelling', cycle=cycle, player=str(player), player_sha256=sha256(copy), corpus=str(corpus),
                 world_start=start, worlds=n)
        self.wait(done, jobs)
        return corpus

    def fit(self, cycle, corpora, seed):
        out = self.out / f'fit_c{cycle:02d}_seed{seed}'
        data = [str(c / f'shard_{i:02d}') for c in corpora if (c / 'shard_00').exists() for i in range(8)]
        data += self.cfg['base_corpus_shards']
        name = f"{self.cfg['name']}-c{cycle:02d}-fit{seed}"
        deadline = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() + 20 * 3600))
        self.queue(name, f"{self.cfg['lane_prefix']}fit", [self.py, str(self.code / 'scripts/rl/fresh_train.py'), '--mixed-sources', '--data', *data,
                   '--probe', self.cfg['probe'], '--out', str(out), '--architecture', 'old', '--phase', 'combat',
                   '--lr', '0.001', '--old-size', 'large', '--old-context', 'none', '--steps', '12000',
                   '--min-steps', '2000', '--batch', '128', '--defer-test', '--eval-every', '100', '--patience', '6',
                   '--seed', str(seed), '--device', 'mps', '--deadline-utc', deadline])
        while not ((out / 'best.pt').exists() and '"status": "completed"' in (out / 'status.json').read_text()
                   if (out / 'status.json').exists() else False):
            if self.job_failed(name):
                raise RuntimeError(f'fit failed: {name}')
            time.sleep(60)
        return out / 'best.pt'

    def evaluate(self, name, checkpoint, out):
        self.queue(name, f"{self.cfg['lane_prefix']}eval", [self.py, str(self.code / 'scripts/rl/fast_eval.py'), '--checkpoint', str(checkpoint),
                   '--world-start', str(self.cfg['gate_world_start']), '--n', str(self.cfg['gate_worlds']),
                   '--processes', '8', '--games-per-process', '32', '--out', str(out)])
        return out / 'summary.json'

    def compare(self, base, arm):
        out = Path(str(arm) + '.compare.json')
        subprocess.run([self.py, str(self.code / 'experiments/2026-10-20-boss-rares/compare_arms.py'), str(base), str(arm),
                        '--out', str(out)], check=True, env={'PYTHONPATH': str(self.code / 'src')}, capture_output=True)
        return next(iter(json.loads(out.read_text()).values()))

    def gate(self, cycle, students):
        player = self.player()
        pc = self.out / 'gate' / f'c{cycle:02d}'
        pc.mkdir(parents=True, exist_ok=True)
        incumbent = pc / 'incumbent.pt'
        shutil.copy2(player, incumbent)
        jobs = [f"{self.cfg['name']}-c{cycle:02d}-gate-incumbent"]
        summaries = {'incumbent': self.evaluate(jobs[0], incumbent, pc / 'incumbent')}
        for seed, student in students.items():
            candidate = pc / f'candidate_seed{seed}.pt'
            subprocess.run([self.py, str(self.code / 'scripts/rl/combine_heads.py'), '--ooc', str(incumbent),
                            '--combat', str(student), '--output', str(candidate)], check=True,
                           env={'STS_LIGHTSPEED_BUILD': self.build, 'PATH': '/usr/bin:/bin'}, capture_output=True)
            job = f"{self.cfg['name']}-c{cycle:02d}-gate-seed{seed}"
            jobs.append(job)
            summaries[seed] = self.evaluate(job, candidate, pc / f'seed{seed}')
        self.wait(list(summaries.values()), jobs)
        results = {seed: self.compare(summaries['incumbent'], summaries[seed]) for seed in students}
        best = max(results, key=lambda s: (results[s]['wins'][1], results[s]['floor']['difference']))
        r = results[best]
        accept = r['wins'][1] >= r['wins'][0] and r['floor']['difference'] >= 0
        self.log(event='gate', cycle=cycle, incumbent=str(player), incumbent_sha256=sha256(incumbent),
                 results={str(s): {'wins': v['wins'], 'floor_difference': v['floor']['difference'],
                                   'mcnemar_p': v['mcnemar_p']} for s, v in results.items()},
                 best_seed=best, accepted=accept)
        if accept:
            follow = Path(self.cfg['follow'])
            temp = follow.with_suffix('.tmp')
            shutil.copy2(students[best], temp)
            temp.replace(follow)
            self.log(event='adopted', cycle=cycle, student=str(students[best]), sha256=sha256(follow))
        return accept

    def run(self):
        corpora = sorted(self.out.glob('corpus_*'))
        cycle = len(corpora)
        while cycle < self.cfg['max_cycles'] and not self.stopped():
            corpora.append(self.label(cycle, self.player()))
            if self.stopped():
                break
            students = {seed: self.fit(cycle, corpora, seed) for seed in self.cfg['seeds']}
            if self.stopped():
                break
            self.gate(cycle, students)
            cycle += 1
        self.log(event='stopped', cycles=cycle)


if __name__ == '__main__':
    Loop(json.loads(Path(sys.argv[1]).read_text())).run()
