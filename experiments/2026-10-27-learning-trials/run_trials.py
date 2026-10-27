"""Four timed learning trials, then continue the best usable result within budget.

This is a small study-specific launch loop. STOP files remain the control path;
all settings, progress, stdout and results are retained in the run directory.
"""
import json
from datetime import datetime
from pathlib import Path
import statistics
import subprocess
import sys
import time


def rows(path):
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text().splitlines(keepends=True)
            if s.endswith('\n') and s.strip()]


def main():
    repo = Path.cwd()
    plan = json.loads(Path(__file__).with_name('config.json').read_text())
    root = repo / 'data/llm_runs' / sys.argv[1]
    root.mkdir(parents=True, exist_ok=True)
    deadline = datetime.fromisoformat(plan['deadline']).timestamp()
    (root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    parent = repo / plan['parent_adapter']
    outcomes = []

    def launch(arm, seconds, adapter=parent, *, full=False, evaluate_only=False, seed=1729):
        folder = root / ('lr_' + arm['key'])
        command = [sys.executable, '-u', '-m', 'sts_llm.mlx_training',
                   '--mlx-model', str(repo / 'data/models/k2-horizon-0.9b-mlx'),
                   '--adapter', str(adapter), '--reference-adapter', str(parent),
                   '--label', arm['label'], '--train', 'data/llm_runs/broader_states_selected_20261026/train.jsonl',
                   '--train-eval', 'data/llm_runs/broader_states_selected_20261026/train_eval.jsonl' if full else 'data/llm_runs/learning_panels_20261027/train_eval.jsonl',
                   '--probe', 'data/llm_runs/broader_states_selected_20261026/probe.jsonl' if full else 'data/llm_runs/learning_panels_20261027/probe.jsonl',
                   '--output', str(folder), '--iterations', '1000', '--max-run-seconds', str(max(1, seconds)),
                   '--states-per-iteration', '4', '--group-size', '8', '--effort', arm['effort'],
                   '--temperature', '.6', '--max-new-tokens', '1024', '--learning-rate', '1e-5',
                   '--kl-reference', 'initial', '--kl-beta', str(arm['kl_beta']),
                   '--policy-token-scope', arm['policy_token_scope'],
                   '--length-penalty-per-256', str(arm.get('length_penalty_per_256', 0.0)),
                   '--free-reasoning-tokens', '256',
                   '--loss-normalization', 'fixed_token_budget', '--loss-token-budget', '2048',
                   '--invalid-reward', '-1', '--eval-every', str(arm['eval_every']),
                   '--eval-orders', '1', '--eval-concurrency', '16', '--save-every', '8',
                   '--seed', str(seed), '--eval-seed', '1729']
        if evaluate_only:
            command.append('--eval-only')
        print(json.dumps({'event': 'start', 'arm': arm['key'], 'seconds': seconds, 'command': command}), flush=True)
        with (root / ('lr_' + arm['key'] + '.log')).open('x') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        state = {'current_arm': arm['key'], 'trainer_pid': process.pid, 'deadline': plan['deadline'],
                 'started_at': time.time(), 'outcomes': outcomes}
        (root / 'session.json').write_text(json.dumps(state, indent=2) + '\n')
        last_progress, last_size, stop_reason = time.time(), None, None
        while process.poll() is None:
            files = [folder / n for n in ('iterations.jsonl', 'evaluations.jsonl', 'evaluation_progress.json')]
            signature = [(p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None for p in files]
            if signature != last_size:
                last_progress, last_size = time.time(), signature
            steps = rows(folder / 'iterations.jsonl')
            if len(steps) >= 2 and not evaluate_only:
                recent = steps[-2:]
                if statistics.fmean(s['all']['valid_fraction'] for s in recent) < .8:
                    stop_reason = stop_reason or 'recent_validity_below_80_percent'
            if (root / 'STOP').exists() or time.time() >= deadline - 60:
                stop_reason = stop_reason or 'session_stop_or_deadline'
            if time.time() - last_progress > 900:
                stop_reason = stop_reason or 'no_progress_for_15_minutes'
            if stop_reason and folder.exists():
                (folder / 'STOP').touch()
            if time.time() - last_progress > 1020:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            time.sleep(5)
        evaluations = rows(folder / 'evaluations.jsonl')
        checkpoints = sorted((folder / 'adapters').glob('*.safetensors'))
        result = {'arm': arm, 'exit_code': process.wait(), 'stop_reason': stop_reason,
                  'iterations': len(rows(folder / 'iterations.jsonl')),
                  'evaluations': evaluations, 'checkpoint': str(checkpoints[-1]) if checkpoints else None}
        outcomes.append(result)
        (root / 'outcomes.json').write_text(json.dumps(outcomes, indent=2) + '\n')
        print(json.dumps({'event': 'end', 'arm': arm['key'], 'exit': result['exit_code'],
                          'iterations': result['iterations'], 'stop_reason': stop_reason,
                          'probe': evaluations[-1].get('probe/all') if evaluations else None}), flush=True)
        return result

    for arm in plan['arms']:
        if (root / 'STOP').exists() or deadline - time.time() < plan['final_reserve_seconds'] + 600:
            break
        launch(arm, min(plan['screen_seconds'], deadline - time.time() - plan['final_reserve_seconds']))

    usable = [r for r in outcomes if r['exit_code'] == 0 and r['checkpoint'] and len(r['evaluations']) >= 2
              and r['evaluations'][-1]['probe/all']['valid_fraction'] >= .9]
    if usable and not (root / 'STOP').exists():
        best = max(usable, key=lambda r: r['evaluations'][-1]['probe/all']['mean_teacher_probability'])
        (root / 'selection.json').write_text(json.dumps({'selected': best['arm']['key'],
            'reason': 'highest final probe teacher probability among runs with at least 90% valid probe answers; exploratory'}, indent=2) + '\n')
        remaining = deadline - time.time() - plan['final_reserve_seconds']
        if remaining > 600:
            extension = dict(best['arm'], key='continue_' + best['arm']['key'], label='Continue: ' + best['arm']['label'])
            extended = launch(extension, min(5400, remaining), Path(best['checkpoint']), seed=2718)
            if extended['exit_code'] == 0 and extended['checkpoint'] and extended['evaluations'][-1]['probe/all']['valid_fraction'] >= .9:
                if extended['evaluations'][-1]['probe/all']['mean_teacher_probability'] >= best['evaluations'][-1]['probe/all']['mean_teacher_probability']:
                    best = extended
        if deadline - time.time() > 300 and not (root / 'STOP').exists():
            launch(dict(best['arm'], key='full_parent', label='Full panel: starting adapter'), 1, parent, full=True, evaluate_only=True)
            launch(dict(best['arm'], key='full_candidate', label='Full panel: selected candidate'), 1, Path(best['checkpoint']), full=True, evaluate_only=True)
    (root / 'session.json').write_text(json.dumps({'finished': True, 'ended_at': time.time(), 'outcomes': outcomes}, indent=2) + '\n')
    print('SCREEN_DONE', flush=True)


if __name__ == '__main__':
    main()
