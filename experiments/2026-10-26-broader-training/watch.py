"""Monitor this continuation; request a stop for sustained deterioration or a stall.

All events live in the output directory. No hyperparameters change automatically.
"""
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


def records(path):
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text().splitlines(keepends=True)
            if s.endswith('\n') and s.strip()]


def main():
    root, pid = Path(sys.argv[1]), int(sys.argv[2])
    arm = root / 'lr_3e-5'
    previous, last_progress, last_power = None, time.monotonic(), None
    alerts = set()

    def alert(reason, stop=False):
        if reason in alerts:
            return
        alerts.add(reason)
        event = {'event': 'alert', 'reason': reason, 'stop_requested': stop, 'time': time.time()}
        print(json.dumps(event), flush=True)
        with (root / 'alerts.jsonl').open('a') as out:
            out.write(json.dumps(event) + '\n')
        if stop:
            (arm / 'STOP').touch()

    while True:
        now = time.monotonic()
        iterations = records(arm / 'iterations.jsonl')
        evaluations = records(arm / 'evaluations.jsonl')
        progress_file = arm / 'evaluation_progress.json'
        evaluation_progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
        progress = (len(iterations), len(evaluations),
                    evaluation_progress.get('completed'), evaluation_progress.get('generated_tokens'))
        power = root / 'babysitter.log'
        power_stamp = power.stat().st_mtime_ns if power.exists() else None
        if power_stamp != last_power:
            last_progress, last_power = now, power_stamp
        if progress != previous:
            last_progress, previous = now, progress
            print(json.dumps({'event': 'progress', 'iterations': progress[0], 'evaluations': progress[1],
                              'last': iterations[-1] if iterations else None}), flush=True)
        if (root / 'STOP').exists() and arm.exists():
            alert('user_stop_file', stop=True)
        if len(iterations) >= 16:
            validity = [statistics.fmean(r['all']['valid_fraction'] for r in block)
                        for block in (iterations[-16:-8], iterations[-8:])]
            if max(validity) < .90:
                alert('validity_below_90_percent_in_two_256_response_windows', stop=True)
            recent = iterations[-16:]
            a_share = [r['all']['first_displayed_option_rate'] for r in recent]
            if all(v is not None for v in a_share) and statistics.fmean(a_share) > .70:
                alert('sustained_A_answer_share_above_70_percent')
        if len(evaluations) >= 3:
            base = evaluations[0]['probe/all']
            bad = [e['probe/all']['teacher_top_agreement'] <= base['teacher_top_agreement'] - .10
                   and e['probe/all']['mean_teacher_probability'] < base['mean_teacher_probability']
                   for e in evaluations[-2:]]
            if all(bad):
                alert('held_out_agreement_down_10_points_and_score_lower_at_two_evaluations', stop=True)
        if now - last_progress > 1200:
            alert('no_progress_for_20_minutes', stop=True)
        if (arm / 'run_summary.json').exists():
            print(json.dumps({'event': 'finished', **json.loads((arm / 'run_summary.json').read_text())}), flush=True)
            return
        alive = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='], capture_output=True, text=True)
        if alive.returncode or not alive.stdout.strip() or alive.stdout.strip().startswith('Z'):
            alert('trainer_exited_without_run_summary')
            return
        if alive.stdout.strip().startswith('T'):
            last_progress = now
        time.sleep(15)


if __name__ == '__main__':
    main()
