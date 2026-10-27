"""Compare every saved evaluation with its starting adapter; no model work.

Run from the worktree root with PYTHONPATH=src .venv-mlx/bin/python.
Paired statistics come from the tested library; this file selects this study's
panels, exports their results, and draws its figure.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics

from sts_llm.statistics import paired_cluster_bootstrap


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('data/llm_runs/broader_training_20261026'))
    parser.add_argument('--figures', type=Path, default=Path('research/figures/broader-training'))
    args = parser.parse_args()
    arm = args.run / 'lr_3e-5'
    config = json.loads((arm / 'config.json').read_text())
    summary = json.loads((arm / 'run_summary.json').read_text())
    settings = config['settings']
    states = {row['id']: row for name in ('train', 'probe')
              for row in read_rows(Path(settings[name]))}
    evaluations = read_rows(arm / 'evaluations.jsonl')
    raw = {row['iteration']: read_rows(arm / 'evaluations' / f"iter_{row['iteration']:04d}.jsonl")
           for row in evaluations}
    baseline = {(row['set'], row['id']): row for row in raw[0]}
    comparisons, sensitivities = [], []

    def compare(before, after, metric):
        def pack(rows):
            return [{'prompt_id': row['id'], 'world_id': states[row['id']]['world_identity'],
                     'value': float(row[metric])} for row in rows]
        result = paired_cluster_bootstrap(pack(before), pack(after))
        result.pop('pairs')
        return result

    for iteration, rows in sorted(raw.items()):
        indexed = {(row['set'], row['id']): row for row in rows}
        assert len(indexed) == len(rows) == len(baseline)
        assert indexed.keys() == baseline.keys()
        for key, row in indexed.items():
            assert row['order'] == baseline[key]['order'], key
            assert row['phase'] == baseline[key]['phase'], key
        if iteration == 0:
            continue
        for panel in ('train', 'probe'):
            for phase in ('all', 'combat', 'out_of_combat'):
                keys = sorted(key for key, row in indexed.items()
                              if row['set'] == panel and (phase == 'all' or row['phase'] == phase))
                before, after = [baseline[k] for k in keys], [indexed[k] for k in keys]
                for metric in ('teacher_top_agreement', 'teacher_probability', 'valid', 'capped'):
                    comparisons.append({'iteration': iteration, 'panel': panel, 'phase': phase,
                                        'metric': metric, **compare(before, after, metric)})
            keys = sorted(key for key, row in indexed.items() if row['set'] == panel
                          and row['valid'] and baseline[key]['valid'])
            for metric in ('teacher_top_agreement', 'teacher_probability'):
                sensitivities.append({'iteration': iteration, 'panel': panel, 'metric': metric,
                                      **compare([baseline[k] for k in keys], [indexed[k] for k in keys], metric)})

    iterations = read_rows(arm / 'iterations.jsonl')
    generations = [row for step in iterations
                   for row in read_rows(arm / 'generations' / f"iter_{step['iteration']:04d}.jsonl")]
    final_iteration = summary['last_iteration']
    power_lines = (args.run / 'babysitter.log').read_text().splitlines()
    # The launcher writes local timestamps; this analysis runs on that same Mac.
    paused_at, pauses = None, []
    for line in power_lines:
        stamp = datetime.strptime(line[:19], '%Y-%m-%d %H:%M:%S').timestamp()
        if 'battery: paused' in line:
            paused_at = stamp
        elif 'mains power: resumed' in line and paused_at is not None:
            pauses.append((paused_at, stamp))
            paused_at = None
    end = (arm / 'run_summary.json').stat().st_mtime
    start = (arm / 'config.json').stat().st_mtime
    eval_start = (arm / 'adapters' / f'iter_{final_iteration:04d}.safetensors').stat().st_mtime
    def active_seconds(begin):
        return end - begin - sum(max(0, min(end, hi) - max(begin, lo)) for lo, hi in pauses)
    output = {'source_commit': config['source_commit'], 'run_summary': summary,
              'evaluation_summaries': evaluations, 'comparisons': comparisons,
              'valid_pairs_only': sensitivities,
              'training': {'iterations': len(iterations), 'responses': len(generations),
                           'valid': sum(r['valid'] for r in generations),
                           'capped': sum(r['capped'] for r in generations),
                           'distinct_states': len({r['id'] for r in generations})},
              'timing': {'training_iteration_seconds': sum(r['iteration_seconds'] for r in iterations),
                         'wall_seconds_excluding_logged_power_pauses': active_seconds(start),
                         'final_eval_wall_seconds_excluding_logged_power_pauses': active_seconds(eval_start),
                         'final_eval_tokens': sum(r['output_tokens'] for r in raw[final_iteration]),
                         'note': 'Active wall time is approximate, inferred from file timestamps and power logs. Raw monotonic timers include awake battery pauses and do not equal GPU working time.'}}
    (args.run / 'analysis.json').write_text(json.dumps(output, indent=2) + '\n')
    args.figures.mkdir(parents=True, exist_ok=True)
    flat = [{k: row[k] for k in ('iteration', 'panel', 'phase', 'metric', 'n_pairs', 'n_worlds',
                                'left_mean', 'right_mean', 'difference')} |
            {'ci_low': row['ci95'][0], 'ci_high': row['ci95'][1]} for row in comparisons]
    with (args.figures / 'comparisons.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    series = [{'iteration': r['iteration'], 'valid_fraction': r['all']['valid_fraction'],
               'capped_fraction': r['all']['capped_fraction'],
               'median_reasoning_tokens': r['all']['median_reasoning_tokens']} for r in iterations]
    with (args.figures / 'training.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(series[0])); writer.writeheader(); writer.writerows(series)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), layout='constrained')
    steps = [r['iteration'] for r in series]
    axes[0].plot(steps, [100*r['valid_fraction'] for r in series], color='#176c80', label='Valid')
    axes[0].plot(steps, [100*r['capped_fraction'] for r in series], color='#b84236', label='Capped')
    axes[0].set(title='Sampled answer health', xlabel='Update', ylabel='Answers (%)', ylim=(0, 105)); axes[0].legend(frameon=False)
    axes[1].plot(steps, [r['median_reasoning_tokens'] for r in series], color='#82542c')
    axes[1].set(title='Sampled reasoning length', xlabel='Update', ylabel='Median tokens', ylim=(0, 4300))
    base, final = evaluations[0]['probe/all'], evaluations[-1]['probe/all']
    axes[2].bar([-.18, .82], [100*base['teacher_top_agreement'], 100*base['valid_fraction']], width=.36, color='#176c80', label='Start')
    axes[2].bar([.18, 1.18], [100*final['teacher_top_agreement'], 100*final['valid_fraction']], width=.36, color='#b84236', label='Final')
    axes[2].set(title='Same 96 held-out states', ylabel='Answers (%)', ylim=(0, 105), xticks=[0, 1], xticklabels=['Teacher match', 'Valid']); axes[2].legend(frameon=False)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(args.figures / 'outcome.png', dpi=140)
    plt.close(fig)
    for row in comparisons:
        if row['metric'] in ('teacher_top_agreement', 'teacher_probability'):
            print(json.dumps({k: row[k] for k in ('panel', 'phase', 'metric', 'n_pairs', 'n_worlds', 'left_mean', 'right_mean', 'difference', 'ci95')}))
    print('timing', json.dumps(output['timing']))
    print('valid-pairs-only', json.dumps(sensitivities))


if __name__ == '__main__':
    main()
