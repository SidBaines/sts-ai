"""Compare saved trial evaluations; this does not load a model or change a run."""
import argparse
import json
from pathlib import Path

from sts_llm.statistics import paired_cluster_bootstrap


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True)
            if line.endswith('\n') and line.strip()]


def paired(before, after, states, panel, metric):
    left = [r for r in before if r['set'] == panel]
    right = [r for r in after if r['set'] == panel]
    assert {(r['id'], tuple(r['order'])) for r in left} == {(r['id'], tuple(r['order'])) for r in right}
    def pack(rows):
        return [{'prompt_id': r['id'], 'world_id': states[r['id']]['world_identity'],
                 'value': float(r[metric])} for r in rows]
    result = paired_cluster_bootstrap(pack(left), pack(right))
    result.pop('pairs')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('data/llm_runs/learning_trials_20261027'))
    args = parser.parse_args()
    comparisons, final_rows, summaries, states = [], {}, {}, {}
    original_comparisons = []
    original = read_rows(args.run / 'lr_small_step/evaluations/iter_0000.jsonl')
    for arm in sorted(args.run.glob('lr_*')):
        if not arm.is_dir() or not (arm / 'config.json').exists():
            continue
        settings = json.loads((arm / 'config.json').read_text())['settings']
        for name in ('train', 'probe'):
            states.update({r['id']: r for r in read_rows(Path(settings[name]))})
        evaluations = read_rows(arm / 'evaluations.jsonl')
        summaries[arm.name] = evaluations
        if not evaluations:
            continue
        baseline = read_rows(arm / 'evaluations/iter_0000.jsonl')
        for evaluation in evaluations[1:]:
            after = read_rows(arm / 'evaluations' / f"iter_{evaluation['iteration']:04d}.jsonl")
            for panel in ('train', 'probe'):
                for metric in ('teacher_top_agreement', 'teacher_probability'):
                    comparisons.append({'arm': arm.name, 'iteration': evaluation['iteration'],
                                        'panel': panel, 'metric': metric,
                                        **paired(baseline, after, states, panel, metric)})
        if arm.name.startswith('lr_continue_') and original:
            for evaluation in evaluations:
                after = read_rows(arm / 'evaluations' / f"iter_{evaluation['iteration']:04d}.jsonl")
                for panel in ('train', 'probe'):
                    for metric in ('teacher_top_agreement', 'teacher_probability'):
                        original_comparisons.append({'arm': arm.name, 'iteration': evaluation['iteration'],
                                                     'panel': panel, 'metric': metric,
                                                     **paired(original, after, states, panel, metric)})
        if (arm / 'run_summary.json').exists() and len(evaluations) > 1:
            final_rows[arm.name] = read_rows(arm / 'evaluations' / f"iter_{evaluations[-1]['iteration']:04d}.jsonl")
    contrasts = []
    if 'lr_small_step' in final_rows:
        for name, rows in final_rows.items():
            if name == 'lr_small_step':
                continue
            for panel in ('train', 'probe'):
                for metric in ('teacher_top_agreement', 'teacher_probability'):
                    contrasts.append({'left': 'lr_small_step', 'right': name, 'panel': panel,
                                      'metric': metric, **paired(final_rows['lr_small_step'], rows, states, panel, metric)})
    result = {'evaluations': summaries, 'within_arm': comparisons,
              'against_original': original_comparisons, 'against_small_step': contrasts}
    (args.run / 'trial_comparisons.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, evaluations in summaries.items():
        for e in evaluations:
            print(name, e['iteration'], {panel: {'n': e[panel+'/all']['decisions'],
                  'matches': round(e[panel+'/all']['teacher_top_agreement']*e[panel+'/all']['decisions']),
                  'valid': round(e[panel+'/all']['valid_fraction']*e[panel+'/all']['decisions']),
                  'score': round(e[panel+'/all']['mean_teacher_probability'], 4)} for panel in ('train', 'probe')})
    print('Saved paired comparisons in', args.run / 'trial_comparisons.json')


if __name__ == '__main__':
    main()
