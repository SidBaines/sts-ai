"""Compare saved trial evaluations; this does not load a model or change a run."""
import argparse
import csv
import json
from pathlib import Path

from sts_llm.statistics import paired_cluster_bootstrap


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True)
            if line.endswith('\n') and line.strip()]


def paired(before, after, states, panel, metric, phase=None):
    left = [r for r in before if r['set'] == panel and (phase is None or r['phase'] == phase)]
    right = [r for r in after if r['set'] == panel and (phase is None or r['phase'] == phase)]
    assert {(r['id'], tuple(r['order'])) for r in left} == {(r['id'], tuple(r['order'])) for r in right}
    def pack(rows):
        return [{'prompt_id': r['id'], 'world_id': states[r['id']]['world_identity'],
                 'value': float(r[metric])} for r in rows]
    result = paired_cluster_bootstrap(pack(left), pack(right))
    result.pop('pairs')
    return result


def export_figures(result, folder):
    """Save every evaluation and contrast, plus one compact research figure."""
    folder.mkdir(parents=True, exist_ok=True)
    evaluations = []
    for arm, rows in result['evaluations'].items():
        for row in rows:
            for key, values in row.items():
                if '/' not in key or not values:
                    continue
                evaluations.append({'arm': arm, 'iteration': row['iteration'], 'panel_phase': key,
                                    **values})
    with (folder / 'evaluations.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evaluations[0]))
        writer.writeheader(); writer.writerows(evaluations)
    contrasts = []
    fields = ['comparison', 'left', 'right', 'arm', 'iteration', 'panel', 'phase', 'subset',
              'metric', 'n_pairs', 'n_worlds', 'left_mean', 'right_mean', 'difference', 'ci_low', 'ci_high']
    for kind in ('within_arm', 'against_original', 'against_small_step', 'full_panel', 'full_probe_subsets'):
        for row in result[kind]:
            contrasts.append({'comparison': kind, **{k: row[k] for k in fields if k in row},
                              'ci_low': row['ci95'][0], 'ci_high': row['ci95'][1]})
    with (folder / 'comparisons.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(contrasts)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    arms = [('lr_small_step', 'Smaller updates'), ('lr_kl_anchor', 'KL anchor'),
            ('lr_answer_credit', 'Answer credit'), ('lr_length_cost', 'Length cost')]
    for i, (arm, label) in enumerate(arms):
        row = [x for x in result['within_arm'] if x['arm'] == arm and x['panel'] == 'probe'
               and x['metric'] == 'teacher_top_agreement'][-1]
        value = 100 * row['difference']; low, high = [100*x for x in row['ci95']]
        axes[0].errorbar(value, i, xerr=[[value-low], [high-value]], fmt='o', color='#176c80', capsize=3)
    axes[0].set(yticks=range(4), yticklabels=[x[1] for x in arms],
                title='Four short screens: 48 held-out states', xlabel='Agreement change (percentage points)')
    axes[0].axvline(0, color='#888', linewidth=1); axes[0].invert_yaxis()
    rows = result['evaluations']['lr_continue_small_step']
    axes[1].plot([r['iteration'] for r in rows], [r['probe/all']['median_reasoning_tokens'] for r in rows],
                 marker='o', color='#a45c25')
    for r in rows:
        axes[1].annotate(f"{round(48*r['probe/all']['capped_fraction'])}/48 capped",
                         (r['iteration'], r['probe/all']['median_reasoning_tokens']),
                         xytext=(0, 10 if r['iteration'] != 24 else -20), textcoords='offset points',
                         ha='center', fontsize=8)
    axes[1].set(title='Continuation: growing reasoning', xlabel='Additional updates',
                ylabel='Median reasoning tokens', ylim=(100, 450), xlim=(-3, 30))
    labels = []
    for i, row in enumerate(x for x in result['full_panel'] if x['metric'] == 'teacher_top_agreement'):
        value = 100 * row['difference']; low, high = [100*x for x in row['ci95']]
        axes[2].errorbar(value, i, xerr=[[value-low], [high-value]], fmt='o', color='#176c80', capsize=3)
        labels.append(('Train' if row['panel']=='train' else 'Held out') + ': ' +
                      {'all':'all', 'combat':'combat', 'out_of_combat':'out of combat'}[row['phase']])
    axes[2].set(yticks=range(len(labels)), yticklabels=labels, title='Selected adapter: full panels',
                xlabel='Agreement change (percentage points)')
    axes[2].axvline(0, color='#888', linewidth=1); axes[2].invert_yaxis()
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Exploratory learning trials — paired 95% world-cluster intervals', fontsize=12)
    fig.savefig(folder / 'outcome.png', dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('data/llm_runs/learning_trials_20261027'))
    parser.add_argument('--figures', type=Path, help='Export CSVs and the completed-study figure')
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
    full_comparisons, probe_subsets = [], []
    if summaries.get('lr_full_parent') and summaries.get('lr_full_candidate'):
        before = read_rows(args.run / 'lr_full_parent/evaluations/iter_0000.jsonl')
        after = read_rows(args.run / 'lr_full_candidate/evaluations/iter_0000.jsonl')
        for panel in ('train', 'probe'):
            for phase in (None, 'out_of_combat', 'combat'):
                for metric in ('teacher_top_agreement', 'teacher_probability', 'valid', 'capped'):
                    full_comparisons.append({'panel': panel, 'phase': phase or 'all', 'metric': metric,
                                             **paired(before, after, states, panel, metric, phase)})
        screened_ids = {r['id'] for r in original if r['set'] == 'probe'}
        for screened in (True, False):
            left = [r for r in before if (r['id'] in screened_ids) == screened]
            right = [r for r in after if (r['id'] in screened_ids) == screened]
            for metric in ('teacher_top_agreement', 'teacher_probability', 'valid'):
                probe_subsets.append({'subset': 'screened_states' if screened else 'other_states',
                                      'metric': metric, **paired(left, right, states, 'probe', metric)})
    result = {'evaluations': summaries, 'within_arm': comparisons,
              'against_original': original_comparisons, 'against_small_step': contrasts,
              'full_panel': full_comparisons, 'full_probe_subsets': probe_subsets}
    (args.run / 'trial_comparisons.json').write_text(json.dumps(result, indent=2) + '\n')
    if args.figures:
        export_figures(result, args.figures)
    for name, evaluations in summaries.items():
        for e in evaluations:
            print(name, e['iteration'], {panel: {'n': e[panel+'/all']['decisions'],
                  'matches': round(e[panel+'/all']['teacher_top_agreement']*e[panel+'/all']['decisions']),
                  'valid': round(e[panel+'/all']['valid_fraction']*e[panel+'/all']['decisions']),
                  'score': round(e[panel+'/all']['mean_teacher_probability'], 4)} for panel in ('train', 'probe')})
    print('Saved paired comparisons in', args.run / 'trial_comparisons.json')


if __name__ == '__main__':
    main()
