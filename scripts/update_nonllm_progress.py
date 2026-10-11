#!/usr/bin/env python3
"""Render a durable, human-readable experiment log from the atomic campaign journal."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')


def metrics_summary(metrics):
    rows = []
    training = metrics.get('training')
    if training:
        seconds = training.get('elapsed_seconds', 0)
        rows.append(f"Training: {training.get('steps', 0):,} updates, {training.get('examples', 0):,} examples, "
                    f"{seconds:.2f}s; {training.get('examples', 0)/max(seconds, .001):,.0f} examples/s; "
                    f"mean loss {training.get('loss', 0):.4f}.")
    development = metrics.get('development')
    if development:
        rows.append('Development imitation: ' + json.dumps(development, sort_keys=True) + '.')
    rounds = metrics.get('rounds', [])
    episodes = [episode for row in rounds for episode in row.get('episodes', [])]
    faults = [fault for row in rounds for fault in row.get('faults', [])]
    if episodes or faults:
        outcomes = Counter(x.get('outcome', 'unknown') for x in episodes)
        floors = [x['floor'] for x in episodes if isinstance(x.get('floor'), (int, float))]
        hp = [x['hp_cost'] for x in episodes if isinstance(x.get('hp_cost'), (int, float))]
        rows.append(f"Live collection: {len(rounds)} rounds, {len(episodes)} non-fault episodes, "
                    f"{len(faults)} surfaced faults, {sum(bool(x.get('truncated')) for x in episodes)} truncations; "
                    f"outcomes {dict(outcomes)}; {sum(r.get('transitions', 0) for r in rounds):,} valid transitions.")
        if floors:
            rows.append(f"Mean terminal/cutoff floor {sum(floors)/len(floors):.2f}; "
                        f"mean fight HP cost {sum(hp)/len(hp):.2f}." if hp else
                        f"Mean terminal/cutoff floor {sum(floors)/len(floors):.2f}.")
        updates = [row['update'] for row in rounds if row.get('update')]
        if updates:
            rows.append(f"PPO: {sum(x.get('steps', 0) for x in updates):,} optimizer steps; "
                        f"last update {json.dumps({k:v for k,v in updates[-1].items() if k != 'config'}, sort_keys=True)}.")
        if faults:
            rows.append('Fault evidence: ' + json.dumps(faults[:5], sort_keys=True) + '.')
        if metrics.get('command') != 'evaluate':
            rows.append('Training-round outcomes combine changing policies; use separate development evaluations for model comparisons.')
    if metrics.get('elapsed_seconds') is not None:
        rows.append(f"Total job wall time {metrics['elapsed_seconds']:.2f}s.")
    return rows


def render(campaign, output, cache):
    state = json.loads(campaign.read_text())
    lines = ['# Non-LLM curriculum experiment progress log', '',
             f"Updated {utc(time.time())}. Campaign status: **{state['status']}**; phase: **{state['phase']}**.", '',
             f"Allocation: {utc(state['started_at'])} → {utc(state['deadline'])}. "
             'Branch `codex/nonllm-curriculum`, immutable starting release `env-v0.1.0`.', '',
             'This page is generated from the atomic campaign journal and raw metrics. '
             'It includes failures and incomplete jobs. [Decision notes and recovery instructions](nonllm_progress_notes.md) '
             'record reasoning; [the experiment plan](nonllm_curriculum_plan.md) records hypotheses. '
             'Development results are not final-test results. Simulator faults are separate from ordinary game defeats.', '',
             '## Current selection', '', '```json', json.dumps(state.get('selected', {}), indent=2, sort_keys=True), '```', '',
             '## Experiment timeline', '']
    for record in state['commands']:
        lines += [f"### {utc(record['started'])} — {record['name']}", '',
                  f"Status: **{record['status']}**. " +
                  (f"Finished {utc(record['finished'])}; exit code {record.get('exit_code')}." if record.get('finished') else
                   f"PID {record.get('pid', 'pending')}; last supervisor heartbeat {utc(state['heartbeat'])}."), '',
                  'Exact command:', '', '```text', ' '.join(record['command']), '```', '']
        path = Path(record['metrics'])
        belongs_to_later_run = record.get('finished') and path.exists() and path.stat().st_mtime > record['finished'] + 5
        if path.exists() and not belongs_to_later_run:
            identity = (str(path), path.stat().st_mtime_ns)
            if identity not in cache:
                try:
                    cache[identity] = metrics_summary(json.loads(path.read_text()))
                except (OSError, json.JSONDecodeError):
                    cache[identity] = ['Metrics are being updated; inspect the raw atomic file after the next refresh.']
            lines += [paragraph + '\n' for paragraph in cache[identity]]
            lines += [f"Raw metrics: `{path}`. Log: `{path.parent / (record['name'] + '.log')}`.", '']
        else:
            lines += [('Metrics at this path belong to a later rerun; this invocation has no completed metrics.' if belongs_to_later_run else
                       'No completed metrics yet; inspect the condition log for a running job or failure.'), '']
    if state.get('frozen_selection'):
        lines += ['## Final evaluation freeze', '', '```json', json.dumps(state['frozen_selection'], indent=2), '```', '']
    lines += ['## Reproduce or resume', '',
              'Use this experiment worktree and its `.venv`; set `PYTHONPATH=src` and '
              f"`STS_LIGHTSPEED_BUILD={state['native_build']}`. The supervisor script is "
              '`scripts/run_nonllm_campaign.py`; the raw journal is `data/nonllm_v1/campaign/campaign.json`. '
              'Check the existing PID and campaign lock before starting another learner. Preserve all checkpoints '
              'and the original UTC deadline. Follow the decision notes for recovery details.', '']
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text('\n'.join(lines))
    os.replace(temporary, output)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT/'data/nonllm_v1/campaign/campaign.json')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/nonllm_progress.md')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    cache = {}
    while True:
        state = render(args.campaign, args.output, cache)
        if not args.watch or state['status'] in ('finished', 'failed') or time.time() > state['deadline'] + 60:
            break
        time.sleep(30)
