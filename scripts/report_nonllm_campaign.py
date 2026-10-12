#!/usr/bin/env python3
"""CPU-only cohort-complete campaign report; final results require explicit opt-in.

No model is loaded or executed. Final output verifies the frozen inputs and keeps
every requested cohort, missing panel, fault and censoring count visible.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from scripts.report_nonllm import summarize_live, paired_world_comparison
    from scripts.plot_nonllm_development import Canvas
except ModuleNotFoundError:
    from report_nonllm import summarize_live, paired_world_comparison
    from plot_nonllm_development import Canvas


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify_panel(metrics):
    rounds = metrics.get('rounds', []) if metrics.get('command') == 'evaluate' else [metrics]
    episodes = [e for r in rounds for e in r.get('episodes', [])]
    faults = [f for r in rounds for f in r.get('faults', [])]
    requested = [task for r in rounds for task in r.get('task_ids', [])]
    actual = [r.get('task_id') for r in episodes + faults]
    complete = bool(requested) and all(actual) and Counter(requested) == Counter(actual)
    complete = complete and all(not r.get('unstarted_tasks', 0) and r.get('cohort_completed') is not False for r in rounds)
    complete = complete and not any(e.get('termination_reason') == 'collector_budget' for e in episodes)
    return {'episodes': episodes, 'faults': faults, 'requested_ids': requested,
            'cohort_id': hashlib.sha256(json.dumps(sorted(requested), separators=(',', ':')).encode()).hexdigest()[:16],
            'cohort_complete': complete, 'requested': len(requested), 'classified': len(actual),
            'missing_ids': list((Counter(requested) - Counter(actual)).elements()),
            'extra_ids': list((Counter(actual) - Counter(requested)).elements()),
            'summary': summarize_live(episodes, faults)}


def compare_panels(left, right):
    if not left['cohort_complete'] or not right['cohort_complete']:
        return {'eligible': False, 'reason': 'requested cohort is incomplete'}
    if Counter(left['requested_ids']) != Counter(right['requested_ids']):
        return {'eligible': False, 'reason': 'requested task cohorts differ; no paired estimate'}
    return {'eligible': True, **paired_world_comparison(left['episodes'], right['episodes'],
               left_faults=left['faults'], right_faults=right['faults'])}


def selected_lineage(state):
    """Use journal/JSON ancestry and digests, never unpickle or initialize a model."""
    frozen = state.get('frozen_identities', {}).get('checkpoints', {})
    selected = {k: r['path'] for k, r in frozen.items()} or state.get('selected', {})
    result = {}
    for label, value in selected.items():
        if not isinstance(value, str): continue
        path = Path(value).resolve()
        row = {'path': str(path), 'exists': path.is_file()}
        if path.is_file(): row['sha256'] = digest(path)
        if label in frozen: row['frozen_sha256'] = frozen[label]['sha256']
        conditions = []
        for record in state.get('commands', []):
            tokens = list(map(str, record.get('command', [])))
            if '--output' not in tokens: continue
            output = Path(tokens[tokens.index('--output') + 1]).resolve()
            if output == path:
                conditions.append({'name': record.get('name'), 'status': record.get('status'),
                                   'parent_inputs': record.get('condition_signature', {}).get('inputs', {}),
                                   'source_identity': record.get('source_identity', {})})
        row['producing_conditions'] = conditions
        manifest = path.with_suffix('.composition.json')
        if manifest.is_file():
            report = json.loads(manifest.read_text())
            row['composition_provenance'] = report.get('composition_provenance')
            row['composition_report_matches_checkpoint'] = report.get('sha256') == row.get('sha256')
        result[label] = row
    return result



def final_evidence(name, metrics, record, classified, frozen):
    """Bind reported outcomes to frozen tasks, policies and executed interface.

    This is file/JSON validation only. Missing historical evidence fails closed;
    it cannot be reconstructed from a successful exit code.
    """
    from sts_policy.campaign_identity import condition_signature, input_paths
    from sts_policy.curriculum import TaskSpec, world_split
    errors = []
    if record.get('status') != 'complete': errors.append('journal command is not complete')
    if not classified['cohort_complete']: errors.append('requested cohort is incomplete')
    tokens = list(map(str, record.get('command', [])))
    flags = {}
    for i, token in enumerate(tokens):
        if token.startswith('--'):
            flag, separator, value = token.partition('=')
            flags[flag] = value if separator else (tokens[i+1] if i+1<len(tokens) and not tokens[i+1].startswith('--') else True)
    try:
        if condition_signature(tokens) != record.get('condition_signature'):
            errors.append('journal command/input signature differs from current inputs')
        inputs = input_paths(tokens)
        baseline = name.startswith('final_baseline_')
        kind = 'battle' if ('battle' in name or 'combat_extended' in name or name.endswith('_hard')) else 'ooc' if 'ooc' in name else 'run'
        max_act = 3 if name.endswith(('_act3','_hard')) or name=='final_composite' else 1
        horizon = 256 if kind=='battle' else frozen['generated_definitions']['horizon']
        expected_flags = {'--split':'final', '--seed':'777', '--max-act':str(max_act),
                          '--horizon':str(horizon), '--simulations':str(frozen['generated_definitions']['simulations'])}
        expected_flags['--kind' if baseline else '--task-kind'] = kind
        for key, value in expected_flags.items():
            if str(flags.get(key)) != value: errors.append('command differs from frozen protocol: '+key)
        config=metrics.get('config', {})
        if not baseline:
            defaults={'gamma':1.0,'reward_mode':'potential','terminal_hp_weight':0.0,'progress_floor_weight':0.05,'goal_act_clear':True}
            for key, value in defaults.items():
                expected=frozen['generated_definitions'].get(key,value)
                if config.get(key) != expected: errors.append('metrics reward/goal differs from frozen protocol: '+key)
        for key, value in expected_flags.items():
            config_key=key.removeprefix('--').replace('-','_')
            if str(config.get(config_key)) != value: errors.append('metrics config differs from command: '+key)
        if kind != 'battle':
            scope=kind+'_act'+str(max_act)
            expected_ids=frozen['generated_definitions']['panels'][scope]['task_ids']
            if str(flags.get('--world-start')) != str(frozen['generated_definitions']['world_start']):
                errors.append('generated world start differs from frozen protocol')
            if int(flags.get('--episodes',0)) != frozen['generated_definitions']['episodes']:
                errors.append('generated episode count differs from frozen protocol')
        else:
            role='later_act_battles' if name.endswith('_hard') else 'act1_battles'
            bank=frozen['panel_files'][role]
            if '--tasks' not in inputs or str(inputs['--tasks']) != bank['path']:
                errors.append('task bank differs from frozen protocol')
            tasks=[TaskSpec.from_dict(json.loads(line)) for line in Path(bank['path']).read_text().splitlines() if line.strip()]
            tasks=[replace(t,horizon=horizon,max_act=max_act,battle_simulations=int(expected_flags['--simulations']))
                   for t in tasks if world_split(t.seed)=='final']
            random.Random(777).shuffle(tasks)
            count=256 if role=='later_act_battles' else 160
            if int(flags.get('--episodes',0)) != count: errors.append('battle episode count differs from frozen protocol')
            expected_ids=[hashlib.sha256(json.dumps(t.to_dict(),sort_keys=True,separators=(',',':')).encode()).hexdigest() for t in tasks[:count]]
        if Counter(classified['requested_ids']) != Counter(expected_ids):
            errors.append('requested task identities differ from frozen cohort')
        expected_models={} if baseline else ({'--combat-checkpoint':'combat','--ooc-checkpoint':'ooc'} if name=='final_composite'
                     else {'--checkpoint':'combat' if kind=='battle' else 'ooc' if kind=='ooc' else 'shared'})
        actual_models={key:path for key,path in inputs.items() if 'checkpoint' in key}
        if set(actual_models) != set(expected_models): errors.append('checkpoint input flags differ from selected policy')
        for flag,role in expected_models.items():
            identity=frozen['checkpoints'][role]
            if flag not in actual_models or str(actual_models[flag]) != identity['path'] or digest(actual_models[flag]) != identity['sha256']:
                errors.append('checkpoint input differs from frozen selection: '+role)
            if str(config.get(flag.removeprefix('--').replace('-','_'))) != str(flags.get(flag)):
                errors.append('metrics checkpoint path differs from executed command')
        if baseline and (flags.get('--baseline') != name.split('_')[2] or config.get('baseline') != flags.get('--baseline')):
            errors.append('baseline label differs from executed command')
        backend=frozen['backend']
        env=metrics.get('training_environment', {})
        if env.get('native_module_sha256') != backend['module']['sha256'] or env.get('sourced_env_manifest') != backend['provenance'] or env.get('sourced_env_manifest_sha256') != backend['manifest']['sha256']:
            errors.append('executed backend evidence is missing or differs from freeze')
        core=sorted((role.removeprefix('src/sts_env/'), row) for role,row in frozen['panel_files'].items() if role.startswith('src/sts_env/') and role.endswith('.py'))
        interface=hashlib.sha256()
        for relative,row in core:
            interface.update(relative.encode());interface.update(b'\0');interface.update(Path(row['path']).read_bytes());interface.update(b'\0')
        if not core or env.get('python_interface_sha256') != interface.hexdigest():
            errors.append('executed Python interface evidence is missing or differs from freeze')
        for result in classified['episodes']+classified['faults']:
            provenance=result.get('native_provenance')
            if provenance is not None and provenance != backend['provenance']:
                errors.append('worker native provenance differs from freeze');break
    except (KeyError, ValueError, TypeError, OSError, IndexError) as error:
        errors.append('cannot verify frozen panel evidence: '+str(error))
    return {'final_evidence_verified': not errors, 'readiness_errors': errors}


def collect_campaign(campaign, *, stage='development', allow_final=False):
    if stage == 'final' and not allow_final:
        raise ValueError('reading final results requires explicit --allow-final-results')
    state = json.loads((campaign/'campaign.json').read_text())
    commands = {r['name']: r for r in state.get('commands', [])}
    freeze_status = {'present': bool(state.get('frozen_identities')), 'verified': None}
    if stage == 'final':
        if state.get('frozen_identities'):
            from sts_policy.final_freeze import verify_final_freeze
            try:
                verify_final_freeze(state['frozen_identities'], backend_build_dir=state.get('native_build'))
                freeze_status['verified'] = True
            except Exception as error:
                freeze_status.update(verified=False, error=str(error))
        names = state.get('final_required_panels', [])
        if not names:
            raise ValueError('final_required_panels is absent; do not invent the final protocol')
        paths = [(name, campaign/(name+'.metrics.json')) for name in names]
    else:
        paths = [(p.name.removesuffix('.metrics.json'), p) for p in sorted(campaign.glob('*_dev.metrics.json'))]
    panels, missing = [], []
    for name, path in paths:
        if not path.is_file(): missing.append(name); continue
        raw = path.read_bytes(); metrics = json.loads(raw)
        split = metrics.get('config', {}).get('split')
        wanted = 'final' if stage == 'final' else 'dev'
        if split not in (None, wanted):
            raise ValueError(f'{name} has split {split!r}, not {wanted}')
        classified = classify_panel(metrics)
        if any(e.get('split') not in (None, wanted) for e in classified['episodes'] + classified['faults']):
            raise ValueError(f'{name} mixes evaluation splits')
        record = commands.get(name, {})
        config = metrics.get('config', {})
        policies = {key: config.get(key) for key in ('checkpoint', 'combat_checkpoint', 'ooc_checkpoint', 'baseline') if config.get(key)}
        stamp = record.get('finished') or path.stat().st_mtime
        panels.append({'name': name, 'source': str(path.resolve()), 'metrics_sha256': hashlib.sha256(raw).hexdigest(),
                       'journal_digest_matches': record.get('metrics_sha256') == hashlib.sha256(raw).hexdigest(),
                       'command_status': record.get('status'), 'timestamp': stamp,
                       'local_time': datetime.fromtimestamp(stamp, ZoneInfo('Europe/London')).isoformat(),
                       'scope': {key: config.get(key) for key in ('split','task_kind','kind','max_act','horizon','simulations','seed')},
                       'policy': policies, 'model_config': metrics.get('model_config'), **classified,
                       **(final_evidence(name,metrics,record,classified,state.get('frozen_identities',{})) if stage=='final' else {})})
    if stage == 'final':
        for panel in panels:
            if not panel['journal_digest_matches']:
                panel['readiness_errors'].append('exact saved journal metrics SHA is missing or stale')
                panel['final_evidence_verified']=False
    panels.sort(key=lambda r: r['timestamp'], reverse=True)
    pairs, duplicates = {}, []
    # Do not manufacture a grid of hypothesis tests from reused development
    # snapshots. Fixed final policy panels receive paired cohort estimates.
    for i, left in enumerate(panels if stage == 'final' else []):
        for right in panels[i+1:]:
            if left['cohort_id'] != right['cohort_id']: continue
            key = left['name'] + '_minus_' + right['name']
            if left['policy'] == right['policy']:
                duplicates.append([left['name'], right['name']])
            else:
                pairs[key] = (compare_panels(left, right) if freeze_status['verified'] is True and
                              all(p['command_status']=='complete' and p['journal_digest_matches'] and p['final_evidence_verified'] for p in (left,right))
                              else {'eligible':False,'reason':'final command/input/backend evidence is unverified'})
    return {'stage': stage, 'scope': 'final' if stage == 'final' else 'development only',
            'campaign_status': state.get('status'), 'freeze': freeze_status, 'missing_panels': missing,
            'panels': panels, 'paired_comparisons': pairs, 'duplicate_policy_cohort_panels': duplicates,
            'selected_checkpoint_lineage': selected_lineage(state),
            'final_report_complete': stage == 'final' and freeze_status['verified'] is True and not missing and
                bool(panels) and all(p['cohort_complete'] and p['journal_digest_matches'] and
                    p['command_status']=='complete' and p['final_evidence_verified'] for p in panels),
            'notes': ['Faults never count as ordinary losses; conditional success estimates retain censoring counts.',
                      'World-cluster intervals give each world equal weight; isolated battles are not whole-run competence.',
                      'Counts and world N are primary. All-success or all-failure small cohorts can produce degenerate empirical bootstrap intervals; those intervals do not establish zero uncertainty or competence.',
                      'Act 1 clear is distinct from full-game victory. Native search is privileged and OOC native choices are scripted.',
                      'Endpoint HP after act transitions includes automatic healing; it is not run-long HP preservation.',
                      'Duplicate policy/cohort panels are listed explicitly and are never pooled as independent trials.',
                      'Development Act 3 uses horizon 4096 and OOC uses 1024; final uses 2048. Differences include censoring and cannot be attributed solely to generalization.']}


def write_outputs(report, output, *, png=False):
    output.mkdir(parents=True, exist_ok=True)
    compact = {**report, 'panels': [{k:v for k,v in p.items() if k not in ('episodes','faults','requested_ids')} for p in report['panels']]}
    (output/'campaign_report.json').write_text(json.dumps(compact, indent=2, sort_keys=True)+'\n')
    rows = []
    for panel in report['panels']:
        for kind, summary in panel['summary'].items():
            rows.append({'panel': panel['name'], 'local_time': panel['local_time'], 'kind': kind,
                         'max_act': panel['scope'].get('max_act'), 'horizon': panel['scope'].get('horizon'), 'cohort_id': panel['cohort_id'],
                         'requested': panel['requested'], 'classified': panel['classified'],
                         'complete': panel['cohort_complete'], 'command_status': panel['command_status'],
                         'evidence_verified': panel.get('final_evidence_verified'), 'wins': summary['successes'],
                         'losses': summary['losses'], 'faults': summary['faults'], 'truncated': summary['truncated'],
                         'unresolved': summary['unresolved'], 'worlds': summary['decisive_success']['worlds'],
                         'world_mean_success': summary['decisive_success']['world_mean'],
                         'world_ci95': json.dumps(summary['decisive_success']['ci95'])})
    if rows:
        with (output/'campaign_panels.csv').open('w', newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    markdown = ['# '+report['scope'].title()+' campaign results', '',
                f"Missing panels: {', '.join(report['missing_panels']) or 'none'}. Freeze verified: {report['freeze']['verified']}. Final report complete: {report['final_report_complete']}.", '',
                '| Panel, latest first | Horizon | Requested | Wins | Defeats | Faults | Truncated | Cohort complete | Evidence |',
                '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    markdown += [f"| {r['panel']} | {r['horizon']} | {r['requested']} | {r['wins']} | {r['losses']} | {r['faults']} | {r['truncated']} | {r['complete']} | {r['evidence_verified']} |" for r in rows]
    markdown += ['',*['Unverified '+p['name']+': '+ '; '.join(p['readiness_errors']) for p in report['panels'] if p.get('readiness_errors')], '',*report['notes']]
    (output/'campaign_report.md').write_text('\n'.join(markdown)+'\n')
    if report['stage'] == 'development':
        selected = [p for p in report['panels'] if p['name'].startswith('joint_act1_') and 'run' in p['summary']]
        selected.sort(key=lambda p:p['timestamp'])
        if selected:
            c=Canvas(1400,650,png=png);c.text(70,30,'Corrected 0.1.1: development Act 1 full-control learning',30,bold=True)
            c.text(70,78,'Observed clears of requested worlds; faults and truncations are marked separately. No final results.',18,'#607286')
            x0,x1,y0,y1=100,1260,140,440
            n=max(1,len(selected)-1)
            for tick in range(0,101,20):
                y=y1-(y1-y0)*tick/100;c.line(x0,y,x1,y);c.text(x0-12,y-9,str(tick)+'%',15,'#607286','end')
            previous=None
            best=max(selected,key=lambda p:p['summary']['run']['successes']/max(1,p['requested']))
            for index,p in enumerate(selected):
                summary=p['summary']['run'];rate=summary['successes']/max(1,p['requested'])
                x=x0+(x1-x0)*index/n;y=y1-(y1-y0)*rate
                if previous and previous[2]==p['cohort_id']:c.line(previous[0],previous[1],x,y,'#2563a6',2)
                c.circle(x,y,5,'#c94c4c' if summary['faults'] or summary['truncated'] else '#2563a6')
                if index in (0,len(selected)-1) or p['name']=='joint_act1_0007_dev' or p['name']==best['name']:
                    c.text(x,y-30,f"{summary['successes']}/{p['requested']}"+(' best dev' if p['name']==best['name'] else ''),18,'#172b40','middle',True)
                    c.text(x,y1+18,p['name'].replace('joint_act1_','').removesuffix('_dev'),15,'#607286','middle')
                previous=x,y,p['cohort_id']
            last=selected[-1];s=last['summary']['run']
            c.text(100,520,f"Latest: {last['name']} — {s['successes']} clears, {s['losses']} defeats, {s['faults']} faults, {s['truncated']} truncated",22,bold=True)
            c.text(100,568,'Connected points have identical requested task IDs. Red markers indicate separate fault/censoring cases.',17,'#607286')
            c.save(output/'native011_act1_development')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,default=Path('data/nonllm_native_revision/campaign'))
    parser.add_argument('--output',type=Path,default=Path('data/nonllm_native_revision/report'))
    parser.add_argument('--stage',choices=('development','final'),default='development')
    parser.add_argument('--allow-final-results',action='store_true')
    parser.add_argument('--png',action='store_true')
    args=parser.parse_args()
    if args.stage=='final' and not args.allow_final_results:parser.error('final reads require --allow-final-results')
    if args.png:
        try:import PIL
        except ImportError:parser.error('--png requires optional Pillow; SVG needs only the standard library')
    report=collect_campaign(args.campaign,stage=args.stage,allow_final=args.allow_final_results)
    write_outputs(report,args.output,png=args.png)
    print(json.dumps({'stage':args.stage,'panels':len(report['panels']),'missing':report['missing_panels'],
                      'freeze':report['freeze'],'report_complete':report['final_report_complete'],'output':str(args.output)},sort_keys=True))


if __name__=='__main__':main()
