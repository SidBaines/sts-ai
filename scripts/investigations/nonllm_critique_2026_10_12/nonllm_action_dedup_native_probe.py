"""Bounded, disposable native replay of a real TRAIN free-once collision."""
from __future__ import annotations
import json
import multiprocessing
from pathlib import Path

from sts_policy.curriculum import TaskSpec, public_hash

TEACHER=Path('/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_curriculum/teacher_act3_200_native011_seed12000/teacher.jsonl')
BUILD='/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/external/sts_lightspeed_v011/build'


def native_probe(connection, spec):
    try:
        from sts_policy.workers import CurriculumEpisode
        episode=CurriculumEpisode({'build_dir':BUILD,'fast_replay':True})
        before,info=episode.reset(spec)
        backend=episode.episode.env._env
        raw,display,mapping=backend._action_views()
        matching=[]
        for index,action in enumerate(raw):
            if str(action.action_type()).endswith('CARD') and action.source_idx() in (1,2):
                matching.append({'raw_index':index,'source_hand_index':action.source_idx(),
                                 'description':action.describe(backend.bc,include_card_type=True,include_computed_damage=True),
                                 'displayed':index in mapping,
                                 'display_mapping':display[mapping.index(index)].index if index in mapping else None})
        branches=[]
        for choice in matching:
            episode.reset(spec)
            backend=episode.episode.env._env
            action=backend.raw_actions()[choice['raw_index']]
            energy_before=int(backend.bc.player_energy)
            action.execute(backend.bc)
            backend.advance_to_decision()
            after=episode.episode._observe()
            branches.append({**choice,'energy_before':energy_before,
                             'energy_after':after['state']['combat']['player_energy'],
                             'remaining_hand':after['state']['combat']['cards']['hand'],
                             'public_after_hash':public_hash(after)})
        connection.send({'status':'ok','world_seed':spec.seed,'prefix':list(spec.prefix),
                         'ooc_previews':list(spec.ooc_previews),'public_start_hash':public_hash(before),
                         'public_start':before,'native_provenance':info['native_provenance'],
                         'raw_strike_branches':branches,'displayed_actions':[a.description for a in display]})
    except BaseException as error:
        connection.send({'status':'error','type':type(error).__name__,'error':str(error)})
    finally:
        connection.close()


def main():
    rows=[]
    with TEACHER.open() as handle:
        for line in handle:
            row=json.loads(line)
            if row.get('world_seed')==12022 and row.get('split')=='train':
                rows.append(row)
                if row['step']==14:break
    rows.sort(key=lambda r:r['step'])
    assert [row['step'] for row in rows]==list(range(15))
    expected=public_hash(rows[-1]['observation'])
    spec=TaskSpec(seed=12022,kind='run',max_act=3,horizon=10,
                  prefix=tuple(row['executed_index'] for row in rows[:-1]),
                  ooc_previews=tuple(row['phase']=='out_of_combat' for row in rows[:-1]),
                  expected_public_hash=expected,seek_combat=False,battle_simulations=200)
    context=multiprocessing.get_context('spawn')
    parent,child=context.Pipe()
    process=context.Process(target=native_probe,args=(child,spec))
    process.start();child.close()
    if parent.poll(15):
        result=parent.recv()
    else:
        result={'status':'timeout','seconds':15}
    process.join(2)
    if process.is_alive():process.terminate();process.join(2)
    parent.close()
    Path('/private/tmp/nonllm_action_dedup_native_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='public_start'},indent=2))


if __name__=='__main__':
    main()
