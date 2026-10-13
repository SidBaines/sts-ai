"""Bounded raw-native regressions for the separate potion action-space candidate."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import multiprocessing
from pathlib import Path
import sys
import traceback


def scene(gc):
    return {'screen':str(gc.screen_state),'floor':gc.floor_num,'act':gc.act,'x':gc.cur_map_node_x,
            'y':gc.cur_map_node_y,'event':gc.current_event,'gold':gc.gold,'deck_size':len(gc.deck)}


def run_case(build):
    sys.path.insert(0,str(build));sts=importlib.import_module('slaythespire')
    gc=sts.GameContext(sts.CharacterClass.IRONCLAD,11,0)
    gc.gold=100;gc.max_hp=80;gc.cur_hp=10
    ordinary=[a.bits for a in gc.legal_actions()]
    gc.obtain_potion(sts.Potion.BLOOD_POTION)
    actions=gc.legal_actions()
    assert [a.bits for a in actions[:len(ordinary)]]==ordinary
    drinks=[a for a in actions if gc.public_action(a)['kind']=='drink_potion']
    discards=[a for a in actions if gc.public_action(a)['kind']=='discard_potion']
    assert len(drinks)==len(discards)==1
    before=scene(gc);meta=dict(gc.public_action(drinks[0]))
    assert meta['source_kind']=='potion' and meta['source_index']==0 and meta['heal']==16
    drinks[0].execute(gc)
    assert gc.cur_hp==26 and scene(gc)==before
    gc.obtain_potion(sts.Potion.FRUIT_JUICE)
    fruit=next(a for a in gc.legal_actions() if gc.public_action(a)['kind']=='drink_potion')
    assert gc.public_action(fruit)['max_hp_gain']==5
    fruit.execute(gc)
    assert gc.max_hp==85 and gc.cur_hp==31 and scene(gc)==before
    gc.obtain_potion(sts.Potion.FIRE_POTION)
    choices=gc.legal_actions();meta=[dict(gc.public_action(a)) for a in choices]
    assert not any(d['kind']=='drink_potion' for d in meta)
    discard=next(a for a in choices if gc.public_action(a)['kind']=='discard_potion')
    discard.execute(gc)
    assert scene(gc)==before and gc.cur_hp==31
    rejected=False
    try:discard.execute(gc)
    except ValueError:rejected=True
    assert rejected
    # The same potion operation on a rest screen must not also rest/heal.
    gc.screen_state=sts.ScreenState.REST_ROOM;gc.cur_hp=10;gc.max_hp=80
    gc.obtain_potion(sts.Potion.BLOOD_POTION)
    before_rest=scene(gc)
    blood=next(a for a in gc.legal_actions() if gc.public_action(a)['kind']=='drink_potion')
    blood.execute(gc)
    assert gc.cur_hp==26 and scene(gc)==before_rest
    return {'native_module_sha256':hashlib.sha256((build/'slaythespire.cpython-312-darwin.so').read_bytes()).hexdigest(),
            'ordinary_indices_preserved':True,'blood_heal':16,'fruit_max_hp_gain':5,'fire_drink_absent':True,
            'discard_has_no_screen_side_effect':True,'invalid_repeat_discard_rejected':True,
            'rest_screen_has_no_fallthrough':True,'scope':'Raw candidate only; current API-3 profile remains active.'}


def child(connection,build):
    try:connection.send({'status':'ok','result':run_case(build)})
    except Exception as error:connection.send({'status':'error','error':str(error),'traceback':traceback.format_exc()})
    finally:connection.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    ctx=multiprocessing.get_context('spawn');parent,remote=ctx.Pipe(False)
    process=ctx.Process(target=child,args=(remote,args.build.resolve()));process.start();remote.close()
    result=parent.recv() if parent.poll(20) else {'status':'timeout'}
    process.join(.5)
    if process.is_alive():process.kill();process.join()
    process.close();parent.close();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))
    if result['status']!='ok':raise SystemExit(1)


if __name__=='__main__':main()
