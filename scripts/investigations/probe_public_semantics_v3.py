"""Bounded intervened native fixtures for unrepresented public semantics."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
from pathlib import Path
import traceback

import numpy as np

from sts_policy import entities
from sts_policy.curriculum import world_split,TaskSpec,public_hash
from sts_policy.workers import CurriculumEpisode
from sts_env.env import _Episode
from scripts.run_strategic_v3 import atomic_json


def new_episode(seed,mode="full"):
    episode=_Episode({"mode":mode,"ascension":0,"max_act":1,"max_steps":200,"battle_simulations":50,
                      "ooc_seed":seed,"build_dir":os.environ["STS_LIGHTSPEED_BUILD"]},seed)
    episode.reset()
    return episode


def equal_candidates(obs,first,second):
    row=entities.encode_observation(obs)
    return {name:bool(np.array_equal(getattr(row,name)[first],getattr(row,name)[second]))
            for name in ("action_ids","action_values","action_links","legacy_actions")}


def probe_bottle():
    bank=Path("data/nonllm_v3/banks/native_train_expand256/worlds")
    filenames=subprocess.check_output(["rg","-l",'"bottled":true',str(bank)],text=True).splitlines()
    for filename in filenames:
        world=json.loads(Path(filename).read_text());prefix=[];previews=[]
        for record in world["records"]:
            obs=record["observation"];deck=obs["state"]["inventory"]["deck"]
            bottled=[c for c in deck if c.get("bottled") and c.get("upgrade_count")==0]
            candidates=[c for c in bottled if sum(d["base_name"]==c["base_name"] and d["upgrade_count"]==0 for d in deck)>=2]
            smiths=[a for a in obs["legal_actions"] if a["description"].startswith("smith")]
            if candidates and smiths and str(obs["state"]["screen_state"]).endswith("REST_ROOM"):
                spec=TaskSpec(seed=world["seed"],kind="run",prefix=tuple(prefix),ooc_previews=tuple(previews),
                    expected_public_hash=public_hash(obs),expected_provenance=world["provenance"],
                    horizon=100,max_act=3,battle_simulations=world["settings"]["simulations"])
                def replay():
                    adapter=CurriculumEpisode({"build_dir":os.environ["STS_LIGHTSPEED_BUILD"],"fast_observations":True,"fast_replay":True})
                    adapter.reset(spec)
                    selected=adapter.step(smiths[0]["index"])[0]
                    return adapter,selected
                adapter,before=replay(); name=candidates[0]["base_name"]
                positions=[i for i,a in enumerate(before["legal_actions"]) if a["public"].get("card",{}).get("base_name")==name]
                outcomes=[]
                for i in positions:
                    adapter,current=replay();after=adapter.step(current["legal_actions"][i]["index"])[0]
                    upgraded=[c["index"] for c in after["state"]["inventory"]["deck"]
                              if c["upgrade_count"]>next(d["upgrade_count"] for d in before["state"]["inventory"]["deck"] if d["index"]==c["index"])]
                    outcomes.append({"candidate_position":i,"upgraded_deck_indices":upgraded,
                        "bottled_upgraded":any(c["index"] in upgraded and c.get("bottled") for c in after["state"]["inventory"]["deck"])})
                bottle=next((o for o in outcomes if o["bottled_upgraded"]),None)
                normal=next((o for o in outcomes if not o["bottled_upgraded"]),None)
                if bottle and normal:
                    return {"fixture":"Recorded TRAIN prefix to real rest room; alternative native smith and upgrade choices",
                        "seed":world["seed"],"source_world":filename,"task":spec.to_dict(),"before":before,
                        "outcomes":outcomes,"identical_candidate_inputs":equal_candidates(before,bottle["candidate_position"],normal["candidate_position"])}
            prefix.append(record["executed_index"]);previews.append(obs["phase"]=="out_of_combat")
    raise RuntimeError("No bounded natural bottled-duplicate smith frontier")


def advance_to_combat(episode):
    for _ in range(32):
        obs=episode._observe()
        if obs["phase"]=="combat":
            return obs
        episode.step(obs["legal_actions"][0]["index"])
    raise RuntimeError("no first combat within fixture bound")


def probe_gold():
    for seed in (s for s in range(10,100) if world_split(s)=="train"):
        episode=new_episode(seed)
        episode.env.gc.gold=100
        for _ in range(30):
            episode.env.gc.obtain_card(episode.env.sts.Card(episode.env.sts.CardId.HAND_OF_GREED))
        before=advance_to_combat(episode)
        enemies=before["state"]["combat"]["enemies"]
        alive=[e for e in enemies if e["alive"]]
        if len(alive)<2:
            continue
        possible=[]
        for action in before["legal_actions"]:
            public=action["public"]
            target=next((e for e in alive if e["index"]==public.get("target_index")),None)
            if public.get("card",{}).get("base_name")=="Hand of Greed" and target and target["cur_hp"]<=20:
                possible.append(action)
        if not possible:
            continue
        after=episode.step(possible[0]["index"])[0]
        if after["phase"]=="combat" and after["state"]["combat"]["player"]["gold"]>100:
            encoded=entities.encode_observation(after)
            end=next(a['index'] for a in after['legal_actions'] if a['public']['kind']=='end_turn')
            after_turn=episode.step(end)[0]
            for _ in range(3):
                if after_turn['state']['combat']['player_cur_hp']<before['state']['combat']['player_cur_hp']:
                    break
                end=next(a['index'] for a in after_turn['legal_actions'] if a['public']['kind']=='end_turn')
                after_turn=episode.step(end)[0]
            episode.env.bc.obtain_potion(episode.env.sts.Potion.FRUIT_JUICE)
            potion_view=episode._observe()
            potion=next(a['index'] for a in potion_view['legal_actions'] if a['public']['kind']=='drink_potion'
                        and 'fruit' in str(a['public'].get('potion','')).lower())
            after_juice=episode.step(potion)[0]
            return {"fixture":"Acquire 30 Hand of Greed copies before first combat; execute an ordinary lethal play",
                "seed":seed,"action":possible[0],"before":before,"after":after,
                "top_level_gold":after["state"]["gold"],"live_player_gold":after["state"]["combat"]["player"]["gold"],
                "encoded_global_gold":float(encoded.global_values[7]),"after_enemy_turn":after_turn,
                "after_fruit_juice":after_juice}
    raise RuntimeError("no bounded multi-enemy Hand of Greed witness")


def probe_rewards():
    seed=next(s for s in range(10,100) if world_split(s)=="train")
    episode=new_episode(seed,"ooc")
    episode.env.gc.obtain_relic(episode.env.sts.RelicId.PRAYER_WHEEL)
    episode.env.gc.cur_hp=episode.env.gc.max_hp=1000
    for _ in range(32):
        before=episode._observe()
        if str(before["state"]["screen_state"]).endswith("REWARDS"):
            cards=[a for a in before["legal_actions"] if a["public"]["kind"]=="take_card"]
            if len(cards)>=6:
                raw=episode.env.raw_actions()
                first=next(a for a in raw if episode.env.gc.public_action(a).get("kind")=="take_card")
                group=int(first.idx1())
                chosen=next(a for a in cards if a["description"]==first.describe(episode.env.gc))
                after=episode.step(chosen["index"])[0]
                return {"fixture":"Acquire Prayer Wheel and set HP/maxHP1000; native first-fight resolution",
                    "seed":seed,"before":before,"after":after,"selected_group_diagnostic":group,
                    "card_offers_before":len(cards),"card_offers_after":sum(a["public"]["kind"]=="take_card" for a in after["legal_actions"]),
                    "public_group_fields":[a["public"] for a in cards]}
        episode.step(before["legal_actions"][0]["index"])
    raise RuntimeError("no dual-card reward fixture")


def child(connection,name):
    try:
        connection.send({"status":"ok","result":globals()["probe_"+name]()})
    except Exception as error:
        connection.send({"status":"error","error":str(error),"traceback":traceback.format_exc()})
    finally:
        connection.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    results={}
    context=multiprocessing.get_context("spawn")
    for name in ("bottle","gold","rewards"):
        parent,remote=context.Pipe(False)
        process=context.Process(target=child,args=(remote,name)); process.start();remote.close()
        if parent.poll(20):
            results[name]=parent.recv()
        else:
            results[name]={"status":"timeout"}
        process.join(timeout=.5)
        if process.is_alive():
            process.kill();process.join()
        process.close();parent.close()
    atomic_json(args.output,results)
    print(json.dumps({name:{k:v for k,v in result.items() if k!="result"} for name,result in results.items()}))


if __name__=="__main__":
    main()
