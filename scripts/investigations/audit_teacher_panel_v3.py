"""Teacher randomness and hidden draw-order sensitivity on identical public inputs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from sts_policy import entities
from sts_policy.combat_residual import equivalent_positions
from sts_policy.curriculum import world_split
from sts_policy.training import file_identity
from scripts.run_strategic_v3 import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    groups={}
    with args.teacher.open() as source:
        for line in source:
            record=json.loads(line)
            if world_split(record["world_seed"])!="train":
                raise ValueError("teacher stability diagnosis must remain TRAIN-only")
            obs=record["observation"]; act=str(obs["state"]["act"])
            row=entities.encode_observation(obs)
            positions={int(a):i for i,a in enumerate(row.action_indices)}
            panel=record["teacher_panels"]["1000"]["results"]
            group=groups.setdefault(act,Counter()); group["states"]+=1
            baseline=positions[panel[0]["index"]]
            for name,results in (("search_rng",panel[:3]),("hidden_draw",panel[3:])):
                signatures=[]; all_baseline=True
                for result in results:
                    if result["status"]!="ok":
                        group[name+"_abstentions"]+=1
                        continue
                    position=positions[result["index"]]
                    signatures.append(equivalent_positions(row,position).tobytes())
                    group[name+"_labels"]+=1
                    group[name+"_matches_baseline"]+=bool(equivalent_positions(row,baseline)[position])
                    group[name+"_winning_sequence_labels"]+=bool(result["winning_sequence_found"])
                    all_baseline &= bool(equivalent_positions(row,baseline)[position])
                    visited=[edge for edge in result["root_edges"] if edge["visits"]>0]
                    if visited:
                        best=max(visited,key=lambda edge:edge["mean_evaluation"])
                        group[name+"_root_comparisons"]+=1
                        group[name+"_selected_differs_from_best_visited_mean"]+=not bool(
                            equivalent_positions(row,position)[positions[best["index"]]])
                if len(signatures)==3:
                    group[name+"_complete_states"]+=1
                    group[name+"_unanimous_states"]+=len(set(signatures))==1
                    group[name+"_all_match_baseline_states"]+=all_baseline
    result={"teacher_sha256":file_identity(args.teacher),"by_act":{a:dict(g) for a,g in groups.items()},
        "note":"Interventions operate on cloned private states. Draw permutations preserve exposed unordered contents; a full public history could constrain the admissible permutations. This measures label sensitivity, not optimality or calibrated action value."}
    atomic_json(args.output,result)
    print(json.dumps(result))


if __name__=="__main__":
    main()
