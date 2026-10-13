"""Compare same-state native budgets and executed public actions on TRAIN."""
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
    groups={}; coverage=Counter()
    for line in args.teacher.open():
        record=json.loads(line)
        if world_split(record["world_seed"])!="train":
            raise ValueError("teacher diagnosis must remain TRAIN-only")
        obs=record["observation"]; act=str(obs["state"]["act"])
        row=entities.encode_observation(obs); coverage.update(row.coverage)
        positions={int(a):i for i,a in enumerate(row.action_indices)}
        executed=positions[record["executed_index"]]
        budget200=positions[record["teacher_by_budget"]["200"]]
        budget1000=positions[record["teacher_by_budget"]["1000"]]
        group=groups.setdefault(act,Counter())
        group["rows"]+=1
        group["exact_budget_agreement"]+=budget200==budget1000
        group["equivalent_budget_agreement"]+=bool(equivalent_positions(row,budget200)[budget1000])
        for budget,pick in ((200,budget200),(1000,budget1000)):
            group[f"executed_exact_{budget}"]+=executed==pick
            group[f"executed_equivalent_{budget}"]+=bool(equivalent_positions(row,pick)[executed])
            group[f"teacher_{budget}_kind:{obs['legal_actions'][pick]['public']['kind']}"]+=1
    result={"teacher_sha256":file_identity(args.teacher),"by_act":{a:dict(g) for a,g in groups.items()},
        "coverage":dict(coverage),"note":"Agreement does not establish teacher optimality; equivalence is model indistinguishability."}
    atomic_json(args.output,result)
    print(json.dumps(result))


if __name__=="__main__":
    main()
