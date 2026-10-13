"""Capture the first reference disagreement under the actual eight-worker layout."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import torch

from sts_policy.entity_model import EntityAgent
from sts_policy.combat_residual import load_policy
from sts_policy.strategic_collection import collect
from sts_policy.workers import CurriculumWorker
from scripts.run_strategic_v3 import worker_settings, atomic_json, matched_tasks, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--zero", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    agent = EntityAgent(args.base)
    model, _ = load_policy(args.zero)
    differences = []
    def observer(observations, seeds, steps, choices):
        reference = agent.choose_actions(observations)
        for obs, seed, step, actual, old in zip(observations, seeds, steps, choices, reference):
            if actual != old:
                differences.append({"seed":seed,"step":step,"candidate_action":actual,"reference_action":old,
                    "batch_phases":[o["phase"] for o in observations],
                    "batch_menu_sizes":[len(o["legal_actions"]) for o in observations],"observation":obs})
    tasks = matched_tasks(bank=None,kind="run",split="dev",episodes=192,world_start=330000000,
                         seed=0,horizon=4096,max_act=3)
    with ExitStack() as stack:
        workers = [stack.enter_context(CurriculumWorker(worker_settings(),timeout=10.)) for _ in range(8)]
        result = collect(model,workers,tasks,deterministic=True,retain_transitions=False,frozen_combat=False,
                         max_seconds=300,decision_observer=observer)
    result.pop("transitions")
    atomic_json(args.output,{"differences":differences,"result":result,"summary":summarize(result)})
    print(json.dumps({"differences":len(differences),"details":[{k:r[k] for k in ('seed','step','candidate_action','reference_action','batch_phases','batch_menu_sizes')} for r in differences]}))


if __name__ == "__main__":
    main()
