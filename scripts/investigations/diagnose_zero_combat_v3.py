"""Compare compact legacy and expanded typed combat scores on a live trajectory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sts_policy import entities
from sts_policy.entity_model import EntityAgent
from sts_policy.combat_residual import load_policy
from sts_policy.curriculum import TaskSpec
from sts_policy.encoding import encode_observation, collate_observations
from sts_policy.legacy_bridge import project_legacy_menu
from sts_policy.workers import CurriculumWorker
from scripts.run_strategic_v3 import worker_settings, atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--zero", required=True)
    parser.add_argument("--seed", type=int, default=330001092)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    base = EntityAgent(args.base)
    zero, _ = load_policy(args.zero)
    zero.eval()
    differences, maxima = [], []
    steps = 0
    with CurriculumWorker(worker_settings(),timeout=10.) as worker:
        obs, info = worker.reset(TaskSpec(seed=args.seed,kind="run",max_act=3,horizon=4096))
        while obs["legal_actions"]:
            action = base.choose_action(obs)
            if obs["phase"] == "combat":
                projection = project_legacy_menu(obs)
                with torch.no_grad():
                    old_logits, _ = base.model.prior(*collate_observations([encode_observation(projection.observation)]))
                    logits, _ = zero(*entities.collate_public([obs]))
                expanded = old_logits[0, list(projection.legacy_positions)].numpy()
                current = logits[0].numpy()
                maxima.append(float(np.max(np.abs(expanded-current))))
                pick = int(current.argmax())
                candidate = int(obs["legal_actions"][pick]["index"])
                if candidate != action:
                    differences.append({"step":steps,"original_action":action,"candidate_action":candidate,
                        "actions":obs["legal_actions"],"original_logits":expanded.tolist(),"typed_logits":current.tolist(),
                        "observation":obs})
            obs, reward, terminated, truncated, info = worker.step(action)
            steps += 1
            if terminated or truncated:
                break
    record = {"seed":args.seed,"steps":steps,"differences":differences,"max_logit_delta":max(maxima,default=0),
              "termination":info.get("termination_reason")}
    atomic_json(args.output,record)
    print(json.dumps({k:record[k] for k in ("seed","steps","max_logit_delta","termination")}|{"different_choices":len(differences)}))


if __name__ == "__main__":
    main()
