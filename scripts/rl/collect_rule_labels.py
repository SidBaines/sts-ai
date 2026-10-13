#!/usr/bin/env python3
"""Distillation records for strategy rules: rest site ("smith at HP >= threshold,
else rest") and optionally map ("path into an offered elite at HP >= threshold").

Plays greedy games with a checkpoint, following the rule at rest sites (so the
states visited are the rule-policy's), and writes:
  - type "rest": every rest-site menu offering both rest and smith, with the
    rule's choice as a hard label (label_index = native action index);
  - type "elite" (with --elite-above): every map menu offering an elite, labelled
    with the policy's best elite node at HP >= threshold, else the policy's own pick;
  - type "rehearsal": a sample of other OOC decisions, for the KL anchor.
Format matches sts_rl.distill (shard_*.jsonl). TRAIN worlds only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import sys
import threading

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_policy.model_agent import collate_public, forward_policy  # noqa: E402
from sts_rl.common import hp_fraction, load_policy, make_workers, train_tasks  # noqa: E402

from sts_rl.recording import begin_run
from sts_env.env import SimulatorError

LOCK = threading.Lock()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--world-start", type=int, default=8_500_000_000)
    p.add_argument("--smith-above", type=float, default=0.85)
    p.add_argument("--elite-above", type=float, default=None)
    p.add_argument("--rehearsal-rate", type=float, default=0.1)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if any(Path(args.output).glob('shard_*.jsonl')) or (Path(args.output)/'summary.json').exists():
        p.error('Output already contains results; choose a fresh output')
    begin_run(args.output, vars(args), args.checkpoint)
    torch.set_num_threads(1)
    model, _ = load_policy(args.checkpoint)
    model.eval()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tasks = train_tasks(args.world_start, args.n)

    def play(slot: int) -> dict[str, int]:
        rng = random.Random(slot)
        worker = make_workers(1, dead_action_filter=True)[0]
        counts = {"games": 0, "rest": 0, "smith_labels": 0, "elite": 0, "elite_labels": 0, "rehearsal": 0,
                  "faults": 0}
        with (out / f"shard_{slot:03d}.jsonl").open("w") as sink:
            for task in tasks[slot::args.threads]:
                rows = []
                try:
                    obs, info = worker.submit("reset", task).result()
                    term = trunc = False
                    while not (term or trunc):
                        legal = [a["description"] for a in obs["legal_actions"]]
                        with LOCK, torch.no_grad():
                            logits, _, _ = forward_policy(model, collate_public([obs]))
                        pos = int(logits[0][: len(legal)].argmax())
                        if obs.get("phase") != "combat" and len(legal) > 1:
                            rest = [i for i, d in enumerate(legal) if d.startswith("rest")]
                            smith = [i for i, d in enumerate(legal) if d.startswith("smith")]
                            if rest and smith:
                                fraction = hp_fraction(obs)
                                pos = smith[0] if fraction is not None and fraction >= args.smith_above else rest[0]
                                rows.append({"type": "rest", "seed": task.seed, "obs": obs,
                                             "label_index": int(obs["legal_actions"][pos]["index"])})
                                counts["rest"] += 1
                                counts["smith_labels"] += pos == smith[0]
                            elif args.elite_above is not None and any("room=ELITE" in d for d in legal):
                                fraction = hp_fraction(obs)
                                if fraction is not None and fraction >= args.elite_above:
                                    elites = [i for i, d in enumerate(legal) if "room=ELITE" in d]
                                    pos = max(elites, key=lambda i: float(logits[0][i]))
                                    counts["elite_labels"] += 1
                                rows.append({"type": "elite", "seed": task.seed, "obs": obs,
                                             "label_index": int(obs["legal_actions"][pos]["index"])})
                                counts["elite"] += 1
                            elif rng.random() < args.rehearsal_rate:
                                rows.append({"type": "rehearsal", "seed": task.seed, "obs": obs})
                                counts["rehearsal"] += 1
                        obs, _, term, trunc, info = worker.submit("step", int(obs["legal_actions"][pos]["index"])).result()
                    for row in rows:
                        sink.write(json.dumps(row) + "\n")
                    counts["games"] += 1
                except Exception as exc:
                    if not isinstance(exc, SimulatorError):
                        raise  # simulator fault: drop the whole game
                    counts["faults"] += 1
                    worker.close()
                    worker = make_workers(1, dead_action_filter=True)[0]
        worker.close()
        return counts

    with ThreadPoolExecutor(args.threads) as ex:
        parts = list(ex.map(play, range(args.threads)))
    total = {k: sum(c[k] for c in parts) for k in parts[0]}
    print(json.dumps(total))
    (out / "summary.json").write_text(json.dumps({**vars(args), **total}))


if __name__ == "__main__":
    main()
