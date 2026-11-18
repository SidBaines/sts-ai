"""Profile a fixed champion workload in one collector or one evaluator process."""

import argparse
import cProfile
import hashlib
import json
import pstats
import time
from pathlib import Path

from sts_rl.batched_runner import configure_cpu, run_in_process
from sts_rl.controlled_policy import policy_factory


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--games", type=int, default=32)
    p.add_argument("--slots", type=int, default=16)
    p.add_argument("--world", type=int, default=91001000000)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--collect", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch

    configure_cpu()
    policy = policy_factory(args.checkpoint)
    seeds = range(args.world, args.world + args.games)

    def evaluate():
        rows = run_in_process(seeds, policy, args.slots)
        return dict(
            games=len(rows),
            decisions=sum(len(r["action_prefix"]) for r in rows),
            faults=sum(r["status"] != "completed" for r in rows),
            hash=digest(
                [
                    {
                        k: r[k]
                        for k in (
                            "world_seed",
                            "action_hash",
                            "terminal_hash",
                            "bosses",
                            "behavior",
                            "status",
                            "win",
                            "floor",
                        )
                    }
                    for r in sorted(rows, key=lambda r: r["world_seed"])
                ]
            ),
        )

    if args.collect:
        from sts_policy.rollout_training import PPOConfig
        from sts_rl.common import make_workers, phase_forward, train_tasks
        from sts_rl.ooc_trainer import collect_ooc_batch

        print("loading collector", flush=True)
        workers = make_workers(
            args.slots,
            in_process=True,
            dead_action_filter=True,
            hide_ooc_potions=True,
            progress_floor_weight=0.05,
        )
        tasks = train_tasks(args.world, args.games)

        def collect():
            torch.manual_seed(1701)
            return collect_ooc_batch(
                policy.champion,
                workers,
                tasks,
                forward=phase_forward(combat="greedy"),
                ppo=PPOConfig(gae_lambda=1.0),
                credit="macro",
                gae_lambda=1.0,
                chunk=args.games,
            )

        def collect_summary(batch):
            eps = batch["game_episodes"]
            return dict(
                games=len(eps),
                kept=sum(map(len, batch["kept"])),
                faults=len(batch["fault_records"]),
                hash=digest(eps),
            )

        def run():
            return collect_summary(collect())
    else:
        run = evaluate
    report = []
    for rep in range(args.repeat):
        prof = cProfile.Profile() if args.profile else None
        t = time.perf_counter()
        result = prof.runcall(run) if prof else run()
        result.update(seconds=time.perf_counter() - t, repeat=rep)
        if prof:
            prof.dump_stats(str(args.output.with_suffix(".pstats")))
            pstats.Stats(prof).strip_dirs().sort_stats("tottime").print_stats(24)
        print(json.dumps(result), flush=True)
        report.append(result)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
