#!/usr/bin/env python3
"""How often each combat head chooses the search teacher's move, on a recorded teacher corpus.

    teacher_agreement.py CORPUS_DIR --head NAME=CHECKPOINT [--head ...] --out rows.jsonl

CORPUS_DIR holds fresh_collect --record-search shards (shard_*/worlds/*.json.gz). Every combat label with
search details is scored: the teacher's move under each vote rule (plurality, first legal index on ties,
as the collector does) against each checkpoint's greedy move. Neither head need have trained on the corpus.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sts_rl.fresh_statistics import clustered_mean  # noqa: E402
from sts_rl.teacher_votes import votes_from_details  # noqa: E402

RULES = ("pooled_mean", "best_line", "most_visited")


def teacher_move(votes, legal):
    if not votes:
        return None
    top = max(votes.values())
    return next(int(a["index"]) for a in legal if votes.get(int(a["index"])) == top)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("corpus")
    p.add_argument("--head", action="append", required=True, help="NAME=CHECKPOINT")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from sts_rl.batched_runner import BatchedPolicy
    from sts_rl.controlled_policy import policy_factory
    heads = {k: BatchedPolicy(policy_factory(v)) for k, v in (h.split("=", 1) for h in a.head)}
    rows = []
    for file in sorted(Path(a.corpus).glob("shard_*/worlds/*.json.gz")):
        for s in json.load(gzip.open(file))["labels"]:
            obs = s.get("observation")
            if not s.get("search_details") or not obs or obs.get("phase") != "combat" or len(obs["legal_actions"]) < 2:
                continue
            legal = obs["legal_actions"]
            desc = {int(x["index"]): x["description"] for x in legal}
            row = {"world": s["world_seed"], "group": s["group"], "source_action": s.get("source_action"),
                   "drink_legal": any(d.startswith("drink potion") for d in desc.values())}
            for rule in RULES:
                row[rule] = teacher_move(votes_from_details(s["search_details"], rule), legal)
            row["obs"] = obs
            rows.append(row)
    print(f"{len(rows)} combat positions", flush=True)
    for name, policy in heads.items():
        for start in range(0, len(rows), 512):
            batch = rows[start:start + 512]
            for r, action in zip(batch, policy.choose_actions([r["obs"] for r in batch])):
                r[name] = action
    with open(a.out, "w") as out:
        for r in rows:
            obs = r.pop("obs")
            desc = {int(x["index"]): x["description"] for x in obs["legal_actions"]}
            r["drinks"] = {k: desc.get(r[k], "").startswith("drink potion") for k in (*RULES, *heads) if r.get(k) is not None}
            out.write(json.dumps(r) + "\n")
    names = list(heads)
    groups = sorted({r["group"] for r in rows}) + ["all", "drink legal", "teacher drinks (pooled_mean)"]
    for rule in RULES:
        print(f"\nAgreement with the teacher's {rule} move")
        print(f"  {'group':30s} {'n':>7s} " + " ".join(f"{n:>10s}" for n in names) + "   difference (world-clustered 95% CI)")
        for g in groups:
            sub = [r for r in rows if r[rule] is not None and (
                g == "all" or r["group"] == g or (g == "drink legal" and r["drink_legal"])
                or (g.startswith("teacher drinks") and r["drinks"].get("pooled_mean")))]
            if not sub:
                continue
            rates = [sum(r[n] == r[rule] for r in sub) / len(sub) for n in names]
            line = f"  {g:30s} {len(sub):7d} " + " ".join(f"{100*x:9.1f}%" for x in rates)
            if len(names) == 2:
                clusters = defaultdict(list)
                for r in sub:
                    clusters[r["world"]].append(float(r[names[1]] == r[rule]) - float(r[names[0]] == r[rule]))
                m = clustered_mean(clusters, resamples=2000)
                line += f"   {100*m['point']:+.1f} [{100*m['ci95'][0]:+.1f}, {100*m['ci95'][1]:+.1f}]"
            print(line)
    print("\nDrink rate where a drink is legal (BOSS / ELITE / MONSTER / all)")
    for k in (*RULES, *names):
        parts = []
        for g in ("BOSS", "ELITE", "MONSTER", ""):
            sub = [r for r in rows if r["drink_legal"] and r["group"].startswith(g) and r.get(k) is not None]
            parts.append(f"{100*sum(r['drinks'].get(k, False) for r in sub)/max(1,len(sub)):5.1f}% (n={len(sub)})")
        print(f"  {k:14s} " + "  ".join(parts))
    if "old" in heads:
        mism = sum(str(r["old"]) != str(r["source_action"]) for r in rows if r["source_action"] is not None)
        print(f"\nold head differs from the recorded source action at {mism} of {len(rows)} positions")


if __name__ == "__main__":
    main()
