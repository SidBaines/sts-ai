#!/usr/bin/env python3
"""Keep only take_card labels at steps where the champion itself was choosing a card.

    filter_decisive_labels.py CHECKPOINT IN_DIR OUT_DIR

A reward screen offering Demon Form is visited several times (gold, potion, card);
collect_rule_labels.py labels every visit. Here a take_card row is kept only if
the champion's own argmax at that menu is a card take, a card skip, or proceeding
past the screen, so the labels teach which card to take, not when to take it.
Rehearsal rows are kept unchanged. Writes OUT_DIR/shard_000.jsonl and a summary.
"""
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sts_policy.model_agent import collate_public, forward_policy  # noqa: E402
from sts_rl.common import load_policy  # noqa: E402

DECIDE = ("take card ", "skip card reward", "skip rewards / proceed")


def main() -> None:
    checkpoint, source, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    if out.exists() and any(out.iterdir()):
        raise SystemExit("output exists; choose a fresh directory")
    out.mkdir(parents=True)
    model, _ = load_policy(checkpoint)
    model.eval()
    kept = dropped = rehearsal = 0
    with (out / "shard_000.jsonl").open("w") as sink:
        for shard in sorted(source.glob("shard_*.jsonl")):
            for line in shard.read_text().splitlines():
                row = json.loads(line)
                if row["type"] == "take_card":
                    obs = row["obs"]
                    with torch.no_grad():
                        logits, _, _ = forward_policy(model, collate_public([obs]))
                    legal = [a["description"] for a in obs["legal_actions"]]
                    own = legal[int(logits[0][: len(legal)].argmax())]
                    if not own.startswith(DECIDE):
                        dropped += 1
                        continue
                    kept += 1
                else:
                    rehearsal += 1
                sink.write(json.dumps(row) + "\n")
    summary = {"checkpoint": checkpoint, "source": str(source), "take_card_kept": kept,
               "take_card_dropped": dropped, "rehearsal": rehearsal}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
