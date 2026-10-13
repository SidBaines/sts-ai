"""Build a standalone, read-only comparison of three saved fight trajectories."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.compare_fight_formats import DEFAULTS, ROOT
from sts_ai import fight_lab as lab
from sts_ai.fight_analysis import same_start
from sts_ai.fight_comparison import ARM_NAMES, ARM_LABELS
from sts_ai.provenance import file_sha256
from sts_ai.rollout_view import to_view
from sts_ai.rollout_view_html import BOARD_CSS, render_decision_html


STATIC = ROOT / "src/sts_ai/interactive/static"


def load_trace(run: Path, state_id: str) -> dict:
    if Path(state_id).name != state_id:
        raise ValueError("state_id must be a single identifier")
    result = lab.read_json(run / f"{state_id}.result.json")
    path = run / f"{state_id}_r0.jsonl"
    meta = path.with_suffix(".meta.json")
    if (result.get("status") != "ok" or result.get("state_id") != state_id
            or file_sha256(path) != result["trace_sha256"]
            or file_sha256(meta) != result["meta_sha256"]):
        raise ValueError("saved trajectory identity or hash differs")
    metadata = lab.read_json(meta)
    if metadata["extra"]["fight_lab"]["state_id"] != state_id:
        raise ValueError("trajectory metadata refers to a different fight")
    records = lab.read_records(path)
    if not records:
        raise ValueError("trajectory has no saved decisions")
    return {"path": str(path.resolve()), "trace_sha256": file_sha256(path),
            "result": result, "settings": lab.read_json(run / "settings.json"),
            "records": records}


def script_json(value: object) -> str:
    # Model text is data, including any literal HTML/script delimiters.
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


def build(runs: dict[str, Path], state_id: str, output: Path) -> Path:
    arms = {arm: load_trace(runs[arm], state_id) for arm in ARM_NAMES}
    first = arms["old_format"]["records"][0]
    seed = arms["old_format"]["result"]["policy_seed"]
    for arm in ARM_NAMES:
        if not same_start(arms[arm]["records"][0], first):
            raise ValueError("trajectories do not start from the same public position")
        if arms[arm]["result"]["policy_seed"] != seed:
            raise ValueError("trajectories have different sampling seeds")
        arms[arm]["label"] = ARM_LABELS[arm]
        arms[arm]["frames"] = [
            {"turn": r["state"].get("combat", {}).get("turn"),
             "board_html": render_decision_html(to_view(r))}
            for r in arms[arm]["records"]]
    enemies = Counter(e["name"] for e in first["state"]["combat"]["enemies"])
    title = " + ".join(f"{name.replace('_', ' ').title()} ×{n}" for name, n in enemies.items())
    data = {"state_id": state_id, "title": title, "world_seed": first["world_seed"],
            "floor": first["state"]["floor"], "arms": arms}
    output.mkdir(parents=True, exist_ok=True)
    template = (STATIC / "fight_compare.html").read_text()
    javascript = (STATIC / "fight_compare.js").read_text()
    page = template.replace("__BOARD_CSS__", BOARD_CSS).replace("__SCRIPT__", javascript)
    page = page.replace("__DATA__", script_json(data))
    (output / "index.html").write_text(page)
    (output / "trajectories.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    (output / "README.md").write_text(
        f"# {title} · three saved trajectories\n\nOpen `index.html` directly in a browser. "
        "The viewer is self-contained, works offline, and cannot start inference.\n\n"
        "Use the shared turn selector to compare the start of the same recorded turn. "
        "Each column has its own decision controls. Opening is recorded turn 0; "
        "decision indices are not synchronized across policies. "
        "The final outcome is a separate position after the last saved decision.\n\n"
        "The Observation tab contains the saved state text and legal menu; "
        "Thinking contains extracted model thinking, and Response contains the raw output. "
        "These are saved records, not a new simulation.\n")
    return output / "index.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-id", default="seed_4050_r0_d95")
    parser.add_argument("--output", type=Path)
    for arm in ARM_NAMES:
        parser.add_argument("--" + arm.replace("_", "-"), type=Path, default=DEFAULTS[arm])
    args = parser.parse_args()
    output = args.output or ROOT / "outputs" / f"fight_comparison_{args.state_id}"
    print(build({a: getattr(args, a).resolve() for a in ARM_NAMES}, args.state_id, output.resolve()))


if __name__ == "__main__":
    main()
