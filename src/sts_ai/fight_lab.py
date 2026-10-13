"""Frozen partial-rollout collections and fail-closed single-fight replay.

DecisionRecord remains the trace contract; all lab identities live in sidecars.
No ML, web framework, or simulator imports are needed to browse/filter a collection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from sts_ai.provenance import file_sha256


MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
MODEL_REVISION = "0d77464eeb233a2da68ebf9d7dc4edaac7db956d"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def trace_digest(records: list[dict[str, Any]]) -> str:
    """Compare trajectories including model text, excluding only elapsed latency."""
    clean = []
    for record in records:
        row = dict(record)
        row["agent"] = {k: v for k, v in record["agent"].items() if k != "latency_s"}
        clean.append(row)
    return digest(clean)


def public_digest(record: dict[str, Any]) -> str:
    return digest({k: record[k] for k in ("phase", "state", "state_text", "legal_actions")})


def hp(state: dict[str, Any]) -> tuple[int, int]:
    combat = state.get("combat") or {}
    return (int(combat.get("player_cur_hp", state.get("cur_hp", 0))),
            int(combat.get("player_max_hp", state.get("max_hp", 0))))


@dataclass(frozen=True)
class StateFilter:
    phase: str = "combat"
    enemy: str = ""
    battle_start: bool = False
    hp_below: float | None = None
    act: int | None = None
    floor_min: int | None = None
    floor_max: int | None = None
    turn: int | None = None
    split: str = ""

    def __post_init__(self) -> None:
        if self.phase not in ("", "combat", "out_of_combat"):
            raise ValueError("unknown phase")
        if self.split not in ("", "train", "eval"):
            raise ValueError("unknown split")
        if self.hp_below is not None and not 0 <= self.hp_below <= 1:
            raise ValueError("hp_below must be a fraction between 0 and 1")
        if self.floor_min is not None and self.floor_max is not None and self.floor_min > self.floor_max:
            raise ValueError("floor_min must be <= floor_max")

    def matches(self, row: dict[str, Any]) -> bool:
        enemy = self.enemy.casefold().replace(" ", "_")
        return all((
            not self.phase or row["phase"] == self.phase,
            not enemy or any(enemy in name.casefold().replace(" ", "_") for name in row["enemies"]),
            not self.battle_start or row["battle_start"],
            self.hp_below is None or (row["hp_fraction"] is not None and row["hp_fraction"] < self.hp_below),
            self.act is None or row["act"] == self.act,
            self.floor_min is None or row["floor"] >= self.floor_min,
            self.floor_max is None or row["floor"] <= self.floor_max,
            self.turn is None or row["turn"] == self.turn,
            not self.split or row["split"] == self.split,
        ))


def index_records(records: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    rows = []
    previous_fight = None
    fight_start = None
    for index, record in enumerate(records):
        state = record["state"]
        combat = state.get("combat") or {}
        in_combat = record["phase"] == "combat"
        fight = (state.get("act"), state.get("floor")) if in_combat else None
        start = in_combat and fight != previous_fight
        if start:
            fight_start = index
        cur, maximum = hp(state)
        seed = int(record["world_seed"])
        rows.append({
            "state_id": f"{Path(source).stem}_d{index}", "source": source,
            "index": index, "public_sha256": public_digest(record),
            "world_seed": seed, "rollout_index": record.get("rollout_index", 0),
            "policy_seed": record.get("policy_seed"),
            "split": "eval" if seed % 4 == 0 else "train",
            "phase": record["phase"], "act": int(state.get("act", 0)),
            "floor": int(state.get("floor", 0)), "turn": combat.get("turn"),
            "enemies": sorted(e["name"] for e in combat.get("enemies", []) if e.get("alive", True)),
            "hp": cur, "max_hp": maximum,
            "hp_fraction": cur / maximum if maximum > 0 else None,
            "battle_start": start, "fight_start_index": fight_start if in_combat else None,
        })
        previous_fight = fight
    return rows


def selection(collection: dict[str, Any], filters: StateFilter,
              state_ids: list[str] | None = None) -> dict[str, Any]:
    rows = [row for row in collection["states"] if filters.matches(row)]
    if state_ids is not None:
        if not state_ids or len(set(state_ids)) != len(state_ids):
            raise ValueError("provide unique state IDs")
        if set(state_ids) - {row["state_id"] for row in rows}:
            raise ValueError("selected state is unknown or does not match the filters")
        rows = [row for row in rows if row["state_id"] in state_ids]
    payload = {"kind": "fight_lab_selection", "version": 1,
               "collection_sha256": digest(collection), "filters": asdict(filters),
               "states": rows}
    if state_ids is not None:
        payload["state_ids"] = [row["state_id"] for row in rows]
    return {**payload, "selection_sha256": digest(payload)}


def validate_selection(collection: dict[str, Any], selected: dict[str, Any]) -> None:
    payload = {k: v for k, v in selected.items() if k != "selection_sha256"}
    if selected.get("selection_sha256") != digest(payload):
        raise ValueError("selection content hash differs")
    if selected.get("collection_sha256") != digest(collection):
        raise ValueError("selection belongs to a different collection")
    expected = selection(collection, StateFilter(**selected["filters"]), selected.get("state_ids"))
    if selected != expected:
        raise ValueError("selection rows differ from the recorded filter")


def source_records(root: Path, collection: dict[str, Any], source: str) -> list[dict[str, Any]]:
    entry = next(s for s in collection["sources"] if s["path"] == source)
    path = (root / source).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("source path escapes collection")
    if file_sha256(path) != entry["sha256"]:
        raise ValueError(f"source trace changed: {source}")
    return read_records(path)


def replay_to(env: Any, records: list[dict[str, Any]], index: int) -> None:
    """Match production observation/getter order, verifying every public prefix."""
    from sts_ai.rollout import prepare_decision

    if not 0 <= index < len(records):
        raise ValueError("state index is outside the source trace")
    for offset, expected in enumerate(records[:index + 1]):
        status, view = prepare_decision(env)
        if status != "ok" or view is None:
            raise ValueError(f"replay ended at decision {offset}: {status}")
        actual = {**view, "legal_actions": view["legal_action_dicts"]}
        if public_digest(actual) != public_digest(expected):
            raise ValueError(f"replay diverged at decision {offset}")
        if offset == index:
            return
        if not expected.get("action_executed", True):
            raise ValueError(f"unexecuted action in prefix at {offset}")
        selected = expected["selected_action"]
        if selected not in view["legal_action_dicts"]:
            raise ValueError(f"source action is not legal at {offset}")
        actual_selected = env.step(selected["index"])
        if env.action_dict(actual_selected) != selected or env.summary() != expected["after_state"]:
            raise ValueError(f"replay after-state diverged at decision {offset}")


class SingleFightTask:
    task_id = "single_fight"

    def completion_reason(self, summary: dict[str, Any]) -> str | None:
        if hp(summary)[0] <= 0:
            return "player_loss"
        if not isinstance(summary.get("combat"), dict):
            return "task_complete"
        return None

    def metrics_from_episode(self, decisions: list[Any], terminal_state: dict[str, Any],
                             stopped_reason: str, window: dict[str, Any]) -> dict[str, Any]:
        entry = window["entry_hp"]
        exit_hp = hp(terminal_state)[0]
        won = stopped_reason == "task_complete" and exit_hp > 0
        complete = stopped_reason in ("task_complete", "player_loss")
        return {"entry_hp": entry, "exit_hp": exit_hp, "hp_loss": max(0, entry - exit_hp),
                "won": won, "completed": complete,
                "label": "win" if won else "loss" if complete else "incomplete",
                "reward": float(won), "n_decisions": len(decisions)}


def collect_worker(config: dict[str, Any], seed: int, output: Path) -> None:
    from sts_ai.agents import SimpleHeuristicAgent
    from sts_ai.lightspeed import LightspeedHybridEnv
    from sts_ai.rollout import run_rollout

    env = LightspeedHybridEnv(world_seed=seed, **config["env"])
    run_rollout(env, SimpleHeuristicAgent(), max_decisions=config["max_decisions"],
                output_path=output, run_meta={"extra": {"fight_lab": config}})


def evaluate_worker(root: Path, selected: dict[str, Any], state_id: str,
                    settings: dict[str, Any], output: Path, *, agent: Any = None) -> None:
    import importlib.metadata
    from sts_ai.agent_factory import build_agent
    from sts_ai.lightspeed import LightspeedHybridEnv
    from sts_ai.local_tasks.runner import run_local_task_episode
    from sts_ai.provenance import adapter_provenance, competence_interface_provenance

    collection = read_json(root / "collection.json")
    validate_selection(collection, selected)
    row = next(row for row in selected["states"] if row["state_id"] == state_id)
    if row["phase"] != "combat":
        raise ValueError("single-fight evaluation requires a combat state")
    records = source_records(root, collection, row["source"])
    env = LightspeedHybridEnv(world_seed=row["world_seed"], **collection["config"]["env"])
    interface = competence_interface_provenance(env, object())
    for key, value in collection["replay_interface"].items():
        if interface.get(key) != value:
            raise ValueError(f"collection interface changed: {key}")
    replay_to(env, records, row["index"])
    source_observation = env.combat_observation
    policy_observation = settings.get("combat_observation", source_observation)
    if policy_observation != source_observation:
        if source_observation != "combat_public_v3" or policy_observation not in ("combat_public_v4", "combat_public_v5"):
            raise ValueError("unsupported source-to-policy observation migration")
        from sts_ai.combat_public_v4 import CombatPublicV4Env
        # Check every original prefix before changing only the policy presentation.
        if policy_observation == "combat_public_v5":
            from sts_ai.combat_public_v5 import CombatPublicV5Env
            env = CombatPublicV5Env(env)
        else:
            env = CombatPublicV4Env(env)
    model = settings.get("model")
    if agent is None and settings["agent"] == "mlx":
        from huggingface_hub import snapshot_download
        model = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    if agent is None:
        agent = build_agent(settings["agent"], model=model or MODEL_ID, temperature=settings["temperature"],
                        top_p=settings.get("top_p", 1.0), top_k=settings.get("top_k", -1),
                        max_tokens=settings["max_tokens"], thinking=settings["thinking"],
                        max_retries=0, adapter_path=settings.get("adapter_path"),
                        output_contract="action_text")
    window = {"window_id": state_id, "source_stem": Path(row["source"]).stem,
              "split": row["split"], "entry_hp": row["hp"]}
    run_local_task_episode(task=SingleFightTask(), env=env, agent=agent, window=window,
        max_decisions=settings["max_decisions"], output_path=output,
        rollout_index=settings["rollout_index"], policy_seed=settings.get("policy_seed"),
        run_meta={"extra": {"fight_lab": {
            "selection_sha256": selected["selection_sha256"], "state_id": state_id,
            "start_public_sha256": row["public_sha256"],
            "source_combat_observation": source_observation,
            "policy_combat_observation": policy_observation,
            "model_id": MODEL_ID if settings["agent"] == "mlx" else None,
            "model_revision": MODEL_REVISION if settings["agent"] == "mlx" else None,
            "runtime_versions": {name: importlib.metadata.version(name) for name in
                                 ("mlx", "mlx-lm", "transformers")}
                                if settings["agent"] == "mlx" else {},
            "adapter": adapter_provenance(settings.get("adapter_path")),
            "settings": settings}}})
