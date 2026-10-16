"""Model-free, outcome-blind panel selection and immutable grid manifests.

Ranking reads only group/world/start keys. Replay and bank identities bind the
chosen states afterwards; they never enter either ranking hash.
"""
from collections import defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path

from sts_policy.curriculum import public_hash
from sts_rl.fresh_study import PROTOCOL
from sts_rl.recording import file_identity

SELECTION_PROTOCOL = "one_start_per_world_group_hash_v1"
SAMPLING_SPEC = {
    "namespace": "fresh_panel_one_start_per_world_group_hash_v1",
    "hash": "sha256_canonical_json_sort_keys_compact_ascii",
    "world_rank_keys": ["namespace", "stage=world", "group", "world_seed"],
    "start_rank_keys": ["namespace", "stage=start", "group", "world_seed", "start_id"],
    "tie_break": "ascending_integer_world_seed_then_start_id",
    "unit": "one_start_per_source_world_per_target_group",
    "weighting": "equal_source_world_within_group",
    "uncertainty_cluster": "source_world_all_salts",
}
DEFAULT_TARGETS = {"BOSS1": 100, "BOSS2": 300, "BOSS3": 100,
                   "ELITE2": 150, "ELITE3": 100, "MONSTER2": 100,
                   "MONSTER3": 100, "EVENT1": 52, "EVENT2": 4}
KNOWN_GROUPS = frozenset(DEFAULT_TARGETS) | {"MONSTER1", "ELITE1", "EVENT3"}
HP_DEFINITION = {
    "id": "salted_entry_minus_pre_end_of_fight_relic_healing_v1",
    "entry_hp": "player_hp_from_initial_salted_public_combat_observation",
    "source_entry_hp": "bank_entry_hp_trace_only",
    "hp_after": "battle_exit_player_cur_hp_before_end_of_fight_relic_healing",
    "death_hp": 0,
    "hp_lost": "entry_hp_minus_hp_after; death_uses_zero_hp_after",
    "bank_baseline_hp": "post_healing; not_expected_to_match_panel_hp",
}


def _rank(stage, group, world, start=None):
    key = {"namespace": SAMPLING_SPEC["namespace"], "stage": stage,
           "group": group, "world_seed": world}
    if start is not None:
        key["start_id"] = start
    return public_hash(key)


def _start_identity(row):
    # Outcomes, baseline HP and model scores are deliberately excluded. The
    # validated shard identities bind the full original rows, including source.
    fields = ("world_seed", "start_id", "group", "public_hash", "first_combat_hash")
    replay = ("prefix", "task", "entering_action", "entering_description", "expected_provenance")
    return {**{key: row[key] for key in fields},
            "provenance_hash": public_hash(row["expected_provenance"]),
            "replay_hash": public_hash({key: row[key] for key in replay})}


def canonical_grid_hash(frozen):
    """Hash the entire manifest except this one derived field (no self-hash)."""
    return public_hash({key: value for key, value in frozen.items() if key != "grid_hash"})


def freeze_grid(rows, targets, resamples, *, bank_identity=None):
    """Choose exact world counts where available; never expand to other fights.

    Missing eligible worlds remain an explicit coverage shortfall. Each chosen
    start retains the unsalted control and every declared salted repeat.
    """
    if (isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1 or
            not isinstance(targets, dict) or not targets or
            any(not isinstance(g, str) or not g or isinstance(n, bool) or
                not isinstance(n, int) or n < 1 for g, n in targets.items())):
        raise ValueError("Require positive integer group WORLD targets and resamples")
    targets = dict(sorted(targets.items()))
    groups, seen = defaultdict(lambda: defaultdict(list)), set()
    for row in rows:
        world, start, group = row["world_seed"], row["start_id"], row["group"]
        if (isinstance(world, bool) or not isinstance(world, int) or
                isinstance(start, bool) or not isinstance(start, int) or start < 0 or
                not isinstance(group, str) or not group):
            raise ValueError("Invalid bank group/world/start key")
        # A start is one fight and cannot be relabelled into a second group.
        if (world, start) in seen:
            raise ValueError("Duplicate bank world/start key")
        seen.add((world, start))
        groups[group][world].append(row)
    selected, coverage = [], {}
    for group, count in targets.items():
        worlds = groups[group]
        ranked = sorted(worlds, key=lambda w: (_rank("world", group, w), w))
        coverage[group] = min(count, len(ranked))
        for world in ranked[:count]:
            selected.append(min(worlds[world], key=lambda r:
                (_rank("start", group, world, r["start_id"]), r["start_id"])))
    selected.sort(key=lambda r: (r["world_seed"], r["start_id"], r["group"]))
    eligible = {g: len(groups[g]) for g in sorted(KNOWN_GROUPS | set(groups) | set(targets))}
    shortfall = {g: n - coverage[g] for g, n in targets.items() if coverage[g] < n}
    salts = [-1, *range(resamples)]
    frozen = {"protocol": PROTOCOL, "selection_protocol": SELECTION_PROTOCOL,
              "sampling_spec": deepcopy(SAMPLING_SPEC), "bank_identity": deepcopy(bank_identity),
              "targets": targets, "resamples": resamples, "salts": salts,
              "eligible_worlds": eligible, "absent_groups": [g for g, n in eligible.items() if not n],
              "coverage": coverage, "coverage_shortfall": shortfall,
              "coverage_sufficient": not shortfall,
              "worlds": sorted({r["world_seed"] for r in selected}),
              "starts": [_start_identity(r) for r in selected],
              "grid": [{"world_seed": r["world_seed"], "start_id": r["start_id"],
                        "group": r["group"], "salt": salt} for r in selected for salt in salts]}
    frozen["grid_hash"] = canonical_grid_hash(frozen)
    return selected, frozen


def bank_identity(cohort):
    """Exact validated artifact and science identities, independent of models run."""
    return {"world_start": cohort["config"]["world_start"], "n": cohort["config"]["n"],
            "science_identity_hash": public_hash(cohort["science_identity"]),
            "expected_provenance": cohort["science_identity"]["expected_provenance"],
            "inputs": sorted((file_identity(p) for p in cohort["inputs"]), key=lambda i: i["path"]),
            "shards": sorted((r["shard_identity"] for r in cohort["records"]), key=lambda i: i["path"])}


def save_grid(path, frozen):
    """Exclusive creation; never overwrite an earlier scientific decision."""
    if frozen.get("grid_hash") != canonical_grid_hash(frozen) or frozen.get("bank_identity") is None:
        raise ValueError("Require canonical grid hash and validated bank identity")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(frozen, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    return file_identity(path)


def load_grid(path, rows, targets, resamples, *, bank_identity):
    """Validate the supplied decision; never repair or substitute another grid."""
    try:
        identity = file_identity(path)
        frozen = json.loads(Path(path).read_text())
        if file_identity(path) != identity:
            raise ValueError("Frozen grid input changed while reading")
        if not isinstance(frozen, dict) or frozen.get("grid_hash") != canonical_grid_hash(frozen):
            raise ValueError("Frozen grid canonical hash mismatch")
        if frozen.get("selection_protocol") != SELECTION_PROTOCOL:
            raise ValueError("Frozen grid selection protocol mismatch")
        if frozen.get("targets") != targets or frozen.get("resamples") != resamples:
            raise ValueError("Frozen grid targets/resamples mismatch")
        if frozen.get("bank_identity") != bank_identity:
            raise ValueError("Frozen grid bank identity mismatch")
        # Recompute solely to verify the recorded decision and all replay IDs;
        # return the supplied artifact, never a newly frozen replacement.
        selected, expected = freeze_grid(rows, targets, resamples, bank_identity=bank_identity)
        if frozen != expected:
            raise ValueError("Frozen grid selection/start/public hash/provenance mismatch")
        return selected, frozen, identity
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid frozen grid: {exc}") from exc
