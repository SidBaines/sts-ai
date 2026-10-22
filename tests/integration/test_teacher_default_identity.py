"""Compare default search results in fresh processes loading two native builds."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from sts_policy.curriculum import TaskSpec, public_hash
from sts_rl.fresh_collection import search_plan
from sts_rl.fresh_panel import FreshPanelEpisode
from tests.support import requires_simulator


def snapshot():
    positions = []
    episode = FreshPanelEpisode({})
    for seed in (3, 7, 11):
        observation, _ = episode.reset(TaskSpec(seed=seed, kind="battle", battle_simulations=16))
        for decision in range(2):
            state_hash = public_hash(observation)
            plan = search_plan(state_hash, samples=4)
            details = episode.search_details_public(256, plan["samples"], plan["effective_search_seeds"],
                                                    plan["search_seed"])
            raw = episode.episode.env._env
            native_results = []
            for future, search_seed in zip(plan["samples"], plan["effective_search_seeds"]):
                scratch = raw.bc.copy()
                scratch.reseed_rngs(plan["search_seed"] * 1009 + future)
                result = scratch.search_best_action(256, search_seed=search_seed, draw_order_seed=future)
                native_results.append({"chosen_bits": int(result["bits"]),
                    "root_edges": [{"bits": int(edge["bits"]), "valid": edge["valid"],
                                    "visits": edge["visits"], "evaluation_sum": edge["evaluation_sum"].hex()}
                                   for edge in result["root_edges"]]})
            # Compare the exact binary floating-point values, including signed zero.
            for row in details:
                for edge in row["edges"]:
                    edge["evaluation_sum"] = edge["evaluation_sum"].hex()
                    edge["mean_evaluation"] = edge["mean_evaluation"].hex()
                for key in ("best_evaluation", "min_evaluation"):
                    row[key] = row[key].hex()
            positions.append({"seed": seed, "decision": decision, "public_hash": state_hash,
                              "details": details, "native": native_results})
            observation, _, terminated, truncated, _ = episode.step(details[0]["chosen"])
            if terminated or truncated or observation["phase"] != "combat":
                break
    return positions


@requires_simulator
class TeacherDefaultIdentityTests(unittest.TestCase):
    def test_two_builds_are_bit_identical(self):
        root = Path(__file__).resolve().parents[2]
        baseline = Path(os.environ.get("STS_SEARCH_BASELINE_BUILD", root / "external/sts_lightspeed_teacher/build_baseline"))
        if not baseline.is_dir():
            self.skipTest("Set STS_SEARCH_BASELINE_BUILD to the unchanged reference build")
        builds = (str(baseline), os.environ.get("STS_LIGHTSPEED_BUILD", str(root / "external/sts_lightspeed/build")))
        snapshots = []
        for build in builds:
            result = subprocess.run([sys.executable, "-m", "tests.integration.test_teacher_default_identity", "--snapshot"],
                                    env={**os.environ, "STS_LIGHTSPEED_BUILD": build}, cwd=root,
                                    capture_output=True, text=True, timeout=60, check=True)
            snapshots.append(json.loads(result.stdout))
        self.assertEqual(len(snapshots[0]), 6)
        self.assertEqual(*snapshots)
        print("Bit identity: 6 positions, 24 searches; chosen actions, root visits and evaluation sums identical.")


if __name__ == "__main__":
    if sys.argv[1:] == ["--snapshot"]:
        print(json.dumps(snapshot(), allow_nan=False))
    else:
        unittest.main()
