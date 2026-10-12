"""Transition-based regressions for the public-v2 instance menu."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from sts_env import SlayTheSpireEnv
from sts_env._backend.combat_public_v5 import CombatPublicV5Env
from sts_env._backend.lightspeed import LightspeedHybridEnv
from sts_env.env import OBSERVATION_VERSION
from tests.support import requires_simulator


# Reached TRAIN world from the independent critique. Legacy prefixes use the
# old display menu. Replaying them under a new menu without translation is wrong.
_PREFIX = (0, 1, 0, 1, 2, 0, 2, 0, 1, 1, 2, 0, 0, 0)
_PREVIEWS = (True, True, True, True, False, False, False,
             False, False, False, False, False, False, False)


def replay_legacy_frontier() -> CombatPublicV5Env:
    raw = LightspeedHybridEnv(12022, combat_control="llm", max_act=3,
                              combat_observation="combat_public_v3",
                              build_dir=os.environ.get("STS_LIGHTSPEED_BUILD"))
    raw.seed_native_ooc(12022)
    env = CombatPublicV5Env(raw)
    env.advance_to_decision()
    for index, preview in zip(_PREFIX, _PREVIEWS):
        if preview:
            env.native_ooc_action()
        env.step(index)
        env.advance_to_decision()
    raw.preserve_card_instances = True
    return env


@requires_simulator
class PublicActionInstanceTests(unittest.TestCase):
    def test_free_and_normal_strikes_survive_and_execute_distinct_resources(self):
        observed = []
        for source in (1, 2):
            env = replay_legacy_frontier()
            actions = env.legal_actions()
            public = env.public_action_metadata()
            strikes = [(action, mechanics) for action, mechanics in zip(actions, public)
                       if mechanics.get("kind") == "play_card"
                       and mechanics["card"]["base_name"] == "Strike"]
            self.assertEqual(len(strikes), 2)
            self.assertEqual({m["source_index"] for _, m in strikes}, {1, 2})
            action, mechanics = next((a, m) for a, m in strikes if m["source_index"] == source)
            self.assertEqual(env.display_index_for_bits(action.bits), action.index)
            self.assertEqual(mechanics["energy_cost"], 0 if source == 1 else 1)
            self.assertNotIn("bits", mechanics)
            self.assertNotIn("unique_id", mechanics["card"])
            self.assertIn(f"hand {source}", action.description)
            env.step(action.index)
            after = env.summary()["combat"]
            observed.append((after["player_energy"], after["cards"]["hand"]))
        self.assertEqual([row[0] for row in observed], [3, 2])
        self.assertNotEqual(observed[0][1], observed[1][1])

    def test_inventory_and_current_actions_are_typed_public_copies(self):
        with SlayTheSpireEnv(isolated=False, mode="full", max_steps=100,
                            max_act=1, battle_simulations=50) as env:
            obs, _ = env.reset(seed=3)
            self.assertEqual(obs["state"]["observation_version"], OBSERVATION_VERSION)
            deck = obs["state"]["inventory"]["deck"]
            self.assertEqual(len(deck), 10)
            self.assertTrue(all("base_name" in c and "upgrade_count" in c for c in deck))
            self.assertTrue(all("public" in a and "bits" not in a for a in obs["legal_actions"]))
            serialized = json.dumps(obs)
            for private in ("shuffle_rng", "unique_id", "source_bits", "world_seed"):
                self.assertNotIn(private, serialized)
            obs["state"]["inventory"]["deck"].clear()
            self.assertEqual(len(env.observation["state"]["inventory"]["deck"]), 10)

    def test_repaired_full_control_traces_match_across_processes_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            files = [Path(directory) / f"trace{i}.jsonl" for i in range(2)]
            for path in files:
                result = subprocess.run([sys.executable, "-m", "sts_env", "run",
                    "--agent", "random", "--mode", "full", "--seed", "12022",
                    "--max-act", "1", "--max-steps", "80", "--output", str(path)],
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            traces = [[json.loads(line) for line in p.read_text().splitlines()] for p in files]
            for trace in traces:
                trace[-1].pop("elapsed_seconds")
            self.assertEqual(traces[0], traces[1])
            result = subprocess.run([sys.executable, "-m", "sts_env", "replay", str(files[0])],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
