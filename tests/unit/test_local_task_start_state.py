from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
import unittest

from sts_ai.local_tasks.runner import replay_task_start
from sts_ai.local_tasks.start_state import (
    LocalTaskStartSignatureError,
    build_start_state_signature,
    validate_stored_start_state_signature,
)


class FakeCombatEnv:
    def __init__(self) -> None:
        self.bc = object()
        self.combat_observation = "legacy"
        self.public_state_text = (
            "Player: 60/80 HP, 3/3 energy\n"
            "Draw pile (5): Defend x2, Strike x3\n"
            "Enemies: GREMLIN_NOB 82/82 HP; intent BELLOW"
        )
        self.hidden_draw_order = ["Strike", "Defend", "Strike"]
        self.action_descriptions = [
            "play Bash (cost 2) -> GREMLIN_NOB [enemy 0] (deal 8)",
            "end turn",
        ]
        self.advance_calls = 0

    def advance_to_decision(self) -> int:
        self.advance_calls += 1
        return 0

    def is_terminal(self) -> bool:
        return False

    def describe_state(self) -> str:
        if self.combat_observation == "combat_public_v2":
            return self.public_state_text
        return "legacy state omits piles"

    def legal_actions(self):
        descriptions = list(self.action_descriptions)
        if self.combat_observation == "combat_public_v2":
            descriptions[0] = descriptions[0].replace(
                "Bash (cost", "Bash [Attack] (cost"
            )
        return [
            SimpleNamespace(index=index, bits=100 + index, description=description)
            for index, description in enumerate(descriptions)
        ]

    def summary(self) -> dict:
        return {
            "combat": {
                "enemies": [
                    {
                        "index": 0,
                        "name": "GREMLIN_NOB",
                        "alive": True,
                    }
                ]
            }
        }


class LocalTaskStartStateTest(unittest.TestCase):
    def test_signature_uses_public_text_display_actions_and_encounter_only(self):
        env = FakeCombatEnv()
        signature = build_start_state_signature(env)

        self.assertEqual(env.combat_observation, "legacy")
        payload = signature["payload"]
        self.assertEqual(payload["state_text"], env.public_state_text)
        self.assertEqual(
            payload["legal_action_descriptions"],
            [
                "play Bash [Attack] (cost 2) -> GREMLIN_NOB [enemy 0] (deal 8)",
                "end turn",
            ],
        )
        self.assertEqual(
            payload["encounter"],
            {
                "enemy_counts": {"GREMLIN_NOB": 1},
                "exact_composition": True,
            },
        )
        self.assertNotIn("bits", json.dumps(payload))
        canonical = json.dumps(
            {"schema_version": 2, "payload": payload},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(signature["sha256"], hashlib.sha256(canonical).hexdigest())

        # Simulator-private pile order is intentionally outside the signature.
        env.hidden_draw_order.reverse()
        self.assertEqual(build_start_state_signature(env), signature)

    def test_action_or_public_state_drift_fails_closed(self):
        env = FakeCombatEnv()
        window = {
            "window_id": "seed_4_r0_w0",
            "start_state_signature": build_start_state_signature(env),
        }
        self.assertIsNotNone(validate_stored_start_state_signature(env, window))

        env.action_descriptions[0] = "end turn"
        with self.assertRaisesRegex(
            LocalTaskStartSignatureError,
            "public start state diverged",
        ):
            validate_stored_start_state_signature(env, window)

    def test_corrupt_stored_payload_fails_before_replay_comparison(self):
        env = FakeCombatEnv()
        signature = build_start_state_signature(env)
        signature["payload"]["state_text"] = "tampered"
        with self.assertRaisesRegex(
            LocalTaskStartSignatureError,
            "internally inconsistent",
        ):
            validate_stored_start_state_signature(
                env,
                {
                    "window_id": "seed_4_r0_w0",
                    "start_state_signature": signature,
                },
            )

    def test_shared_replay_entrypoint_enforces_validated_signature(self):
        env = FakeCombatEnv()
        window = {
            "window_id": "seed_4_r0_w0",
            "pre_actions": [],
            "start_state_signature": build_start_state_signature(env),
        }
        replay_task_start(env, window)

        env.public_state_text += "\nPotions: [0] Strength Potion"
        with self.assertRaisesRegex(
            LocalTaskStartSignatureError,
            "public start state diverged",
        ):
            replay_task_start(env, window)


if __name__ == "__main__":
    unittest.main()
