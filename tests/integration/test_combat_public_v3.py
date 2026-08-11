"""Integration coverage for the combat_public_v3 observation boundary."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap
import unittest
from typing import Any

from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.rollout import prepare_decision
from sts_ai.turn_math import TurnMathInputs, turn_math_lines

from tests.support import requires_simulator


_HAND_ATTACK_RE = re.compile(
    r"^\s*\[\d+\]\s+(.*?)\s+\[Attack\]\s+\(cost\s+(\S+)\)$"
)
_ACTION_ATTACK_RE = re.compile(
    r"^play\s+(.*?)\s+\[Attack\]\s+\(cost\s+(\S+)\).*?"
    r"(?:\(deal\s+(\d+)[^)]*\))?$"
)
_PLAYER_RE = re.compile(r"^Player HP:.*?block:\s*(-?\d+), energy:\s*(-?\d+)/")
_ENEMY_RE = re.compile(
    r"^\s*\[\d+\]\s+(.*?)\s+HP\s+(\d+)/\d+,\s+block\s+(-?\d+)\b"
)
_INCOMING_RE = re.compile(r"^Incoming attack damage this turn:\s+(\d+)\b")
_METALLICIZE_RE = re.compile(r"\bMetallicize\s+(-?\d+)\b")
_TURN_PREFIXES = (
    "End-turn projection:",
    "Max attack damage playable this turn ",
    "Lethal check vs ",
)


def _reach_uninitialized_battle(env: LightspeedHybridEnv) -> None:
    for _ in range(20):
        if env.gc.screen_state == env.sts.ScreenState.BATTLE:
            return
        actions = list(env.gc.legal_actions())
        if not actions:
            break
        actions[0].execute(env.gc)
    raise AssertionError("did not reach an uninitialized battle")


def _cleave_env(observation: str) -> LightspeedHybridEnv:
    env = LightspeedHybridEnv(
        world_seed=4,
        max_act=1,
        combat_control="llm",
        combat_observation=observation,
    )
    _reach_uninitialized_battle(env)
    while len(env.gc.deck):
        env.gc.remove_card(0)
    for _ in range(5):
        env.gc.obtain_card(env.sts.Card(env.sts.CardId.CLEAVE))
    return env


def _warcry_env() -> LightspeedHybridEnv:
    env = LightspeedHybridEnv(
        world_seed=4,
        max_act=1,
        combat_control="llm",
        combat_observation="combat_public_v3",
    )
    _reach_uninitialized_battle(env)
    while len(env.gc.deck):
        env.gc.remove_card(0)
    env.gc.obtain_card(env.sts.Card(env.sts.CardId.WARCRY))
    for _ in range(4):
        env.gc.obtain_card(env.sts.Card(env.sts.CardId.STRIKE_RED))
    return env


def _view(env: LightspeedHybridEnv) -> dict[str, Any]:
    status, view = prepare_decision(env)
    if status != "ok" or view is None or view["phase"] != "combat":
        raise AssertionError(f"expected a combat decision, got {status!r}")
    return view


def _recompute_turn_math(state_text: str, actions: list[dict[str, Any]]) -> list[str]:
    block = energy = incoming = metallicize = 0
    enemies: list[tuple[str, int, int]] = []
    attacks: Counter[tuple[str, int | None]] = Counter()
    for line in state_text.splitlines():
        player_match = _PLAYER_RE.match(line)
        if player_match:
            block, energy = map(int, player_match.groups())
        incoming_match = _INCOMING_RE.match(line)
        if incoming_match:
            incoming = int(incoming_match.group(1))
        if line.startswith("Player powers:"):
            metallicize_match = _METALLICIZE_RE.search(line)
            if metallicize_match:
                metallicize = int(metallicize_match.group(1))
        enemy_match = _ENEMY_RE.match(line)
        if enemy_match:
            enemies.append(
                (
                    enemy_match.group(1),
                    int(enemy_match.group(2)),
                    int(enemy_match.group(3)),
                )
            )
        hand_match = _HAND_ATTACK_RE.match(line)
        if hand_match:
            raw_cost = hand_match.group(2)
            attacks[(hand_match.group(1), int(raw_cost) if raw_cost.isdigit() else None)] += 1

    shown_damage: dict[tuple[str, int | None], int] = {}
    for action in actions:
        match = _ACTION_ATTACK_RE.match(str(action["description"]))
        if match is None or match.group(3) is None:
            continue
        raw_cost = match.group(2)
        key = (match.group(1), int(raw_cost) if raw_cost.isdigit() else None)
        shown_damage[key] = max(shown_damage.get(key, 0), int(match.group(3)))
    return turn_math_lines(
        TurnMathInputs(
            incoming_damage=incoming,
            player_block=block,
            player_metallicize=metallicize,
            energy=energy,
            hand_attacks=[
                (name, cost, shown_damage.get((name, cost)), copies)
                for (name, cost), copies in attacks.items()
            ],
            living_enemies=enemies,
        )
    )


_DETERMINISM_DRIVER = """
import json
from sts_ai.lightspeed import LightspeedHybridEnv
from sts_ai.rollout import prepare_decision

env = LightspeedHybridEnv(
    world_seed=4,
    battle_simulations=2,
    max_act=1,
    combat_control="llm",
    combat_observation="combat_public_v3",
)
for _ in range(100):
    status, view = prepare_decision(env)
    if status == "ok" and view["phase"] == "combat":
        print(json.dumps({
            "state_text": view["state_text"],
            "legal_actions": view["legal_action_dicts"],
        }, sort_keys=True))
        break
    if status != "ok":
        raise RuntimeError(status)
    env.step(0)
else:
    raise RuntimeError("did not reach combat")
"""


@requires_simulator
class CombatPublicV3Test(unittest.TestCase):
    def test_v3_adds_computed_damage_and_consistent_turn_math_only(self) -> None:
        v2 = _view(_cleave_env("combat_public_v2"))
        v3 = _view(_cleave_env("combat_public_v3"))

        v2_cleave = next(
            action.description
            for action in v2["legal_actions"]
            if action.description.startswith("play Cleave")
        )
        v3_cleave = next(
            action.description
            for action in v3["legal_actions"]
            if action.description.startswith("play Cleave")
        )
        self.assertNotIn("(deal ", v2_cleave)
        self.assertIn("(deal ", v3_cleave)
        self.assertFalse(
            any(line.startswith(_TURN_PREFIXES) for line in v2["state_text"].splitlines())
        )

        actual_lines = [
            line
            for line in v3["state_text"].splitlines()
            if line.startswith(_TURN_PREFIXES)
        ]
        self.assertEqual(
            actual_lines,
            _recompute_turn_math(v3["state_text"], v3["legal_action_dicts"]),
        )
        self.assertEqual(len(actual_lines), 3)
        v3_without_turn_math = "\n".join(
            line
            for line in v3["state_text"].splitlines()
            if not line.startswith(_TURN_PREFIXES)
        )
        # This equality depends on a first-decision fixture: after any action, v3's
        # recent-action history legitimately echoes its annotated description.
        self.assertEqual(v3_without_turn_math, v2["state_text"])
        lines = v3["state_text"].splitlines()
        incoming_index = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("Incoming attack damage this turn:")
        )
        self.assertEqual(lines[incoming_index + 1:incoming_index + 4], actual_lines)
        key_index = lines.index("-- KEY (effects/statuses; numbers are shown next to each above) --")
        self.assertLess(incoming_index + 3, key_index)

    def test_card_select_omits_turn_math(self) -> None:
        env = _warcry_env()
        normal = _view(env)
        warcry_index = next(
            action.index
            for action in normal["legal_actions"]
            if action.description.startswith("play Warcry ")
        )
        env.step(warcry_index)
        self.assertEqual(env.bc.input_state, env.sts.InputState.CARD_SELECT)

        card_select = _view(env)
        self.assertFalse(
            any(
                line.startswith(_TURN_PREFIXES)
                for line in card_select["state_text"].splitlines()
            )
        )

    def test_v3_is_deterministic_across_processes(self) -> None:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = "src" if not existing else f"src:{existing}"
        outputs: list[str] = []
        for _ in range(2):
            completed = subprocess.run(
                [sys.executable, "-c", textwrap.dedent(_DETERMINISM_DRIVER)],
                cwd=Path.cwd(),
                env=env,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"child failed\nstdout={completed.stdout}\nstderr={completed.stderr}",
            )
            json.loads(completed.stdout)
            outputs.append(completed.stdout)
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
