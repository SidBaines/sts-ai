from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from sts_env._backend.combat_public_v4 import CombatPublicV4Env
from sts_env._backend.lightspeed import LightspeedHybridEnv
from sts_env._backend.schemas import LegalAction


class CombatPublicV4AdapterTest(unittest.TestCase):
    def _env(self) -> Mock:
        env = Mock(combat_observation="combat_public_v3", bc=None)
        env.legal_actions.return_value = [
            LegalAction(0, 1610612736, "{ GAMBLE none"),
            LegalAction(1, 17, "{ GAMBLE 1"),
        ]
        env.step.side_effect = lambda index: env.legal_actions()[index]
        env.summary.return_value = {
            "combat_observation": "combat_public_v3",
            "combat": {
                "cards": {
                    "selection": {"type": "GAMBLE", "candidates": ["{ GAMBLE none"]},
                    "draw": [{"name": "Strike", "count": 2}],
                },
                "recent_actions": ["{ GAMBLE none", "play Strike -> BLUE_SLAVER"],
            },
        }
        env.public_combat_cards.return_value = env.summary()["combat"]["cards"]
        return env

    def test_only_audited_action_label_changes_and_step_keeps_identity(self) -> None:
        base = self._env()
        wrapped = CombatPublicV4Env(base)
        actions = wrapped.legal_actions()
        self.assertEqual(actions[0], LegalAction(
            0, 1610612736, "keep current hand (discard no cards)",
        ))
        self.assertEqual(actions[1], base.legal_actions()[1])
        self.assertEqual(wrapped.step(0), actions[0])
        base.step.assert_called_once_with(0)
        self.assertEqual(base.legal_actions()[0].description, "{ GAMBLE none")

    def test_structured_public_views_are_consistent_and_do_not_mutate_v3(self) -> None:
        base = self._env()
        original = deepcopy(base.summary())
        wrapped = CombatPublicV4Env(base)
        summary = wrapped.summary()
        self.assertEqual(summary["combat_observation"], "combat_public_v4")
        self.assertEqual(summary["combat"]["cards"]["selection"]["candidates"],
                         [wrapped.legal_actions()[0].description])
        self.assertEqual(summary["combat"]["recent_actions"][0],
                         wrapped.legal_actions()[0].description)
        self.assertEqual(wrapped.public_combat_cards(), summary["combat"]["cards"])
        summary["combat"]["cards"]["draw"][0]["name"] = "mutated copy"
        self.assertEqual(base.summary(), original)
        self.assertEqual(base.combat_observation, "combat_public_v3")

    def test_state_text_normalizes_candidate_and_history_only(self) -> None:
        base = self._env()
        base.describe_state.return_value = (
            "Selection type: GAMBLE\nSelection candidates:\n  - { GAMBLE none\n"
            "  - { GAMBLE 1\nRecent actions this turn: { GAMBLE none -> "
            "play Strike [Attack] (cost 1) -> BLUE_SLAVER (deal 6)\n"
        )
        text = CombatPublicV4Env(base).describe_state()
        self.assertIn("  - keep current hand (discard no cards)\n", text)
        self.assertIn("Recent actions this turn: keep current hand (discard no cards) -> "
                      "play Strike [Attack] (cost 1) -> BLUE_SLAVER (deal 6)", text)
        self.assertIn("  - { GAMBLE 1\n", text)
        self.assertTrue(text.endswith("\n"))

    def test_requires_v3_base_and_delegates_phase(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a combat_public_v3"):
            CombatPublicV4Env(SimpleNamespace(combat_observation="legacy"))
        base = self._env()
        base.phase.return_value = "combat"
        self.assertEqual(CombatPublicV4Env(base).phase(), "combat")

    def test_no_turn_math_outside_normal_player_input(self) -> None:
        base = self._env()
        self.assertEqual(CombatPublicV4Env(base)._turn_math_lines(), [])
        base.sts = SimpleNamespace(InputState=SimpleNamespace(PLAYER_NORMAL="normal"))
        base.bc = SimpleNamespace(input_state="select")
        self.assertEqual(CombatPublicV4Env(base)._turn_math_lines(), [])

    def test_ooc_text_and_state_remain_identical_except_version(self) -> None:
        base = self._env()
        base.summary.return_value = {"phase": "out_of_combat", "combat_observation": "combat_public_v3"}
        base.describe_state.return_value = "Your HP: 80/80\n"
        wrapped = CombatPublicV4Env(base)
        self.assertEqual(wrapped.describe_state(), base.describe_state())
        self.assertEqual(wrapped.summary(), {"phase": "out_of_combat", "combat_observation": "combat_public_v4"})


class TurnMathCardFlagsTest(unittest.TestCase):
    def test_damage_lookup_accepts_flags_and_matches_the_whole_cost(self):
        for free_once, retained in ((False, False), (True, False), (False, True), (True, True)):
            for adapter in (None, CombatPublicV4Env):
                with self.subTest(free_once=free_once, retained=retained, adapter=adapter):
                    suffix = (", free once" if free_once else "") + (", retained" if retained else "")
                    descriptions = [
                        f"play Strike+ [Attack] (cost 1{suffix}) -> FOE (deal 9)",
                        "play Strike+ [Attack] (cost 10) -> FOE (deal 99)",
                    ]
                    bc = SimpleNamespace(
                        input_state="normal",
                        public_cards=lambda **kw: {"hand": [{
                            "name": "Strike+", "type": "Attack", "cost_for_turn": 1,
                            "free_to_play_once": free_once, "retain": retained,
                        }]},
                        public_player=lambda: {"energy": 3},
                        enemies=lambda: [],
                        legal_actions=lambda **kw: [
                            SimpleNamespace(describe=lambda *args, text=text, **kw: text)
                            for text in descriptions
                        ],
                    )
                    base = object.__new__(LightspeedHybridEnv)
                    base.bc = bc
                    base.combat_observation = "combat_public_v3"
                    base.sts = SimpleNamespace(InputState=SimpleNamespace(PLAYER_NORMAL="normal"))
                    env = base if adapter is None else adapter(base)
                    module = "lightspeed" if adapter is None else "combat_public_v4"
                    with patch(f"sts_env._backend.{module}.turn_math_lines", return_value=[]) as render:
                        env._turn_math_lines()
                    inputs = render.call_args.args[0]
                    self.assertEqual(inputs.hand_attacks, [("Strike+", 0 if free_once else 1, 9, 1)])
