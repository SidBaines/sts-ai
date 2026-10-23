"""Card-instance flags must survive combat text and display-menu de-duplication."""
from __future__ import annotations

import unittest

from sts_env._backend.lightspeed import LightspeedHybridEnv
from tests.integration.test_public_combat_teacher import _advance_to_first_combat
from tests.support import load_search_test_fixture, requires_simulator


@requires_simulator
class CombatCardFlagsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = load_search_test_fixture()

    def duplicate_strikes(self) -> tuple[LightspeedHybridEnv, list[dict]]:
        env = LightspeedHybridEnv(
            world_seed=3, max_act=1, combat_control="llm",
            combat_observation="combat_public_v3",
        )
        _advance_to_first_combat(env)
        strikes = [card for card in env.public_combat_cards()["hand"]
                   if card["name"] == "Strike" and card["cost_for_turn"] == 1]
        self.assertGreaterEqual(len(strikes), 2, "Seed 3 must provide two cost-1 Strikes")
        return env, strikes

    def test_free_once_and_retained_copies_stay_selectable_and_spend_correct_energy(self) -> None:
        for free_once, retained in ((True, False), (False, True), (True, True)):
            with self.subTest(free_once=free_once, retained=retained):
                env, strikes = self.duplicate_strikes()
                special_idx, ordinary_idx = strikes[0]["index"], strikes[1]["index"]
                env.bc = self.fixture.hand_flags_copy(env.bc, special_idx, free_once, retained)
                suffix = (", free once" if free_once else "") + (", retained" if retained else "")

                # Test both legacy and typed descriptions; ordinary labels retain
                # their exact cost block, without the pile label's base cost.
                for typed in (False, True):
                    actions = [action for action in env.bc.legal_actions(typed)
                               if action.action_type() == env.sts.BattleActionType.CARD]
                    special = next(a for a in actions if a.source_idx() == special_idx)
                    ordinary = next(a for a in actions if a.source_idx() == ordinary_idx
                                    and a.target_idx() == special.target_idx())
                    special_text = special.describe(env.bc, include_card_type=typed)
                    ordinary_text = ordinary.describe(env.bc, include_card_type=typed)
                    self.assertEqual(special_text.replace(suffix, "", 1), ordinary_text)
                    self.assertIn(f"(cost 1{suffix})", special_text)
                    hand_text = env.bc.describe_state(public_state=True, include_card_type=typed)
                    tag = " [Attack]" if typed else ""
                    self.assertIn(f"  [{special_idx}] Strike{tag} (cost 1{suffix})", hand_text)
                    self.assertIn(f"  [{ordinary_idx}] Strike{tag} (cost 1)", hand_text)

                raw, display, mapping = env._action_views()
                for action in (special, ordinary):
                    raw_index = next(i for i, candidate in enumerate(raw) if candidate.bits == action.bits)
                    self.assertIn(raw_index, mapping)
                    self.assertEqual(display[mapping.index(raw_index)].description,
                                     action.describe(env.bc, include_card_type=True,
                                                     include_computed_damage=True))

                # Execute each choice on a separate scratch copy so target damage,
                # hand-slot shifts, and energy cannot confound the comparison.
                for action, energy_cost in ((special, 0 if free_once else 1), (ordinary, 1)):
                    copy = self.fixture.copy(env.bc)
                    before = copy.public_player()["energy"]
                    action.execute(copy)
                    self.assertEqual(before - copy.public_player()["energy"], energy_cost)


if __name__ == "__main__":
    unittest.main()
