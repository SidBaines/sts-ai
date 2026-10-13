"""Scoped public arithmetic for ``combat_public_v4``.

The v3 implementation remains frozen for replay. Its inputs contain displayed
damage and current costs, not enough state to simulate an action sequence or an
end-of-turn transition. V4 therefore reports only the arithmetic those inputs
support and makes no outcome or HP-loss forecast.
"""
from __future__ import annotations

from sts_ai.turn_math import TurnMathInputs, _max_attack_damage


def turn_math_lines(inputs: TurnMathInputs) -> list[str]:
    """Render restricted calculations, without tactical or outcome claims."""
    attack_difference = max(
        0,
        inputs.incoming_damage - inputs.player_block - inputs.player_metallicize,
    )
    attack_sum, excluded = _max_attack_damage(inputs.hand_attacks, inputs.energy)

    lines = [
        "Incoming attack arithmetic only: max(0, shown incoming "
        f"{inputs.incoming_damage} - current Block {inputs.player_block} "
        f"- Metallicize {inputs.player_metallicize}) = {attack_difference}. "
        "This calculation omits Plated Armor, Burn, Combust, and all other "
        "end-turn effects; it does not predict HP loss.",
        "Fixed-cost attack arithmetic: largest sum of shown Attack damage "
        f"within current energy {max(0, inputs.energy)}, holding current costs "
        f"and shown damage fixed: {attack_sum}. "
        "This does not simulate an action sequence or determine whether an "
        "enemy can be defeated this turn. Card draw, energy generation or "
        "refunds, other card effects, potions, target allocation, and subsequent "
        "changes to costs, damage, or defenses are outside this calculation.",
    ]
    if excluded:
        lines[1] += f" Attacks excluded from this sum: {', '.join(excluded)}."
    return lines
