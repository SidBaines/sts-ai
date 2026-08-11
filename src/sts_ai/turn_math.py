"""Pure derived arithmetic for the ``combat_public_v3`` observation."""
from __future__ import annotations

from dataclasses import dataclass


# (display label, current integer cost or None for X/unusable, shown total damage,
# hand multiplicity).  Damage is deliberately supplied by the simulator-facing
# layer; this module does not reproduce card damage rules.
HandAttack = tuple[str, int | None, int | None, int]


@dataclass(frozen=True)
class TurnMathInputs:
    incoming_damage: int
    player_block: int
    player_metallicize: int
    energy: int
    hand_attacks: list[HandAttack]
    living_enemies: list[tuple[str, int, int]]


def _max_attack_damage(attacks: list[HandAttack], energy: int) -> tuple[int, list[str]]:
    """Bounded 0/1 knapsack after expanding each grouped hand-card copy."""
    available_energy = max(0, energy)
    dp = [0] * (available_energy + 1)
    excluded: list[str] = []
    for label, cost, deal_total, copies in attacks:
        if (
            cost is None
            or cost < 0
            or cost > available_energy
            or deal_total is None
        ):
            if label not in excluded:
                excluded.append(label)
            continue
        for _ in range(max(0, copies)):
            if cost == 0:
                for current_energy in range(available_energy + 1):
                    dp[current_energy] += deal_total
                continue
            for current_energy in range(available_energy, cost - 1, -1):
                dp[current_energy] = max(
                    dp[current_energy],
                    dp[current_energy - cost] + deal_total,
                )
    return dp[available_energy], excluded


def turn_math_lines(inputs: TurnMathInputs) -> list[str]:
    """Render the frozen public turn-arithmetic lines from displayed values."""
    projected_damage = max(
        0,
        inputs.incoming_damage - inputs.player_block - inputs.player_metallicize,
    )
    max_damage, excluded = _max_attack_damage(inputs.hand_attacks, inputs.energy)

    lines = [
        "End-turn projection: you would take "
        f"{projected_damage} damage (incoming {inputs.incoming_damage} - block "
        f"{inputs.player_block} - Metallicize {inputs.player_metallicize}; minimum 0).",
        "Max attack damage playable this turn (using shown deal values, current "
        f"modifiers only): {max_damage}.",
    ]
    if excluded:
        lines[-1] += f" Excluded from this total: {', '.join(excluded)}."

    if len(inputs.living_enemies) == 1:
        enemy_name, enemy_hp, enemy_block = inputs.living_enemies[0]
        outcome = (
            "lethal available this turn."
            if max_damage >= enemy_hp + enemy_block
            else "not lethal this turn."
        )
        lines.append(
            f"Lethal check vs {enemy_name} (HP {enemy_hp}, block "
            f"{enemy_block}): {outcome}"
        )
    return lines
