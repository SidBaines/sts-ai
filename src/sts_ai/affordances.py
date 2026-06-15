"""Structured, sim-grounded per-decision "affordances" for eval support.

The agent already *sees* block/damage (via the glossary KEY and the `(deal N)`
annotations); this module is purely for downstream evals — it answers, for each
combat decision, "what could the agent have done?" so analyses like "did it
forgo a full block when one was available" or "did it take an available lethal"
can be computed directly instead of re-parsing `state_text`.

Pure Python, no simulator import. Block is reimplemented faithfully from
`BattleContext::calculateCardBlock` (NO_BLOCK -> 0; +Dexterity floored at 0;
x3/4 if Frail); card block bases are a curated Ironclad table (the engine has no
block table — values from the playCard switch). Best-effort and defensive: any
parse gap degrades to a smaller/empty dict rather than raising, so a rollout is
never broken by an affordance bug. Out of combat returns `{}`.
"""
from __future__ import annotations

import re
from typing import Any, Optional

# Base Block for Ironclad block-producing cards (unupgraded, upgraded), as fed to
# calculateCardBlock in the playCard switch. Keyed by display name (Cards.h
# cardNames). State-dependent cards (Second Wind = base x non-attack count) are
# omitted; Entrench (doubles current Block) and Iron Wave (double-applied) are
# handled specially below.
BLOCK_BASE: dict[str, tuple[int, int]] = {
    "Defend": (5, 8),
    "Sentinel": (5, 8),
    "Shrug It Off": (8, 11),
    "True Grit": (7, 9),
    "Iron Wave": (5, 7),        # block is calculateCardBlock(calculateCardBlock(base))
    "Ghostly Armor": (10, 13),
    "Power Through": (15, 20),
    "Flame Barrier": (12, 16),
    "Impervious": (30, 40),
    "Good Instincts": (6, 9),
    "Finesse": (2, 4),
    "Armaments": (5, 5),
    "Panic Button": (30, 40),
}

_PLAY_RE = re.compile(
    r"^play\s+(.*?)\s+\(cost\s+(\S+?)\)"
    r"(?:\s*->\s*(.*?))?"
    r"(?:\s*\(deal\s+(\d+)[^)]*\))?\s*$"  # leading number = total damage; anchor forces full match
)
_HAND_RE = re.compile(r"^\[(\d+)\]\s+(.*?)\s+\(cost\s+(\S+?)\)\s*$")
_DEX_RE = re.compile(r"\bDexterity\s+(-?\d+)")


def calc_block(base: int, dex: int, frail: bool, no_block: bool) -> int:
    """Faithful reimplementation of BattleContext::calculateCardBlock."""
    if no_block:
        return 0
    block = max(0, base + dex)
    if frail:
        return block * 3 // 4
    return block


def _parse_player_powers(state_text: str) -> tuple[int, bool, bool]:
    """(dexterity, frail, no_block) from the 'Player powers:' line."""
    for line in state_text.splitlines():
        if line.startswith("Player powers:"):
            dex_match = _DEX_RE.search(line)
            dex = int(dex_match.group(1)) if dex_match else 0
            return dex, ("Frail" in line), ("No Block" in line)
    return 0, False, False


def _parse_hand(state_text: str) -> list[tuple[str, Optional[int]]]:
    """(name, cost) per hand card; cost is None for X-cost / unparseable."""
    hand: list[tuple[str, Optional[int]]] = []
    in_hand = False
    for line in state_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Hand:"):
            in_hand = stripped[len("Hand:"):].strip() != "empty"
            continue
        if in_hand:
            match = _HAND_RE.match(stripped)
            if not match:
                break
            cost_tok = match.group(3)
            cost = int(cost_tok) if cost_tok.lstrip("-").isdigit() else None
            hand.append((match.group(2), cost))
    return hand


def _card_block(name: str, dex: int, frail: bool, no_block: bool, cur_block: int) -> Optional[int]:
    """Block a single hand card would produce now (None if not a block card)."""
    base_name = name[:-1] if name.endswith("+") else name
    upgraded = name.endswith("+")
    if base_name == "Entrench":
        return cur_block  # doubles current Block, i.e. adds another `cur_block`
    entry = BLOCK_BASE.get(base_name)
    if entry is None:
        return None
    base = entry[1] if upgraded else entry[0]
    if base_name == "Iron Wave":  # engine double-applies calculateCardBlock
        return calc_block(calc_block(base, dex, frail, no_block), dex, frail, no_block)
    return calc_block(base, dex, frail, no_block)


def _max_block_within_energy(items: list[tuple[int, int]], energy: int) -> int:
    """0/1 knapsack: most total block achievable spending <= energy.
    items = (cost, block). cost-0 cards are always free to include."""
    dp = [0] * (energy + 1)
    for cost, block in items:
        if cost <= 0:
            for e in range(energy + 1):
                dp[e] += block
            continue
        for e in range(energy, cost - 1, -1):
            dp[e] = max(dp[e], dp[e - cost] + block)
    return dp[energy] if energy >= 0 else 0


def compute(state: dict[str, Any], state_text: str, legal_actions: list[dict[str, Any]],
            phase: str) -> dict[str, Any]:
    """Per-decision affordances dict (combat only; {} otherwise)."""
    if phase != "combat":
        return {}
    combat = (state or {}).get("combat")
    if not isinstance(combat, dict):
        return {}

    enemies = [e for e in combat.get("enemies", []) if e.get("alive")]
    energy = int(combat.get("player_energy", 0))
    cur_block = int(combat.get("player_block", 0))
    dex, frail, no_block = _parse_player_powers(state_text)

    incoming = sum(
        int(e.get("intent_damage", 0)) * int(e.get("intent_hits", -1))
        for e in enemies
        if int(e.get("intent_hits", -1)) > 0
    )

    # Block options from the hand (multi-card, energy-constrained).
    block_items: list[tuple[int, int]] = []
    for name, cost in _parse_hand(state_text):
        block = _card_block(name, dex, frail, no_block, cur_block)
        if block is not None and cost is not None:
            block_items.append((cost, block))
    max_block = _max_block_within_energy(block_items, energy)
    defensible_total = cur_block + max_block

    # Attack options from the (already-computed) (deal N) on legal actions.
    enemy_hp_by_name: dict[str, int] = {}
    for e in enemies:
        enemy_hp_by_name.setdefault(str(e.get("name")), int(e.get("cur_hp", 0)) + int(e.get("block", 0)))
    max_single = 0
    lethal = False
    play_count = 0
    end_turn = False
    for action in legal_actions:
        desc = str(action.get("description", ""))
        if desc.strip() == "end turn":
            end_turn = True
            continue
        match = _PLAY_RE.match(desc)
        if not match:
            continue
        play_count += 1
        deal = match.group(4)
        if deal is None:
            continue
        dmg = int(deal)
        max_single = max(max_single, dmg)
        target = (match.group(3) or "").strip()
        if target and target in enemy_hp_by_name and dmg >= enemy_hp_by_name[target]:
            lethal = True

    return {
        "incoming_damage_total": incoming,
        "player_block": cur_block,
        "max_block_available": max_block,
        "defensible_total": defensible_total,
        "full_block_possible": bool(incoming > 0 and defensible_total >= incoming),
        "n_living_enemies": len(enemies),
        "max_single_target_damage": max_single,
        "single_target_lethal_available": lethal,
        "player_energy": energy,
        "playable_card_count": play_count,
        "end_turn_available": end_turn,
    }
