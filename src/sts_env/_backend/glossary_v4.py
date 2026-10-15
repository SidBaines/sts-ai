"""Combat-public-v4 reference text, preserving the frozen v1-v3 glossary.

Only public state and legal choices enter this pure renderer. The existing
glossary supplies unchanged definitions/parsers; overrides live here rather than
mutating its dictionaries or changing historical prompt reconstruction.
"""
from __future__ import annotations

import re

from sts_env._backend import glossary as legacy


_INTENT_RE = re.compile(r"\bintent\s+([A-Z][A-Z0-9_]*)\b")
_INTENT_ADDITIONS = {
    # MonsterSpecific.cpp, GREMLIN_NOB_SKULL_BASH (all ascensions).
    "GREMLIN_NOB_SKULL_BASH": "applies 2 Vulnerable to you",
}

# Native grounding: BattleContext.cpp onUseCard/onExhaust/end-turn relics,
# Player.cpp applyStartOfTurnRelics, and Player.h debuff immunity checks.
_RELIC_ADDITIONS = {
    "Blue Candle": "You can play Curse cards; playing one loses 1 HP and Exhausts it. Other play restrictions still apply.",
    "Captains Wheel": "At the start of the third combat turn (displayed Battle turn 2), gain 18 Block.",
    "Charons Ashes": "Whenever a card is Exhausted, deal 3 damage to ALL enemies.",
    "Gambling Chip": "At the start of combat, resolve a discard/redraw selection for your opening hand: discard the selected cards and draw that many replacements. Selecting none keeps the hand.",
    "Ginger": "Prevents Weak from being applied to you.",
    "Incense Burner": "At the start of each turn, increase its counter by 1. At 6, reset the counter to 0 and gain 1 Intangible.",
    "Ornamental Fan": "Every third Attack played in a turn grants 4 Block; use the current-turn attacks counter.",
    "Paper Phrog": "Vulnerable enemies take 75% more attack damage instead of 50%; already included in shown deal N values.",
    "Stone Calendar": "At the end of the seventh combat turn (displayed Battle turn 6), deal 52 damage to ALL enemies.",
    "Turnip": "Prevents Frail from being applied to you.",
}

_DUPLICATION_DEFINITION = (
    "Duplication: Your next card is played twice without paying its cost again. "
    "Each original card played consumes 1 charge. Its number is remaining "
    "charges; after the enemy phase, 1 unused charge is removed."
)

_INTERACTION_NOTE = (
    "One response executes one legal action, not a whole turn. After that action "
    "resolves, you receive the updated state and another choice if combat still "
    "needs a decision. You can play multiple cards in the same turn, one action "
    "per response, while they remain legal. Choosing end turn starts end-of-turn "
    "effects and the enemy phase; some card or enemy effects can also end your turn."
)

_CARD_VALUE_LEGEND = (
    "Card rule notation: a + marks an upgraded card. Paired numbers such as "
    "6 (9) mean unmodified base (upgraded) values, not values after Strength or "
    "other modifiers. Inline (deal N) annotations on actions give the current "
    "computed damage instead; do not add those modifiers again. Battle turn "
    "numbers start at 0 (the first turn)."
)


def _status_names_in_text(text: str) -> set[str]:
    names = legacy._status_names_in_text(text)
    if re.search(r"(?<![A-Za-z])Duplication(?![A-Za-z])", text):
        names.add("Duplication")
    return names


def _scan_status_names(line: str) -> set[str]:
    names = legacy._scan_status_names(line)
    if re.search(r"(?<![A-Za-z])Duplication\s+-?\d+", line):
        names.add("Duplication")
    return names


def _label_intent(line: str) -> tuple[str, set[str]]:
    match = _INTENT_RE.search(line)
    if match is None:
        return line, set()
    move = match.group(1)
    effect = _INTENT_ADDITIONS.get(move) or legacy.intent_effect(move)
    refs = _status_names_in_text(effect) if effect else set()
    if "(deal" in line:
        return (f"{line} (also: {effect})" if effect else line), refs
    if effect:
        prefix = "no attack" if move in legacy._NON_ATTACK_DAMAGE_INTENTS else "no damage"
        label = f"{prefix}; {effect}"
    else:
        label = "no attack"
    cut = match.end(1)
    return f"{line[:cut]} ({label}){line[cut:]}", refs


def _status_definition(name: str, relic_names: list[str]) -> str | None:
    if name == "Duplication":
        return _DUPLICATION_DEFINITION
    if name == "Vulnerable" and "Paper Phrog" in relic_names:
        player_multiplier = "25%" if "Odd Mushroom" in relic_names else "50%"
        return (
            "Vulnerable: affected enemies take 75% more attack damage because of "
            f"Paper Phrog; you take {player_multiplier} more attack damage while "
            "Vulnerable. These modifiers are already reflected in shown deal N "
            "values; its number is turns remaining."
        )
    return legacy.status_definition(name)


def _card_definition(display_name: str, relic_names: list[str]) -> str | None:
    # Keep instance labels on this corrected effect so a mixed base/upgraded
    # hand gets distinct, unambiguous definitions.
    base = re.sub(r"\+\d*$", "", display_name)
    if base == "Flame Barrier":
        upgraded = display_name != base
        return (
            f"{display_name}: Gain {16 if upgraded else 12} base Block. This turn, "
            f"deal {6 if upgraded else 4} damage to any enemy that attacks you."
        )
    definition = legacy._card_definition(display_name)
    if definition and "Blue Candle" in relic_names and ": Curse. " in definition:
        definition = definition.replace(
            "Unplayable.",
            "Playable with Blue Candle: lose 1 HP and Exhaust this card when played; other play restrictions still apply.",
        )
    return definition


def _build_key(state_text: str, statuses: set[str]) -> str:
    relic_names = legacy._relic_names(state_text)
    # A future relic trigger must be understandable before its power is active.
    # For example, the Incense Burner counter forecasts Intangible even when
    # the current player-powers line is empty. Use the same contextual status
    # definitions as active powers (especially Paper Phrog's Vulnerable rule).
    statuses = set(statuses)
    for name in relic_names:
        effect = _RELIC_ADDITIONS.get(name)
        if effect:
            statuses |= _status_names_in_text(effect)
    definitions: list[str] = []
    for name in sorted(statuses):
        definition = _status_definition(name, relic_names)
        if definition:
            definitions.append(definition)
    for name in legacy._combat_card_names(state_text):
        definition = _card_definition(name, relic_names)
        if definition and definition not in definitions:
            definitions.append(definition)
    for name in relic_names:
        effect = _RELIC_ADDITIONS.get(name)
        definition = f"{name}: {effect}" if effect else legacy.relic_definition(name)
        if definition:
            definitions.append(definition)
    for name in legacy._potion_names(state_text):
        definition = legacy.potion_definition(name)
        if definition:
            definitions.append(definition)
    if not definitions:
        return ""
    return legacy._KEY_HEADER + "\n" + "\n".join(f"  {line}" for line in definitions)


def _selection_note(state_text: str) -> str | None:
    match = re.search(r"^Selection type:\s*(\S+)", state_text, re.MULTILINE)
    if match is None and "(card select in progress)" not in state_text:
        return None
    note = (
        "A card selection is pending. Resolve it using a listed selection action "
        "before normal card play can continue."
    )
    if match and match.group(1) == "GAMBLE":
        note += (
            " GAMBLE means discard the selected cards from your hand and draw "
            "the same number of replacements. Selecting none keeps your hand "
            "and completes this selection. Choose only among the offered legal selections."
        )
    return note


def augment(
    state_text: str,
    legal_actions: list[dict],
    phase: str,
    map_graph: dict | None = None,
    damage_note: bool = True,
    turn_math_lines: list[str] | None = None,
) -> str:
    """Render v4 combat text; noncombat remains byte-identical to the old arm."""
    if phase != "combat":
        return legacy.augment(
            state_text, legal_actions, phase, map_graph, damage_note, turn_math_lines,
        )
    statuses: set[str] = set()
    out_lines: list[str] = []
    for line in state_text.split("\n"):
        if legacy._ENEMY_LINE_RE.match(line):
            statuses |= _scan_status_names(line)
            line, refs = _label_intent(line)
            statuses |= refs
        elif line.startswith("Player powers:"):
            statuses |= _scan_status_names(line)
        out_lines.append(line)

    selection_note = _selection_note(state_text)
    # The old helper's playability inference is valid only during normal play.
    # An empty menu here suppresses that warning, retaining incoming arithmetic.
    notes = legacy._combat_notes(
        state_text, [] if selection_note else legal_actions, damage_note=damage_note,
    ).removeprefix("\n").splitlines()
    if turn_math_lines:
        insert_at = next(
            (i + 1 for i, line in enumerate(notes)
             if line.startswith("Incoming attack damage this turn:")),
            len(notes),
        )
        notes[insert_at:insert_at] = turn_math_lines
    if selection_note:
        notes.append(selection_note)
    notes.extend((_INTERACTION_NOTE, _CARD_VALUE_LEGEND))
    return "\n".join(out_lines + notes) + _build_key(state_text, statuses)
