"""Ascension-aware effect text over the frozen v5 renderer.

Grounding: patched MonsterSpecific.cpp::takeTurn, Acts 1-3. Only the current
public move and ascension are used; no future move, private counter or RNG.
"""
from __future__ import annotations

from functools import lru_cache

from sts_env._backend import glossary as legacy, glossary_v4 as v4, glossary_v5 as v5


@lru_cache(maxsize=21)
def _effects(ascension: int) -> dict[str, str]:
    effects = {}
    if ascension >= 2:
        effects.update({
            "CULTIST_INCANTATION": f"buffs itself with {5 if ascension >= 17 else 4} Ritual",
            "MYSTIC_BUFF": f"grants {4 if ascension >= 17 else 3} Strength to itself and the Centurion if it is alive",
            "WRITHING_MASS_FLAIL": "gains 18 Block",
        })
    if ascension >= 3:
        effects.update({
            "GREMLIN_LEADER_ENCOURAGE": f"grants {5 if ascension >= 18 else 4} Strength to itself and minions; grants 10 Block to minions",
            "NEMESIS_DEBUFF": "adds 5 Burn cards to your discard pile and gains 2 Intangible if it does not already have Intangible",
        })
    if ascension >= 4:
        effects.update({
            "BRONZE_AUTOMATON_BOOST": f"gains 4 Strength and {12 if ascension >= 9 else 9} Block",
            "THE_CHAMP_ANGER": f"removes its debuffs and gains {12 if ascension >= 19 else 9} Strength",
            "THE_CHAMP_GLOAT": f"gains {5 if ascension >= 19 else 4} Strength",
            "THE_COLLECTOR_BUFF": f"grants {5 if ascension >= 19 else 4} Strength to itself and living Torch Heads; gains {23 if ascension >= 19 else 18} Block",
        })
    if ascension >= 9:
        effects.update({
            "THE_CHAMP_DEFENSIVE_STANCE": f"gains {20 if ascension >= 19 else 18} Block and {7 if ascension >= 19 else 6} Metallicize",
            "AWAKENED_ONE_REBIRTH": "revives at 320 HP, removes negative Strength, and enters its second phase",
        })
    if ascension >= 17:
        effects.update({
            "BEAR_BEAR_HUG": "reduces your Dexterity by 4",
            "ROMEO_AGONIZING_SLASH": "applies 3 Weak to you",
            "BLUE_SLAVER_RAKE": "applies 2 Weak to you",
            "CENTURION_DEFEND": "grants 20 Block to the Mystic if it is alive",
            "MYSTIC_HEAL": "heals itself and the Centurion for 20 HP if the Centurion is alive",
            "FAT_GREMLIN_SMASH": "applies 1 Weak and 1 Frail to you",
            "MUGGER_SMOKE_BOMB": "gains 17 Block, then prepares to escape",
            "RED_LOUSE_GROW": "buffs itself: gains 4 Strength",
            "RED_SLAVER_SCRAPE": "applies 2 Vulnerable to you",
            "SNECKO_TAIL_WHIP": "applies 2 Vulnerable and 2 Weak to you",
            "SPHERIC_GUARDIAN_ACTIVATE": "gains 35 Block",
            "SPIKE_SLIME_L_LICK": "applies 3 Frail to you",
            "THE_MAW_DROOL": "buffs itself: gains 5 Strength",
            "THE_MAW_ROAR": "applies 5 Weak and 5 Frail to you",
            "DARKLING_HARDEN": "gains 12 Block and 2 Strength",
            "SPIRE_GROWTH_CONSTRICT": "applies 12 Constricted to you",
        })
    if ascension >= 3:
        effects["TASKMASTER_SCOURING_WHIP"] = (
            "adds 3 Wound cards to your discard pile and gains 1 Strength" if ascension >= 18
            else "adds 2 Wound cards to your discard pile")
    if ascension >= 18:
        effects.update({
            "LAGAVULIN_SIPHON_SOUL": "reduces your Strength and Dexterity by 2 each",
            "SENTRY_BOLT": "adds 3 Dazed cards to your discard pile",
            "REPTOMANCER_SUMMON": "summons 2 Daggers into open slots",
        })
    if ascension >= 19:
        effects.update({
            "HEXAGHOST_INFLAME": "buffs itself: gains 12 Block and 3 Strength",
            "HEXAGHOST_SEAR": "adds 2 Burn cards to your discard pile",
            "SLIME_BOSS_GOOP_SPRAY": "adds 5 Slimed cards to your discard pile",
            "THE_GUARDIAN_DEFENSIVE_MODE": "enters Defensive Mode: gains 4 Sharp Hide (deals damage back when you attack it)",
            "TIME_EATER_HASTE": "sets its HP to half of max HP, gains 32 Block and removes its debuffs",
            "TIME_EATER_HEAD_SLAM": "reduces your next turn's card draw by 1 and adds 2 Slimed cards to your discard pile",
            "TIME_EATER_RIPPLE": "gains 20 Block and applies 1 Weak, 1 Vulnerable and 1 Frail to you",
            "DECA_SQUARE_OF_PROTECTION": "grants 16 Block and 3 Plated Armor to Donu and Deca",
        })
    return effects


def augment(state_text: str, legal_actions: list[dict], phase: str,
            map_graph: dict | None = None, damage_note: bool = True,
            turn_math_lines: list[str] | None = None) -> str:
    rendered = v5.augment(state_text, legal_actions, phase, map_graph, damage_note, turn_math_lines)
    if phase != "combat":
        return rendered
    # v5 already validates the presence of the public ascension line.
    ascension = int(v5._ASCENSION_RE.search(state_text)[1])
    effects = _effects(ascension)
    if not effects and "Ascender's Bane" not in rendered:
        return rendered
    lines = rendered.splitlines()
    referenced = set()
    for index, line in enumerate(lines):
        if legacy._ENEMY_LINE_RE.match(line):
            match = v4._INTENT_RE.search(line)
            effect = effects.get(match[1]) if match else None
            if effect:
                old = v4._INTENT_ADDITIONS.get(match[1]) or legacy.intent_effect(match[1])
                # Replace the known clause, not an inferred attack type:
                # Guardian's non-attack text itself contains "(deals damage".
                line = line.replace(f"(also: {old})", f"(also: {effect})")
                line = line.replace(f"(no damage; {old})", f"(no damage; {effect})")
                lines[index] = line
                referenced |= v4._status_names_in_text(effect)
        elif line.startswith("  Ascender's Bane:"):
            lines[index] += " Innate: starts in your opening hand. Ethereal: Exhausts at the end of your turn if still in hand."
    # Add definitions for effects absent from the base-ascension move, e.g.
    # Frail from Time Eater. Do not duplicate the existing v5 definitions.
    relics = legacy._relic_names(state_text)
    definitions = [definition for name in sorted(referenced)
                   if not any(line.startswith(f"  {name}:") for line in lines)
                   and (definition := v4._status_definition(name, relics))]
    if definitions:
        index = lines.index(legacy._KEY_HEADER.strip()) + 1
        lines[index:index] = ["  " + definition for definition in definitions]
    return "\n".join(lines)
