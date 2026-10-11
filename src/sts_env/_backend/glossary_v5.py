"""Self-contained instance rules and public mechanics for combat_public_v5.

All calculations use already public text. Native execution and v3/v4 remain
unchanged. No draw order, RNG, sampled future intent, or search labels are read.
"""
from __future__ import annotations

import re

from sts_env._backend import glossary as legacy, glossary_v4 as v4


_PAIR = re.compile(r"(-?\d+) \((-?\d+)\)")
_SLIMES = {"SLIME_BOSS", "ACID_SLIME_L", "SPIKE_SLIME_L"}
_BLOCK = {"Defend": (5, 8), "Armaments": (5, 5), "Flame Barrier": (12, 16),
          "Ghostly Armor": (10, 13), "Good Instincts": (6, 9),
          "Impervious": (30, 40), "Panic Button": (30, 40),
          "Power Through": (15, 20), "Sentinel": (5, 8),
          "Shrug It Off": (8, 11), "True Grit": (7, 9), "Finesse": (2, 4),
          "Iron Wave": (5, 7)}

_ROUND_RULES = (
    "Energy A/B means A energy available now and B normal energy recharge per "
    "turn; B is not a cap. At the start of your turn, energy resets to B, except "
    "Ice Cream retains unused energy and adds B; other effects can add energy. "
    "Ordinary Block is removed at the start of your turn; Barricade or Blur "
    "preserves it, otherwise Calipers removes 15 instead. Unless an effect "
    "retains them, remaining hand cards are discarded at end of turn (Ethereal "
    "cards Exhaust); you normally draw the displayed draw-per-turn amount for "
    "the next turn. Listed player powers omit zero values: absent Strength or "
    "Dexterity means zero. A shown intent describes the upcoming enemy phase."
)


def card_parts(name: str) -> tuple[str, bool, int | None]:
    upgraded = bool(re.search(r"\+\d*$", name))
    plain = re.sub(r"\+\d*$", "", name)
    match = re.fullmatch(r"(.+)=(-?\d+)", plain)
    return (match[1], upgraded, int(match[2])) if match else (plain, upgraded, None)


def card_definition(name: str, relics: list[str]) -> str | None:
    base, upgraded, value = card_parts(name)
    original = v4._card_definition(name, relics)
    if original is None and base == "Ritual Dagger" and value is not None:
        original = "Ritual Dagger: Deal damage."
    if original is None:
        return None
    effect = original.split(": ", 1)[1]
    effect = _PAIR.sub(lambda m: m[2] if upgraded else m[1], effect)
    overrides = {
        "Armaments": "Gain 5 Block. Upgrade " + ("all cards in your hand" if upgraded else "a card in your hand") + " for the rest of combat.",
        "True Grit": f"Gain {9 if upgraded else 7} Block. Exhaust " + ("a chosen card" if upgraded else "a random card") + " from your hand.",
        "Heavy Blade": f"Deal 14 damage; Strength affects it {5 if upgraded else 3} times.",
        "Limit Break": "Double your Strength." + ("" if upgraded else " Exhaust."),
        "Blind": "Apply 2 Weak to " + ("ALL enemies." if upgraded else "one enemy.") + " Deals no damage.",
        "Trip": "Apply 2 Vulnerable to " + ("ALL enemies." if upgraded else "one enemy."),
        "Enlightenment": "Reduce the cost of cards in your hand to 1 " + ("for the rest of combat." if upgraded else "this turn."),
        "Whirlwind": f"X-cost: deal {8 if upgraded else 5} damage to ALL enemies, once per energy spent.",
    }
    effect = overrides.get(base, effect)
    # Costs already have per-instance/current values in every card label.
    effect = re.sub(r" \(costs? 0 when upgraded\)", "", effect)
    if base == "Rampage":
        bonus = value if value is not None else 0
        effect = (f"Deal {8 + bonus} unmodified damage (base 8 plus accumulated bonus {bonus}). "
                  f"Each play this combat adds {8 if upgraded else 5} to this card's bonus. "
                  "The =N in its name is the accumulated bonus, not total damage.")
    elif base == "Ritual Dagger" and value is not None:
        effect = (f"Deal {value} unmodified damage. If this kills a non-minion enemy, "
                  f"permanently increase this card's damage by {5 if upgraded else 3}. Exhaust. "
                  "The =N in its name is its current unmodified damage.")
    elif base == "Searing Blow":
        match = re.search(r"\+(\d*)$", name)
        level = int(match[1] or 1) if match else 0
        effect = f"Deal {12 + level * (level + 7) // 2} damage. Can be upgraded repeatedly."
    return f"{name}: {effect}"


def buff_effect(move: str, ascension: int) -> str | None:
    hallway = 2 if ascension >= 17 else 1 if ascension >= 2 else 0
    if move == "FUNGI_BEAST_GROW":
        return f"gains {(3, 4, 5)[hallway]} Strength"
    if move == "JAW_WORM_BELLOW":
        return f"gains {(3, 4, 5)[hallway]} Strength and {9 if ascension >= 17 else 6} Block"
    if move == "GREMLIN_NOB_BELLOW":
        return f"gains {3 if ascension >= 18 else 2} Enrage (each Skill you play then grants it that much Strength)"
    return None


def calculated_block(name: str, powers: dict[str, int]) -> int | None:
    base, upgraded, _ = card_parts(name)
    if base not in _BLOCK:
        return None
    amount = _BLOCK[base][int(upgraded)]
    # Mirrors BattleContext::calculateCardBlock; Iron Wave invokes it twice in
    # the installed simulator. This describes that behavior, not the retail game.
    for _ in range(2 if base == "Iron Wave" else 1):
        if powers.get("No Block", 0):
            amount = 0
        else:
            amount = max(0, amount + powers.get("Dexterity", 0))
            if powers.get("Frail", 0):
                amount = amount * 3 // 4
    return amount


def augment(state_text: str, legal_actions: list[dict], phase: str,
            map_graph: dict | None = None, damage_note: bool = True,
            turn_math_lines: list[str] | None = None) -> str:
    if phase != "combat":
        return v4.augment(state_text, legal_actions, phase, map_graph, damage_note, turn_math_lines)
    rendered = v4.augment(state_text, legal_actions, phase, map_graph, damage_note, turn_math_lines)
    head = rendered.split(legacy._KEY_HEADER, 1)[0].rstrip()
    asc_match = re.search(r"^Ascension: (\d+)$", state_text, re.MULTILINE)
    if asc_match is None:
        raise ValueError("v5 requires public ascension for exact enemy buff rules")
    ascension = int(asc_match[1])
    relics = legacy._relic_names(state_text)
    lines = head.splitlines()
    statuses: set[str] = set()
    slime_present = False
    for i, line in enumerate(lines):
        if legacy._ENEMY_LINE_RE.match(line):
            statuses |= v4._scan_status_names(line)
            move = v4._INTENT_RE.search(line)
            if move:
                effect = buff_effect(move[1], ascension)
                if effect:
                    line = re.sub(r"\(no damage; .*?\)(?=,|$)", f"(no damage; {effect})", line)
                    lines[i] = line
                statuses |= v4._status_names_in_text(effect or legacy.intent_effect(move[1]) or "")
            slime_present |= any(re.search(rf"\] {name} HP", line) for name in _SLIMES)
        elif line.startswith("Player powers:"):
            statuses |= v4._scan_status_names(line)
    # Replace the generic base/upgraded legend with per-instance semantics.
    lines = [line for line in lines if not line.startswith("Card rule notation:")]
    lines += [_ROUND_RULES,
              "Card rules below describe the named instance: + means upgraded. "
              "Rule damage/Block are unmodified unless explicitly calculated; "
              "inline deal N already includes current attack modifiers. Battle turns start at 0."]
    if slime_present:
        lines.append("Slime Boss and Large Acid/Spike Slimes: HP loss taking one to at or below half its maximum HP "
                     "changes its current intent to Split, replacing its previous move. At its enemy action, "
                     "it splits into two living offspring, each starting with the parent's current HP at the split. "
                     "Slime Boss creates one Large Acid and one Large Spike Slime; each large slime creates two medium slimes of its own type.")
    selection = re.search(r"^Selection type: (\S+)", state_text, re.MULTILINE)
    if selection and selection[1] == "EXHAUST_MANY":
        lines.append("EXHAUST_MANY means exhaust the selected hand cards. Exhausting no cards finishes this optional selection. Choose only a listed selection.")
    definitions = [d for name in legacy._combat_card_names(state_text)
                   if (d := card_definition(name, relics))]
    powers_line = next((line for line in lines if line.startswith("Player powers:")), "")
    powers = {name: int(m[1]) for name in ("Dexterity", "Frail", "No Block")
              if (m := re.search(rf"\b{name} (-?\d+)", powers_line))}
    if not selection:
        for name in legacy._hand_card_names(state_text):
            amount = calculated_block(name, powers)
            if amount is not None:
                lines.append(f"Current direct Block calculation for {name}: {amount} "
                             "(includes current Dexterity, Frail and No Block; excludes other card-play triggers, copies and later changes).")
    for name in relics:
        effect = v4._RELIC_ADDITIONS.get(name)
        definition = f"{name}: {effect}" if effect else legacy.relic_definition(name)
        if definition:
            definitions.append(definition)
    definitions += [d for name in legacy._potion_names(state_text) if (d := legacy.potion_definition(name))]
    # Explain referenced statuses before they activate, not only active icons.
    for definition in definitions:
        statuses |= v4._status_names_in_text(definition)
    status_definitions = [d for name in sorted(statuses) if (d := v4._status_definition(name, relics))]
    return "\n".join(lines) + "\n\n" + legacy._KEY_HEADER + "\n" + "\n".join(
        "  " + d for d in status_definitions + list(dict.fromkeys(definitions)))
