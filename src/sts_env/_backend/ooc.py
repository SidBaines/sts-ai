from __future__ import annotations

from typing import Any

# Descriptions reflect the pinned simulator's implemented effects, not assumed
# vanilla-game mechanics; these include the event fixes carried by
# patches/sts_lightspeed_python_api.patch.
_STATIC: dict[str, dict[int, str]] = {
    "OMINOUS_FORGE": {0: "upgrade a card", 1: "gain Warped Tongs and the Pain curse", 2: "leave"},
    "PLEADING_VAGRANT": {0: "pay 85 gold for a random relic", 1: "gain a random relic and the Shame curse", 2: "leave"},
    "ANCIENT_WRITING": {0: "remove a card", 1: "upgrade all starter Strikes and Defends"},
    "OLD_BEGGAR": {0: "pay 75 gold to remove a card", 1: "leave"},
    "BIG_FISH": {0: "heal one third of maximum HP", 1: "gain 5 maximum HP and heal 5 HP", 2: "gain a random relic and the Regret curse"},
    "COLOSSEUM": {0: "leave (combat event is not implemented by this simulator)"},
    "AUGMENTER": {0: "gain J.A.X.", 1: "transform two cards", 2: "gain Mutagenic Strength"},
    "DUPLICATOR": {0: "duplicate a card", 1: "leave"},
    "THE_DIVINE_FOUNTAIN": {0: "remove removable curses from your deck", 1: "leave"},
    "GOLDEN_SHRINE": {1: "gain 275 gold and the Regret curse", 2: "leave"},
    "WING_STATUE": {0: "take 7 damage and remove a card", 1: "gain 50–80 gold", 2: "leave"},
    "THE_SSSSSERPENT": {1: "leave"},
    "LIVING_WALL": {0: "remove a card", 1: "transform a card", 2: "upgrade a card"},
    "MASKED_BANDITS": {0: "give up all your gold", 1: "fight the Masked Bandits; victory rewards include Red Mask"},
    "MINDBLOOM": {0: "fight a random Act 1 boss for a rare relic and combat rewards", 1: "upgrade all upgradeable cards and gain Mark of the Bloom (prevents healing)", 2: "gain 999 gold and two Normality curses", 3: "heal to full HP and gain the Doubt curse"},
    "HYPNOTIZING_COLORED_MUSHROOMS": {0: "fight the Mushrooms; victory rewards include Odd Mushroom"},
    "MYSTERIOUS_SPHERE": {0: "fight the Mysterious Sphere encounter for a rare relic and combat rewards", 1: "leave"},
    "THE_NEST": {1: "take 6 damage and gain Ritual Dagger"},
    "NLOTH": {2: "leave"},
    "NOTE_FOR_YOURSELF": {0: "obtain the stored card and choose a card to remove", 1: "leave"},
    "PURIFIER": {0: "remove a card", 1: "leave"},
    "SCRAP_OOZE": {1: "leave"},
    "SECRET_PORTAL": {0: "travel directly to the Act 3 boss", 1: "leave"},
    "SENSORY_STONE": {0: "choose from one colorless card reward", 1: "lose 5 HP and choose from two colorless card rewards", 2: "lose 10 HP and choose from three colorless card rewards", 3: "choose from four colorless card rewards (simulator option)"},
    "SHINING_LIGHT": {1: "leave"},
    "THE_CLERIC": {2: "leave"},
    "THE_JOUST": {0: "pay 50 gold; 70% chance to gain 100 gold", 1: "pay 50 gold; 30% chance to gain 250 gold"},
    "THE_LIBRARY": {0: "choose one card from 20 offered cards"},
    "THE_MAUSOLEUM": {1: "leave"},
    "THE_MOAI_HEAD": {1: "give up Golden Idol and gain 333 gold", 2: "leave"},
    "THE_WOMAN_IN_BLUE": {0: "pay 20 gold for one random potion", 1: "pay 30 gold for two random potions", 2: "pay 40 gold for three random potions"},
    "TOMB_OF_LORD_RED_MASK": {0: "gain 222 gold", 1: "give up all gold and gain Red Mask", 2: "leave"},
    "TRANSMORGRIFIER": {0: "transform a card", 1: "leave"},
    "UPGRADE_SHRINE": {0: "upgrade a card", 1: "leave"},
    "VAMPIRES": {0: "give up Blood Vial; replace starter Strikes with five Bites", 2: "leave"},
    "WE_MEET_AGAIN": {3: "leave"},
    "WHEEL_OF_CHANGE": {0: "spin the wheel: random gold, relic, full heal, Decay curse, card removal, or HP loss"},
    "WORLD_OF_GOOP": {0: "take 11 damage and gain 75 gold"},
}


def describe_event_action(event: dict[str, Any], index: int, *, ascension: int,
                          max_hp: int, act: int) -> str | None:
    """Public event option effects, without revealing pre-sampled hidden rolls.

    ``event`` comes from the binding's public_event() projection. Only initialized
    event-specific fields may enter it. Unknown events/options return None so
    callers can explicitly label incomplete information instead of inventing it.
    """
    name = event.get("event", "")
    unfavorable = ascension >= 15
    description = _STATIC.get(name, {}).get(index)
    hp0, hp1, hp2 = (event.get(f"hp_amount{i}", "the displayed amount of") for i in range(3))
    if name == "CURSED_TOME":
        descriptions = {0: "read the tome", 1: "leave", 2: f"continue reading; lose {event.get('event_data', 1)} HP", 3: f"continue reading; lose {event.get('event_data', 2)} HP", 4: f"continue reading; lose {event.get('event_data', 3)} HP", 5: f"lose {15 if unfavorable else 10} HP and receive a random book relic", 6: "leave; lose 3 HP"}
        description = descriptions.get(index)
    elif name == "DEAD_ADVENTURER" and index == 0:
        chance = int(event.get("phase", 0)) * 25 + (35 if unfavorable else 25)
        description = f"search for loot; {chance}% chance of an elite fight"
    elif name == "DEAD_ADVENTURER" and index == 1:
        description = "leave"
    elif name == "DESIGNER_IN_SPIRE":
        description = {
            0: f"pay {50 if unfavorable else 40} gold to upgrade a card",
            1: f"pay {50 if unfavorable else 40} gold to upgrade two random cards",
            2: f"pay {75 if unfavorable else 60} gold to remove a card",
            3: f"pay {75 if unfavorable else 60} gold to transform two random cards",
            4: f"pay {110 if unfavorable else 90} gold to remove a card, then upgrade a random card",
            5: f"leave; lose {5 if unfavorable else 3} HP",
        }.get(index)
    elif name == "FACE_TRADER" and index == 0:
        description = f"take {hp0} damage and gain {50 if unfavorable else 75} gold"
    elif name == "FACE_TRADER":
        description = {1: "trade for a random face relic", 2: "leave"}.get(index)
    elif name == "FALLING":
        description = {0: f"lose {event.get('falling_skill_card', 'the offered Skill card')}", 1: f"lose {event.get('falling_power_card', 'the offered Power card')}", 2: f"lose {event.get('falling_attack_card', 'the offered Attack card')}", 3: "leave"}.get(index)
    elif name == "FORGOTTEN_ALTAR":
        description = {0: "exchange Golden Idol for Bloody Idol", 1: f"gain 5 maximum HP and heal 5 HP, then lose {hp0} HP", 2: "gain the Decay curse"}.get(index)
    elif name == "GHOSTS":
        description = {0: f"lose {hp0} maximum HP and gain {3 if unfavorable else 5} Apparitions", 1: "leave"}.get(index)
    elif name == "GOLDEN_IDOL":
        description = {0: "take Golden Idol; then choose its consequence", 1: "leave", 2: "gain the Injury curse", 3: f"take {hp0} damage", 4: f"lose {hp1} maximum HP"}.get(index)
    elif name == "GOLDEN_SHRINE" and index == 0:
        description = f"gain {50 if unfavorable else 100} gold"
    elif name == "KNOWING_SKULL":
        description = {0: f"lose {hp0} HP and gain 90 gold", 1: f"lose {hp1} HP and gain a random uncommon colorless card", 2: f"lose {hp2} HP and gain a random potion", 3: "leave; lose 6 HP"}.get(index)
    elif name == "THE_SSSSSERPENT" and index == 0:
        description = f"gain {150 if unfavorable else 175} gold and the Doubt curse"
    elif name == "HYPNOTIZING_COLORED_MUSHROOMS" and index == 1:
        description = f"heal {max_hp // 4} HP and gain the Parasite curse"
    elif name == "THE_NEST" and index == 0:
        description = f"gain {50 if unfavorable else 99} gold"
    elif name == "NLOTH" and index in {0, 1}:
        description = f"give up {event.get(f'relic{index}', 'the offered relic')} and gain N'loth's Gift"
    elif name == "SCRAP_OOZE" and index == 0:
        chance = int(event.get("event_data", 0)) * 10 + 25
        description = f"take {5 if unfavorable else 3} damage; {min(chance + 1, 100)}% chance to obtain a random relic (implemented simulator probability)"
    elif name == "SHINING_LIGHT" and index == 0:
        description = f"take {hp0} damage and upgrade up to two random cards"
    elif name == "THE_CLERIC":
        description = {0: f"pay 35 gold and heal {hp0} HP", 1: f"pay {75 if unfavorable else 50} gold to remove a card", 2: "leave"}.get(index)
    elif name == "THE_LIBRARY" and index == 1:
        description = f"heal {hp0} HP"
    elif name == "THE_MAUSOLEUM" and index == 0:
        description = ("open the coffin; gain a random relic and the Writhe curse" if unfavorable
                       else "open the coffin; gain a random relic and 50% chance of the Writhe curse")
    elif name == "THE_MOAI_HEAD" and index == 0:
        description = f"lose {hp0} maximum HP and heal to full"
    elif name == "THE_WOMAN_IN_BLUE" and index == 3:
        description = f"leave; lose {hp0} HP" if unfavorable else "leave"
    elif name == "VAMPIRES" and index == 1:
        description = f"lose {hp0} maximum HP; replace starter Strikes with five Bites"
    elif name == "WE_MEET_AGAIN":
        description = {0: f"give up {event.get('potion', 'the offered potion')} for a random relic", 1: f"give up {event.get('gold', 'the offered amount of')} gold for a random relic", 2: f"give up {event.get('card', 'the offered card')} for a random relic", 3: "leave"}.get(index)
    elif name == "WINDING_HALLS":
        description = {0: f"lose {hp0} HP and gain two Madness cards", 1: f"heal {hp1} HP and gain Writhe", 2: f"lose {hp2} maximum HP"}.get(index)
    elif name == "WORLD_OF_GOOP" and index == 1:
        description = f"leave; lose {event.get('gold_loss', 'the displayed amount of')} gold"
    if description is None:
        return None
    return f"{name.replace('_', ' ').title()}: {description}"
