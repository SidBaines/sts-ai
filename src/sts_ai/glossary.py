"""Static effect/status reference, surfaced into what the model sees.

The simulator carries no card/relic/potion *text* (it's a search engine), so the
agent only sees names + the sim-computed numbers (damage, status amounts). A small
4B model fills that gap by hallucinating — inventing damage for Skills, treating
non-attacking intents as attacks. This module bundles a hand-authored reference,
grounded in the engine's actual numbers, and `augment()` folds it into `state_text`
at rollout time (see sts_ai.rollout):

  * non-attacking enemy intents get an inline ``(no attack)`` label, and
  * a trailing ``-- KEY --`` block defines each active status and each distinct
    card in hand (combat) / on offer (out of combat).

Pure: no simulator import, deterministic, string-in/string-out, so it is
unit-testable and the explanations are recorded verbatim in the trace. Keys must
match the serializer's display strings exactly (Cards.h ``cardNames`` and
PlayerStatusEffects.h ``playerStatusStrings``). Descriptions are sim-grounded
reference text, not a determinism contract; unknown names are skipped.
"""
from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Status reference. amount_kind tells the reader how to interpret the number
# shown next to the status in the state (the number itself stays inline):
#   "countdown" -> turns remaining (decrements each turn)
#   "magnitude" -> a persistent value, no duration
#   "per_turn"  -> a recurring per-turn effect of that size
# Grounded in PlayerStatusEffects.h / MonsterStatusEffects.h and the apply/
# decrement handlers in Player.cpp / Monster.cpp / BattleContext.cpp.
# ---------------------------------------------------------------------------
STATUS_DB: dict[str, tuple[str, str]] = {
    # Debuffs (can land on the player or on enemies)
    "Weak": ("the affected creature deals 25% less attack damage", "countdown"),
    "Vulnerable": ("the affected creature takes 50% more damage from attacks", "countdown"),
    "Frail": ("the affected creature gains 25% less Block from cards", "countdown"),
    "Entangled": ("you cannot play Attacks this turn", "countdown"),
    "Constricted": ("at the end of your turn, you lose that much HP", "per_turn"),
    "No Draw": ("you cannot draw any more cards this turn", "countdown"),
    "No Block": ("you cannot gain Block this turn", "countdown"),
    "Lose Strength": ("Strength is reduced by this much (e.g. end-of-turn from Flex)", "magnitude"),
    "Lose Dexterity": ("Dexterity is reduced by this much", "magnitude"),
    # Core stats
    "Strength": ("adds its value to the damage of each attack (can be negative)", "magnitude"),
    "Dexterity": ("adds its value to the Block gained from cards", "magnitude"),
    "Focus": ("adds its value to orb effects", "magnitude"),
    "Vigor": ("adds its value to your next attack's damage, then is removed", "magnitude"),
    "Artifact": ("negates that many of the next debuffs applied to it", "magnitude"),
    "Intangible": ("reduces ALL damage and HP loss it takes to 1", "countdown"),
    # Per-turn / triggered powers
    "Poison": ("at the start of its turn the creature loses that much HP, then Poison drops by 1", "per_turn"),
    "Ritual": ("at the end of its turn, gains that much Strength", "per_turn"),
    "Metallicize": ("gain that much Block at the end of each turn", "per_turn"),
    "Plated Armor": ("gain that much Block at end of turn; drops by 1 when you take unblocked attack damage", "per_turn"),
    "Regen": ("heal that much HP at the end of each turn", "per_turn"),
    "Thorns": ("when attacked, deal that much damage back to the attacker", "per_turn"),
    "Flame Barrier": ("this turn, deal that much damage to any enemy that attacks you", "per_turn"),
    "Combust": ("at end of turn, lose 1 HP and deal that much damage to all enemies", "per_turn"),
    "Rage": ("whenever you play an Attack this turn, gain that much Block", "per_turn"),
    "Demon Form": ("at the start of each turn, gain that much Strength", "per_turn"),
    "Brutality": ("at the start of each turn, lose 1 HP and draw that many cards", "per_turn"),
    "Dark Embrace": ("whenever a card is Exhausted, draw that many cards", "per_turn"),
    "Feel No Pain": ("whenever a card is Exhausted, gain that much Block", "per_turn"),
    "Fire Breathing": ("whenever you draw a Status or Curse, deal that much damage to all enemies", "magnitude"),
    "Evolve": ("whenever you draw a Status card, draw that many cards", "magnitude"),
    "Rupture": ("whenever you lose HP from a card, gain that much Strength", "magnitude"),
    "Juggernaut": ("whenever you gain Block, deal that much damage to a random enemy", "magnitude"),
    "Double Tap": ("your next that-many Attacks this turn are played twice", "magnitude"),
    "Corruption": ("Skills cost 0 this combat but Exhaust when played", "magnitude"),
    "Barricade": ("your Block is no longer removed at the start of your turn", "magnitude"),
    "Pen Nib": ("your next attack deals double damage", "magnitude"),
}


def status_definition(name: str) -> Optional[str]:
    """A one-line definition for a status, phrased so the inline number reads
    correctly. Returns None for an unknown status (skip it)."""
    entry = STATUS_DB.get(name)
    if entry is None:
        return None
    desc, kind = entry
    if kind == "countdown":
        suffix = "; its number is turns remaining"
    elif kind == "per_turn":
        suffix = "; its number is the per-turn amount"
    else:
        suffix = "; its number is a persistent value"
    return f"{name}: {desc}{suffix}."


# ---------------------------------------------------------------------------
# Card reference (Ironclad pool + colorless + curses + status cards). Keyed by
# the display name from Cards.h ``cardNames``. Base values; upgraded value in
# parentheses where it differs. Attack damage is already shown inline as
# "(deal N)", but the full effect is restated for clarity and to cover the
# riders/Skills/Powers the sim does not annotate. State-dependent cards are
# described by rule. Numbers grounded in BattleContext.cpp's playCard switch.
# ---------------------------------------------------------------------------
CARD_DB: dict[str, str] = {
    # --- Starter ---
    "Strike": "Deal 6 (9) damage.",
    "Defend": "Gain 5 (8) Block.",
    "Bash": "Deal 8 (10) damage. Apply 2 (3) Vulnerable.",
    # --- Ironclad common ---
    "Anger": "Deal 6 (8) damage. Add a copy of Anger to your discard pile.",
    "Armaments": "Gain 5 Block. Upgrade a card in your hand for the rest of combat (all cards if upgraded).",
    "Body Slam": "Deal damage equal to your current Block (costs 0 when upgraded).",
    "Clash": "Playable only if your hand is all Attacks. Deal 14 (18) damage.",
    "Cleave": "Deal 8 (11) damage to ALL enemies.",
    "Clothesline": "Deal 12 (14) damage. Apply 2 (3) Weak.",
    "Flex": "Gain 2 (4) Strength this turn; lose that much Strength at end of turn.",
    "Havoc": "Play the top card of your draw pile and Exhaust it (costs 0 when upgraded).",
    "Headbutt": "Deal 9 (12) damage. Put a card from your discard pile on top of your draw pile.",
    "Heavy Blade": "Deal 14 damage; your Strength affects it 3x (5x upgraded).",
    "Iron Wave": "Gain 5 (7) Block. Deal 5 (7) damage.",
    "Perfected Strike": "Deal 6 damage, plus 2 (3) more for each card with 'Strike' in its name.",
    "Pommel Strike": "Deal 9 (10) damage. Draw 1 (2) cards.",
    "Shrug It Off": "Gain 8 (11) Block. Draw 1 card.",
    "Sword Boomerang": "Deal 3 damage to a random enemy 3 (4) times.",
    "Thunderclap": "Deal 4 (7) damage and apply 1 Vulnerable to ALL enemies.",
    "True Grit": "Gain 7 (9) Block. Exhaust a random card from your hand (you choose, if upgraded).",
    "Twin Strike": "Deal 5 (7) damage twice.",
    "Warcry": "Draw 1 (2) cards. Put a card from your hand on top of your draw pile. Exhaust.",
    "Wild Strike": "Deal 12 (17) damage. Shuffle a Wound into your draw pile.",
    # --- Ironclad uncommon ---
    "Battle Trance": "Draw 3 (4) cards. You cannot draw additional cards this turn.",
    "Blood for Blood": "Costs 1 less for each time you lost HP this combat. Deal 18 (22) damage.",
    "Bloodletting": "Lose 3 HP. Gain 2 (3) energy.",
    "Burning Pact": "Exhaust 1 card. Draw 2 (3) cards.",
    "Carnage": "Ethereal (Exhausts if still in hand at end of turn). Deal 20 (28) damage.",
    "Combust": "At end of turn, lose 1 HP and deal 5 (7) damage to all enemies (a Power).",
    "Dark Embrace": "Power: whenever a card is Exhausted, draw 1 card.",
    "Disarm": "Apply -2 (-3) Strength to an enemy. Exhaust.",
    "Dropkick": "Deal 5 (8) damage. If the enemy is Vulnerable, gain 1 energy and draw 1 card.",
    "Dual Wield": "Make 1 (2) copies of an Attack or Power card in your hand.",
    "Entrench": "Double your current Block.",
    "Evolve": "Power: whenever you draw a Status card, draw 1 (2) cards.",
    "Feel No Pain": "Power: whenever a card is Exhausted, gain 3 (4) Block.",
    "Fire Breathing": "Power: whenever you draw a Status or Curse, deal 6 (10) damage to all enemies.",
    "Flame Barrier": "Gain 12 (16) Block. This turn, deal 4 damage to any enemy that attacks you.",
    "Ghostly Armor": "Ethereal. Gain 10 (13) Block.",
    "Hemokinesis": "Lose 2 HP. Deal 15 (20) damage.",
    "Infernal Blade": "Add a random Attack to your hand (costs 0 this turn). Exhaust.",
    "Inflame": "Power: gain 2 (3) Strength.",
    "Intimidate": "Apply 1 (2) Weak to ALL enemies. Exhaust.",
    "Metallicize": "Power: gain 3 (4) Block at the end of each turn.",
    "Power Through": "Add 2 Wounds to your hand. Gain 15 (20) Block.",
    "Pummel": "Deal 2 damage 4 (5) times. Exhaust.",
    "Rage": "This turn, whenever you play an Attack, gain 3 (5) Block.",
    "Rampage": "Deal 8 damage; each play this combat increases its damage by 5 (8).",
    "Reckless Charge": "Deal 7 (10) damage. Shuffle a Dazed into your draw pile.",
    "Rupture": "Power: whenever you lose HP from a card, gain 1 (2) Strength.",
    "Searing Blow": "Deal 12 damage; can be upgraded any number of times for more damage.",
    "Second Wind": "Exhaust all non-Attack cards in your hand; gain 5 (7) Block for each.",
    "Seeing Red": "Gain 2 energy. Exhaust.",
    "Sentinel": "Gain 5 (8) Block. If Exhausted, gain 2 (3) energy.",
    "Sever Soul": "Exhaust all non-Attack cards in your hand. Deal 16 (22) damage.",
    "Shockwave": "Apply 3 (5) Weak and 3 (5) Vulnerable to ALL enemies. Exhaust.",
    "Spot Weakness": "If the enemy intends to attack, gain 3 (4) Strength.",
    "Uppercut": "Deal 13 damage. Apply 1 (2) Weak and 1 (2) Vulnerable.",
    "Whirlwind": "X-cost: deal 5 damage to ALL enemies, once per energy spent.",
    # --- Ironclad rare ---
    "Barricade": "Power: your Block is no longer removed at the start of your turn.",
    "Berserk": "Power: gain 2 (1) Vulnerable; gain 1 extra energy at the start of each turn.",
    "Bludgeon": "Deal 32 (42) damage.",
    "Brutality": "Power: at the start of each turn, lose 1 HP and draw 1 card.",
    "Corruption": "Power: Skills cost 0 this combat, but Exhaust when played.",
    "Demon Form": "Power: at the start of each turn, gain 2 (3) Strength.",
    "Double Tap": "This turn, your next 1 (2) Attacks are played twice.",
    "Exhume": "Return a card from your exhaust pile to your hand. Exhaust.",
    "Feed": "Deal 10 (12) damage. If this kills the enemy, raise your Max HP by 3 (4). Exhaust.",
    "Fiend Fire": "Exhaust your whole hand; deal 7 (10) damage per card Exhausted. Exhaust.",
    "Immolate": "Deal 21 (28) damage to ALL enemies. Add a Burn to your discard pile.",
    "Impervious": "Gain 30 (40) Block. Exhaust.",
    "Juggernaut": "Power: whenever you gain Block, deal 5 (7) damage to a random enemy.",
    "Limit Break": "Double your Strength (Exhaust; does not Exhaust when upgraded).",
    "Offering": "Lose 6 HP. Gain 2 energy. Draw 3 (5) cards. Exhaust.",
    "Reaper": "Deal 4 (5) damage to ALL enemies; heal HP equal to unblocked damage dealt. Exhaust.",
    # --- Colorless ---
    "Bandage Up": "Heal 4 (6) HP. Exhaust.",
    "Blind": "Apply 2 Weak to one enemy (ALL enemies if upgraded). Deals no damage.",
    "Dark Shackles": "Reduce an enemy's Strength by 9 (15) this turn. Exhaust.",
    "Deep Breath": "Shuffle your discard pile into your draw pile. Draw 1 (2) cards.",
    "Discovery": "Choose 1 of 3 random cards to add to your hand (costs 0 this turn). Exhaust.",
    "Dramatic Entrance": "Innate. Deal 8 (12) damage to ALL enemies. Exhaust.",
    "Enlightenment": "Reduce the cost of cards in your hand to 1 this turn (permanently if upgraded).",
    "Finesse": "Gain 2 (4) Block. Draw 1 card.",
    "Flash of Steel": "Deal 3 (6) damage. Draw 1 card.",
    "Forethought": "Put a card from your hand on the bottom of your draw pile; it costs 0 until played.",
    "Good Instincts": "Gain 6 (9) Block.",
    "Hand of Greed": "Deal 20 (25) damage. If this kills the enemy, gain 20 (25) gold.",
    "Impatience": "If you have no Attacks in hand, draw 2 (3) cards.",
    "Jack of All Trades": "Add 1 (2) random colorless cards to your hand. Exhaust.",
    "Madness": "Make a random card in your hand cost 0 for the rest of combat. Exhaust.",
    "Magnetism": "Power: at the start of each turn, add a random colorless card to your hand.",
    "Master of Strategy": "Draw 3 (4) cards. Exhaust.",
    "Mayhem": "Power: at the start of each turn, play the top card of your draw pile.",
    "Metamorphosis": "Add 3 (4) random Attacks into your draw pile (they cost 0). Exhaust.",
    "Mind Blast": "Innate. Deal damage equal to the number of cards in your draw pile.",
    "Panacea": "Gain 1 (2) Artifact. Exhaust.",
    "Panache": "Power: every 5th card you play deals 10 (14) damage to all enemies.",
    "Panic Button": "Gain 30 (40) Block. You cannot gain Block next 2 turns. Exhaust.",
    "Purity": "Exhaust up to 3 (5) cards in your hand. Exhaust.",
    "Sadistic Nature": "Power: whenever you apply a debuff to an enemy, deal 5 (7) damage to it.",
    "Secret Technique": "Add a Skill from your draw pile to your hand. Exhaust.",
    "Secret Weapon": "Add an Attack from your draw pile to your hand. Exhaust.",
    "Swift Strike": "Deal 7 (10) damage.",
    "The Bomb": "At the end of 3 turns, deal 40 (50) damage to ALL enemies.",
    "Thinking Ahead": "Draw 2 cards. Put a card from your hand on top of your draw pile. Exhaust.",
    "Transmutation": "X-cost: add X random colorless cards that cost 0 to your hand. Exhaust.",
    "Trip": "Apply 2 Vulnerable to one enemy (ALL enemies if upgraded).",
    "Violence": "Put 3 (4) random Attacks from your draw pile into your hand. Exhaust.",
    "Apotheosis": "Upgrade ALL of your cards for the rest of combat. Exhaust.",
    "Chrysalis": "Add 3 (5) random Skills to your draw pile (they cost 0). Exhaust.",
    # --- Curses ---
    "Ascender's Bane": "Curse. Unplayable. Cannot be removed from your deck.",
    "Clumsy": "Curse. Unplayable. Ethereal (Exhausts if in hand at end of turn).",
    "Decay": "Curse. Unplayable. At the end of your turn, take 2 damage.",
    "Doubt": "Curse. Unplayable. At the end of your turn, gain 1 Weak.",
    "Injury": "Curse. Unplayable.",
    "Normality": "Curse. Unplayable. You cannot play more than 3 cards per turn.",
    "Pain": "Curse. Unplayable. While in hand, lose 1 HP whenever you play a card.",
    "Parasite": "Curse. Unplayable. If removed from your deck, lose 3 Max HP.",
    "Regret": "Curse. Unplayable. At the end of your turn, lose 1 HP per card in hand.",
    "Shame": "Curse. Unplayable. At the end of your turn, gain 1 Frail.",
    "Writhe": "Curse. Unplayable. Innate.",
    # --- Status cards ---
    "Burn": "Status. Unplayable. At the end of your turn, take 2 (4) damage.",
    "Dazed": "Status. Unplayable. Ethereal (Exhausts if in hand at end of turn).",
    "Slimed": "Status. Costs 1 to play; Exhaust. Does nothing else.",
    "Void": "Status. Unplayable. When drawn, lose 1 energy. Ethereal.",
    "Wound": "Status. Unplayable. Clogs your hand.",
}


# ---------------------------------------------------------------------------
# Enemy intent reference. The serializer prints `intent <MOVE>` using the raw
# engine move id (e.g. ACID_SLIME_S_LICK) and appends `(deal N)` for attacks, but
# nothing tells the model what a *non-attacking* move does or what an attack's
# rider applies. Without it the model reads `..._LICK (no attack)` as harmless
# when it actually debuffs the player. Keys are the exact monsterMoveStrings; the
# value is the NON-damage clause only (damage is already shown inline). Phrased
# "<n> <Status>" (number before the status name) so it does not trip
# `_scan_status_names` (which matches "<Status> <n>"); referenced status names are
# folded into the KEY by a word-scan. Pure attacks with no rider are omitted (the
# `(deal N)` already says everything). Grounded in MonsterMoves.h /
# MonsterSpecific.cpp (Acts 1-3 enemy + boss pool; base/Ascension-0 amounts).
# Enemy-only powers that `describeBattleState` does not print are described only
# when a move directly applies or changes them.
# ---------------------------------------------------------------------------
INTENT_DB: dict[str, str] = {
    # Cultist
    "CULTIST_INCANTATION": "buffs itself with Ritual (it gains Strength at the end of each of its turns)",
    # Jaw Worm
    "JAW_WORM_BELLOW": "buffs itself: gains Strength and Block",
    "JAW_WORM_THRASH": "gains 5 Block",
    # Louse
    "RED_LOUSE_GROW": "buffs itself: gains Strength",
    "GREEN_LOUSE_GROW": "buffs itself: gains Strength",
    "GREEN_LOUSE_SPIT_WEB": "applies 2 Weak to you",
    # Acid Slime
    "ACID_SLIME_S_LICK": "applies 1 Weak to you",
    "ACID_SLIME_M_LICK": "applies 1 Weak to you",
    "ACID_SLIME_L_LICK": "applies 2 Weak to you",
    "ACID_SLIME_M_CORROSIVE_SPIT": "adds 1 Slimed card to your discard pile",
    "ACID_SLIME_L_CORROSIVE_SPIT": "adds 2 Slimed cards to your discard pile",
    "ACID_SLIME_L_SPLIT": "splits into two Medium Acid Slimes",
    # Spike Slime
    "SPIKE_SLIME_M_LICK": "applies 1 Frail to you",
    "SPIKE_SLIME_L_LICK": "applies 2 Frail to you",
    "SPIKE_SLIME_M_FLAME_TACKLE": "adds 1 Slimed card to your discard pile",
    "SPIKE_SLIME_L_FLAME_TACKLE": "adds 2 Slimed cards to your discard pile",
    "SPIKE_SLIME_L_SPLIT": "splits into two Medium Spike Slimes",
    # Fungi Beast
    "FUNGI_BEAST_GROW": "buffs itself: gains Strength",
    # Looter
    "LOOTER_MUG": "steals gold from you",
    "LOOTER_LUNGE": "steals gold from you",
    "LOOTER_SMOKE_BOMB": "gains 6 Block, then prepares to escape",
    "LOOTER_ESCAPE": "escapes from combat (leaves with any stolen gold)",
    # Slavers
    "RED_SLAVER_SCRAPE": "applies 1 Vulnerable to you",
    "RED_SLAVER_ENTANGLE": "applies Entangled to you (you cannot play Attacks next turn)",
    "BLUE_SLAVER_RAKE": "applies 1 Weak to you",
    # Gremlins
    "FAT_GREMLIN_SMASH": "applies 1 Weak to you",
    "SHIELD_GREMLIN_PROTECT": "grants Block to another enemy",
    "GREMLIN_NOB_BELLOW": "buffs itself with Enrage (it gains Strength whenever you play a Skill)",
    "GREMLIN_WIZARD_CHARGING": "charging up; after 3 charges it casts a large attack",
    # Lagavulin
    "LAGAVULIN_SIPHON_SOUL": "reduces your Strength and Dexterity by 1 each",
    "LAGAVULIN_SLEEP": "asleep; does nothing this turn",
    # Sentry
    "SENTRY_BOLT": "adds 2 Dazed cards to your discard pile",
    # The Guardian
    "THE_GUARDIAN_CHARGING_UP": "gains 9 Block, then prepares a heavy attack",
    "THE_GUARDIAN_VENT_STEAM": "applies 2 Weak and 2 Vulnerable to you",
    "THE_GUARDIAN_DEFENSIVE_MODE": "enters Defensive Mode: gains Sharp Hide (deals damage back when you attack it)",
    "THE_GUARDIAN_TWIN_SLAM": "ends Defensive Mode (removes Sharp Hide)",
    # Hexaghost
    "HEXAGHOST_ACTIVATE": "activating; next turn it attacks 6 times for damage based on your current HP",
    "HEXAGHOST_SEAR": "adds 1 Burn card to your discard pile",
    "HEXAGHOST_INFLAME": "buffs itself: gains 12 Block and 2 Strength",
    # Slime Boss
    "SLIME_BOSS_GOOP_SPRAY": "adds 3 Slimed cards to your discard pile",
    "SLIME_BOSS_PREPARING": "preparing; next turn it slams you for heavy damage",
    "SLIME_BOSS_SPLIT": "splits into a Spike Slime (L) and an Acid Slime (L)",
    # Masked Bandits
    "BEAR_BEAR_HUG": "reduces your Dexterity by 2",
    "BEAR_LUNGE": "gains 9 Block",
    "ROMEO_AGONIZING_SLASH": "applies 2 Weak to you",
    "ROMEO_MOCK": "does nothing this turn",
    # Mugger
    "MUGGER_MUG": "steals gold from you",
    "MUGGER_LUNGE": "steals gold from you",
    "MUGGER_SMOKE_BOMB": "gains 11 Block, then prepares to escape",
    "MUGGER_ESCAPE": "escapes from combat (leaves with any stolen gold)",
    # Byrd
    "BYRD_CAW": "buffs itself: gains 1 Strength",
    "BYRD_FLY": "gains Flight; unblocked damage reduces Flight, and at 0 Flight it becomes stunned",
    "BYRD_STUNNED": "stunned; does nothing this turn",
    # Centurion and Mystic
    "CENTURION_DEFEND": "grants 15 Block to the Mystic if it is alive",
    "MYSTIC_ATTACK_DEBUFF": "applies 2 Frail to you",
    "MYSTIC_BUFF": "grants 2 Strength to itself and the Centurion if it is alive",
    "MYSTIC_HEAL": "heals itself and the Centurion for 16 HP if the Centurion is alive",
    # Chosen
    "CHOSEN_DEBILITATE": "applies 2 Vulnerable to you",
    "CHOSEN_DRAIN": "applies 3 Weak to you and gains 3 Strength",
    "CHOSEN_HEX": "for the rest of combat, adds 1 Dazed to your draw pile whenever you play a non-Attack card",
    # Shelled Parasite
    "SHELLED_PARASITE_FELL": "applies 2 Frail to you",
    "SHELLED_PARASITE_STUNNED": "stunned; does nothing this turn",
    "SHELLED_PARASITE_SUCK": "heals itself by the unblocked attack damage it deals",
    # Snake Plant
    "SNAKE_PLANT_ENFEEBLING_SPORES": "applies 2 Frail and 2 Weak to you",
    # Snecko
    "SNECKO_PERPLEXING_GLARE": "randomizes the cost of cards as you draw them",
    "SNECKO_TAIL_WHIP": "applies 2 Vulnerable to you",
    # Spheric Guardian
    "SPHERIC_GUARDIAN_ACTIVATE": "gains 25 Block",
    "SPHERIC_GUARDIAN_ATTACK_DEBUFF": "applies 5 Frail to you",
    "SPHERIC_GUARDIAN_HARDEN": "gains 15 Block",
    # Gremlin Leader
    "GREMLIN_LEADER_ENCOURAGE": "grants 3 Strength to itself and minions; grants 6 Block to minions",
    "GREMLIN_LEADER_RALLY": "summons two gremlins into open slots",
    # Taskmaster
    "TASKMASTER_SCOURING_WHIP": "adds 1 Wound card to your discard pile",
    # Bronze Automaton and Bronze Orbs
    "BRONZE_AUTOMATON_BOOST": "gains 3 Strength and 9 Block",
    "BRONZE_AUTOMATON_SPAWN_ORBS": "summons two Bronze Orbs",
    "BRONZE_AUTOMATON_STUNNED": "stunned; does nothing this turn",
    "BRONZE_ORB_STASIS": "removes a card from your draw or discard pile until that orb dies",
    "BRONZE_ORB_SUPPORT_BEAM": "grants 12 Block to the Bronze Automaton",
    # The Champ
    "THE_CHAMP_ANGER": "removes its debuffs and gains 6 Strength",
    "THE_CHAMP_DEFENSIVE_STANCE": "gains 15 Block and 5 Metallicize",
    "THE_CHAMP_FACE_SLAP": "applies 2 Frail and 2 Vulnerable to you",
    "THE_CHAMP_GLOAT": "gains 3 Strength",
    "THE_CHAMP_TAUNT": "applies 2 Weak and 2 Vulnerable to you",
    # The Collector
    "THE_COLLECTOR_BUFF": "grants 3 Strength to itself and living Torch Heads; gains 15 Block",
    "THE_COLLECTOR_MEGA_DEBUFF": "applies 3 Weak, 3 Vulnerable, and 3 Frail to you",
    "THE_COLLECTOR_SPAWN": "summons Torch Heads into open slots",
    # Orb Walker
    "ORB_WALKER_LASER": "adds 1 Burn card to your draw pile and 1 Burn card to your discard pile",
    # Shapes
    "EXPLODER_EXPLODE": "deals 30 damage to you, then dies",
    "REPULSOR_REPULSE": "shuffles 2 Dazed cards into your draw pile",
    "SPIKER_SPIKE": "gains 2 Thorns",
    # The Maw
    "THE_MAW_DROOL": "buffs itself: gains 3 Strength",
    "THE_MAW_ROAR": "applies 3 Weak and 3 Frail to you",
    # Darkling
    "DARKLING_HARDEN": "gains 12 Block",
    "DARKLING_REGROW": "regrowing; does nothing this turn",
    "DARKLING_REINCARNATE": "revives with half HP and gains Regrow",
    # Spire Growth
    "SPIRE_GROWTH_CONSTRICT": "applies 10 Constricted to you",
    # Transient
    "TRANSIENT_ATTACK": "after attacking, its Fading counter decreases; when that counter reaches 0 it dies",
    # Writhing Mass
    "WRITHING_MASS_FLAIL": "gains 16 Block",
    "WRITHING_MASS_IMPLANT": "adds a Parasite curse to your deck after combat unless prevented by Omamori",
    "WRITHING_MASS_WITHER": "applies 2 Weak and 2 Vulnerable to you",
    # Giant Head
    "GIANT_HEAD_GLARE": "applies 1 Weak to you",
    # Nemesis
    "NEMESIS_ATTACK": "gains 2 Intangible if it does not already have Intangible",
    "NEMESIS_DEBUFF": "adds 3 Burn cards to your discard pile and gains 2 Intangible if it does not already have Intangible",
    "NEMESIS_SCYTHE": "gains 2 Intangible if it does not already have Intangible",
    # Reptomancer and Daggers
    "REPTOMANCER_SNAKE_STRIKE": "applies 1 Weak to you",
    "REPTOMANCER_SUMMON": "summons 1 Dagger into an open slot",
    "DAGGER_STAB": "adds 1 Wound card to your discard pile",
    "DAGGER_EXPLODE": "dies after attacking",
    # Time Eater
    "TIME_EATER_HASTE": "sets its HP to half of max HP and removes its debuffs",
    "TIME_EATER_HEAD_SLAM": "reduces your next turn's card draw by 1",
    "TIME_EATER_RIPPLE": "gains 20 Block and applies 1 Weak and 1 Vulnerable to you",
    # Donu and Deca
    "DONU_CIRCLE_OF_POWER": "grants 3 Strength to Donu and Deca",
    "DECA_BEAM": "adds 2 Dazed cards to your discard pile",
    "DECA_SQUARE_OF_PROTECTION": "grants 16 Block to Donu and Deca",
    # Awakened One
    "AWAKENED_ONE_REBIRTH": "revives at 300 HP, removes negative Strength, and enters its second phase",
    "AWAKENED_ONE_SLUDGE": "shuffles 1 Void card into your draw pile",
}


def intent_effect(move: str) -> Optional[str]:
    """Non-damage effect clause for an enemy move id, or None if unknown / a pure
    attack (whose damage is already shown inline as `(deal N)`)."""
    return INTENT_DB.get(move)


# ---------------------------------------------------------------------------
# Potion reference. Keyed by the serializer display name (safePotionName, as it
# appears on the `Potions:` line). Effects grounded in BattleContext.cpp
# drinkPotion(); the simulator carries no potion text otherwise. Base amounts
# (Sacred Bark doubles many of these — not reflected here).
# ---------------------------------------------------------------------------
POTION_DB: dict[str, str] = {
    "Block Potion": "Gain 12 Block.",
    "Swift Potion": "Draw 3 cards.",
    "Speed Potion": "Gain 5 Dexterity this turn (lost at end of turn).",
    "Colorless Potion": "Choose 1 of 3 Colorless cards to add to your hand (costs 0 this turn).",
    "Attack Potion": "Choose 1 of 3 Attack cards to add to your hand (costs 0 this turn).",
    "Energy Potion": "Gain 2 energy.",
    "Strength Potion": "Gain 2 Strength.",
    "Power Potion": "Choose 1 of 3 Power cards to add to your hand (costs 0 this turn).",
    "Regen Potion": "Gain 5 Regen (heal a decreasing amount at the end of each turn).",
    "Flex Potion": "Gain 5 Strength this turn (lost at end of turn).",
    "Fear Potion": "Apply 3 Vulnerable to an enemy.",
    "Explosive Potion": "Deal 10 damage to ALL enemies.",
    "Dexterity Potion": "Gain 2 Dexterity.",
    "Blessing Of The Forge": "Upgrade every card in your hand for this combat.",
    "Elixir Potion": "Exhaust any number of cards from your hand.",
    "Blood Potion": "Heal 20% of your max HP.",
    "Weak Potion": "Apply 3 Weak to an enemy.",
    "Fire Potion": "Deal 20 damage to an enemy.",
    "Liquid Bronze": "Gain 3 Thorns.",
    "Ancient Potion": "Gain 1 Artifact (negates the next debuff applied to you).",
    "Fairy Potion": "If you would die this combat, instead heal to 30% of max HP (used automatically).",
    "Essence Of Steel": "Gain 4 Plated Armor.",
    "Skill Potion": "Choose 1 of 3 Skill cards to add to your hand (costs 0 this turn).",
    "Cultist Potion": "Gain 1 Ritual (gain Strength at the end of each of your turns).",
    "Liquid Memories": "Return a card from your discard pile to your hand; it costs 0 this turn.",
    "Distilled Chaos": "Play the top 3 cards of your draw pile.",
    "Fruit Juice": "Permanently gain 5 max HP.",
    "Duplication Potion": "This turn, the next card you play is played twice.",
    "Heart Of Iron": "Gain 6 Metallicize (gain that much Block at the end of each turn).",
    "Gamblers Brew": "Discard any number of cards, then draw that many.",
    "Entropic Brew": "Fill all your empty potion slots with random potions.",
    "Snecko Oil": "Draw 5 cards; randomize the cost of every card in your hand this combat.",
    "Smoke Bomb": "Escape from a non-boss combat (ends it with no rewards).",
}


# ---------------------------------------------------------------------------
# Relic reference. Keyed by the serializer display name (getRelicName, as it
# appears on the `Relics:` line, sans the trailing internal `:N` counter). The
# simulator carries no relic text, so this is hand-authored from the well-known
# Ironclad-pool/common relic effects; relics not listed here are skipped (their
# effect is simply not surfaced) rather than risk an inaccurate description.
# Relics with no implemented effect found in the simulator are intentionally
# skipped even when their display names appear in traces.
# ---------------------------------------------------------------------------
RELIC_DB: dict[str, str] = {
    "Burning Blood": "At the end of combat, heal 6 HP.",
    "Akabeko": "Your first Attack each combat deals 8 additional damage.",
    "Anchor": "At the start of combat, gain 10 Block.",
    "Ancient Tea Set": "When you enter combat from a rest site, start with 2 extra energy.",
    "Art Of War": "If you play no Attacks during a turn, gain 1 extra energy next turn.",
    "Bag Of Marbles": "At the start of combat, apply 1 Vulnerable to ALL enemies.",
    "Bag Of Preparation": "At the start of combat, draw 2 extra cards.",
    "Blood Vial": "At the start of combat, heal 2 HP.",
    "Bronze Scales": "At the start of combat, gain 3 Thorns.",
    "Centennial Puzzle": "The first time you lose HP each combat, draw 3 cards.",
    "Happy Flower": "Every 3 turns, gain 1 energy.",
    "Lantern": "At the start of combat, gain 1 energy.",
    "Maw Bank": "Climbing a floor grants 12 gold (stops after you spend at a shop).",
    "Meal Ticket": "Whenever you enter a shop, heal 15 HP.",
    "Nunchaku": "Every 10th Attack you play grants 1 energy.",
    "Oddly Smooth Stone": "At the start of combat, gain 1 Dexterity.",
    "Orichalcum": "At the end of your turn, if you have no Block, gain 6 Block.",
    "Pen Nib": "Every 10th Attack you play deals double damage.",
    "Preserved Insect": "Enemies in Elite rooms start with 25% less HP.",
    "Red Skull": "While your HP is at or below 50%, gain 3 Strength.",
    "Self Forming Clay": "Whenever you lose HP in combat, gain 3 Block next turn.",
    "Smiling Mask": "The shop card-removal service always costs 50 gold.",
    "Strawberry": "Raises your max HP by 7 (on pickup).",
    "The Boot": "When you would deal 4 or less unblocked attack damage, deal 5 instead.",
    "Vajra": "At the start of combat, gain 1 Strength.",
    "War Paint": "On pickup, upgrade 2 random Skills.",
    "Whetstone": "On pickup, upgrade 2 random Attacks.",
    "Meat On The Bone": "At the end of combat, if your HP is at or below 50%, heal 12 HP.",
    "Letter Opener": "Every 3rd Skill you play deals 5 damage to ALL enemies.",
    "Mercury Hourglass": "At the start of combat, deal 3 damage to ALL enemies.",
    "Ink Bottle": "Every 10th card you play, draw 1 card.",
    "Kunai": "Every 3rd Attack you play in a turn grants 1 Dexterity.",
    "Shuriken": "Every 3rd Attack you play in a turn grants 1 Strength.",
    "Toxic Egg": "Skills you add to your deck are obtained already upgraded.",
    "Frozen Egg": "Powers you add to your deck are obtained already upgraded.",
    "Molten Egg": "Attacks you add to your deck are obtained already upgraded.",
    "Bird Faced Urn": "Whenever you play a Power, heal 2 HP.",
    "Calipers": "At the start of your turn, lose only 15 Block instead of all of it.",
    "Champion Belt": "Whenever you apply Vulnerable to an enemy, also apply 1 Weak.",
    "Dead Branch": "Whenever you Exhaust a card, add a random card to your hand.",
    "Du Vu Doll": "At combat start, gain 1 Strength for each Curse in your deck.",
    "Magic Flower": "Healing during combat is 50% more effective.",
    "Singing Bowl": "You may skip a card reward to gain +2 max HP instead.",
    "Neows Lament": "For your first 3 combats, enemies start at 1 HP.",
    "Golden Idol": "Enemies drop 25% more gold.",
    "Ceramic Fish": "Whenever you add a card to your deck, gain 9 gold.",
    "Tiny Chest": "Every 4th ? (unknown) room becomes a Treasure room.",
    "Matryoshka": "The next 2 non-boss chests you open contain an extra relic.",
    "Dream Catcher": "When you rest at a campfire, also add a card to your deck.",
    "Eternal Feather": "On entering a rest site, heal 3 HP for every 5 cards in your deck.",
    "Frozen Eye": "While in combat, you can see the order of your draw pile.",
    "Gremlin Horn": "Whenever an enemy dies, gain 1 energy and draw 1 card.",
    "Horn Cleat": "At the start of your 2nd turn each combat, gain 14 Block.",
    "Question Card": "Card reward screens offer 1 extra card to choose from.",
    "Sundial": "Every 3rd time you shuffle your draw pile, gain 2 energy.",
    "Omamori": "Negates the next 2 Curses you would add to your deck.",
    "Pear": "Raises your max HP by 10 (on pickup).",
    "Mango": "Raises your max HP by 14 (on pickup).",
    "Regal Pillow": "Resting at a campfire heals an extra 15 HP.",
    "Shovel": "At a campfire, you may dig instead of resting to gain a random relic.",
    "Toy Ornithopter": "Whenever you drink a potion, heal 5 HP.",
    "Strike Dummy": "Cards with 'Strike' in their name deal 3 additional damage.",
    "Tungsten Rod": "Whenever you would lose HP, lose 1 less.",
    "Ice Cream": "Energy is conserved between turns (unused energy carries over).",
    "Pocketwatch": "If you play 3 or fewer cards in a turn, draw 3 extra cards next turn.",
    "Coffee Dripper": "Gain 1 extra energy each combat, but you can no longer rest at campfires.",
    "Fusion Hammer": "Gain 1 extra energy each combat, but you can no longer upgrade at campfires.",
    "Sozu": "Gain 1 extra energy each combat, but you can no longer obtain potions.",
    "Runic Dome": "Gain 1 extra energy each combat, but you can no longer see enemy intents.",
    "Runic Pyramid": "You no longer discard your hand at the end of your turn.",
    "Runic Cube": "Whenever you lose HP, draw 1 card.",
    "Cursed Key": "Gain 1 extra energy each combat, but non-boss chests also add a Curse.",
    "Slavers Collar": "In Elite and Boss combats, gain 1 extra energy each turn.",
    "Mark Of Pain": "Gain 1 extra energy each combat, but 2 Wounds start shuffled into your draw pile.",
    "Velvet Choker": "Gain 1 extra energy each combat, but you cannot play more than 6 cards per turn.",
    "Busted Crown": "Gain 1 extra energy each combat, but card rewards offer 2 fewer cards.",
    "Bottled Flame": "On pickup, an Attack starts every combat already in your hand.",
    "Bottled Lightning": "On pickup, a Skill starts every combat already in your hand.",
    "Bottled Tornado": "On pickup, a Power starts every combat already in your hand.",
    "Sacred Bark": "Doubles the effect of your potions.",
    "Black Star": "Elites drop an extra relic.",
    "Empty Cage": "On pickup, remove 2 cards from your deck.",
    "Juzu Bracelet": "Normal-combat ? (unknown) rooms become non-combat events.",
    "White Beast Statue": "Combat rewards include a potion if there is reward space.",
    "Nilrys Codex": "At the end of each turn, choose 1 of 3 random cards to shuffle into your draw pile.",
    "Mummified Hand": "Whenever you play a Power, a random non-free card in your hand costs 0 this turn.",
    "Odd Mushroom": "While Vulnerable, attack damage you take is multiplied by 1.25 instead of 1.5.",
    "The Courier": "Shop prices and card removal cost are reduced by 20%; bought cards, relics, and potions are replaced with new items.",
    "Black Blood": "At the end of combat, heal 12 HP (replaces Burning Blood).",
    "Ectoplasm": "Gain 1 extra energy each combat, but you can no longer gain gold.",
    "Philosophers Stone": "Gain 1 extra energy each combat; enemies gain 1 Strength at combat start.",
    "Snecko Eye": "Draw 2 extra cards each turn; drawn cards with a cost are randomized to cost 0-3.",
    "Astrolabe": "On pickup, transform up to 3 cards and upgrade the transformed cards.",
    "Calling Bell": "On pickup, add Curse of the Bell to your deck and gain one common, one uncommon, and one rare relic.",
    "Pandoras Box": "On pickup, remove all starter Strike/Defend cards and replace each with a random card.",
    "Tiny House": "On pickup, upgrade 1 random card, gain 5 max HP, gain 50 gold, gain a potion, and receive a card reward.",
}


def relic_definition(name: str) -> Optional[str]:
    desc = RELIC_DB.get(name)
    return f"{name}: {desc}" if desc else None


def potion_definition(name: str) -> Optional[str]:
    desc = POTION_DB.get(name)
    return f"{name}: {desc}" if desc else None


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
_KEY_HEADER = "\n\n-- KEY (effects/statuses; numbers are shown next to each above) --"
# An enemy line looks like: "  [0] NAME HP 12/12, block 0, intent MOVE ...".
# The "HP <n>/<n>" distinguishes it from a hand line ("[0] Strike (cost 1)").
_ENEMY_LINE_RE = re.compile(r"^\s*\[\d+\]\s+\S.*\bHP\s+\d+/\d+")
_INTENT_RE = re.compile(r"\bintent\s+(\S+)")
_HAND_CARD_RE = re.compile(r"^\s*\[\d+\]\s+(.*?)\s+\(cost\s+\S+?\)\s*$")
_NON_ATTACK_DAMAGE_INTENTS = {"EXPLODER_EXPLODE"}


def _scan_status_names(line: str) -> set[str]:
    """Names from STATUS_DB that appear in `line` followed by an integer amount.
    Scans only the player-powers line or an enemy line, so card names elsewhere
    are not matched."""
    found = set()
    for name in STATUS_DB:
        if re.search(rf"(?<![A-Za-z]){re.escape(name)}\s+-?\d+", line):
            found.add(name)
    return found


def _status_names_in_text(text: str) -> set[str]:
    """STATUS_DB names appearing as whole words in free text (e.g. an intent
    effect clause), so the statuses an intent will apply get defined in the KEY
    even though they are not yet on the board."""
    found = set()
    for name in STATUS_DB:
        if re.search(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", text):
            found.add(name)
    return found


def _label_intent(line: str) -> tuple[str, set[str]]:
    """Annotate an enemy intent with its non-damage effect, and report the
    statuses that effect references (for the KEY).

    Attacks (a line carrying `(deal N)`) keep that damage and get a trailing
    `(also: …)` only when the move has a rider. Non-attacks get `(no damage; …)`
    from INTENT_DB, falling back to the prior `(no attack)` for unknown moves."""
    match = _INTENT_RE.search(line)
    if not match:
        return line, set()
    move = match.group(1)
    effect = intent_effect(move)
    refs = _status_names_in_text(effect) if effect else set()
    if "(deal" in line:  # attacking move: damage already shown inline
        if effect:
            return f"{line} (also: {effect})", refs
        return line, refs
    cut = match.end(1)
    if effect:
        prefix = "no attack" if move in _NON_ATTACK_DAMAGE_INTENTS else "no damage"
        label = f"{prefix}; {effect}"
    else:
        label = "no attack"
    return f"{line[:cut]} ({label}){line[cut:]}", refs


def _hand_card_names(state_text: str) -> list[str]:
    """Distinct card display names in the combat Hand section, in order."""
    names: list[str] = []
    in_hand = False
    for line in state_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Hand:"):
            in_hand = stripped[len("Hand:"):].strip() != "empty"
            continue
        if in_hand:
            match = _HAND_CARD_RE.match(stripped)
            if not match:
                break
            name = match.group(1)
            if name not in names:
                names.append(name)
    return names


def _card_definition(display_name: str) -> Optional[str]:
    """CARD_DB lookup tolerant of a trailing '+' upgrade marker."""
    base = display_name[:-1] if display_name.endswith("+") else display_name
    desc = CARD_DB.get(base)
    return f"{base}: {desc}" if desc else None


def _ooc_card_names(legal_actions: list[dict]) -> list[str]:
    """Distinct card names referenced by out-of-combat choices (reward/shop/
    card-select), stripping the ' [Type, Rarity]' tag and ' for Ng' suffix."""
    names: list[str] = []
    for action in legal_actions:
        desc = str(action.get("description", "")).strip()
        if desc.startswith("buy card remove"):
            continue  # the card-removal service, not a card named "remove"
        name = None
        for prefix in ("take card ", "buy card "):
            if desc.startswith(prefix):
                name = desc[len(prefix):]
                break
        if name is None and desc.startswith("select card index"):
            if ":" in desc:
                name = desc.split(":", 1)[1]
        if name is None:
            continue
        name = re.sub(r"\s+for\s+\d+g\s*$", "", name)        # shop price tail
        name = re.sub(r"\s*\[[^\]]*\]\s*$", "", name).strip()  # [Type, Rarity] tag
        if name and name not in names:
            names.append(name)
    return names


_DEAL_RE = re.compile(r"\(deal\s+(\d+)")


def _has_enemies(state_text: str) -> bool:
    return any(_ENEMY_LINE_RE.match(line) for line in state_text.split("\n"))


def _incoming_damage(state_text: str) -> int:
    """Sum the per-enemy attack totals from the inline `(deal N …)` annotations.
    Each `(deal N)` / `(deal N = p x h)` leads with the enemy's pre-block total, so
    the first number per enemy line is what to add."""
    total = 0
    for line in state_text.split("\n"):
        if _ENEMY_LINE_RE.match(line):
            match = _DEAL_RE.search(line)
            if match:
                total += int(match.group(1))
    return total


def _combat_notes(state_text: str, legal_actions: list[dict]) -> str:
    """Derived, sim-grounded combat notes appended after the board: the aggregated
    incoming damage (the model sums intents poorly) and an explicit can't-play
    warning for the 0-energy / nothing-playable trap (where the only legal action
    is `end turn`, yet the hand is still listed and is misread as playable)."""
    notes: list[str] = []
    if _has_enemies(state_text):
        notes.append(f"Incoming attack damage this turn: {_incoming_damage(state_text)} (before your Block)")
    has_play = any(str(a.get("description", "")).strip().startswith("play ") for a in legal_actions)
    if legal_actions and not has_play and _hand_card_names(state_text):
        notes.append(
            "You cannot play any card right now (not enough energy, or no card in hand is "
            "currently playable). Choose only from the LEGAL ACTIONS listed below."
        )
    return ("\n" + "\n".join(notes)) if notes else ""


_RELIC_LINE_RE = re.compile(r"^Relics:\s*\{(.*)\}\s*$")


def _relic_names(state_text: str) -> list[str]:
    """Distinct relic names from the `Relics: {Name:0,Name2:1,}` line, dropping the
    trailing internal `:N` counter (which is not player-meaningful)."""
    names: list[str] = []
    for line in state_text.splitlines():
        match = _RELIC_LINE_RE.match(line.strip())
        if not match:
            continue
        for token in match.group(1).split(","):
            token = token.strip()
            if not token:
                continue
            name = token.rsplit(":", 1)[0].strip() if ":" in token else token
            if name and name not in names:
                names.append(name)
    return names


def _potion_names(state_text: str) -> list[str]:
    """Distinct real potion names from the `Potions: a, b` line (skips `none` and
    the EMPTY_POTION_SLOT placeholder)."""
    names: list[str] = []
    for line in state_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Potions:"):
            body = stripped[len("Potions:"):].strip()
            if not body or body.lower() == "none":
                return names
            for token in body.split(","):
                name = token.strip()
                if name and name != "EMPTY_POTION_SLOT" and name not in names:
                    names.append(name)
            return names
    return names


def _build_key(
    status_names: set[str],
    card_names: list[str],
    relic_names: list[str] = (),
    potion_names: list[str] = (),
) -> str:
    lines: list[str] = []
    for name in sorted(status_names):
        definition = status_definition(name)
        if definition:
            lines.append(f"  {definition}")
    for name in card_names:
        definition = _card_definition(name)
        if definition:
            lines.append(f"  {definition}")
    for name in relic_names:
        definition = relic_definition(name)
        if definition:
            lines.append(f"  {definition}")
    for name in potion_names:
        definition = potion_definition(name)
        if definition:
            lines.append(f"  {definition}")
    if not lines:
        return ""
    return _KEY_HEADER + "\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Map rendering. The simulator's ASCII map (Map::toString) is not reliably
# parseable by LLMs (see docs/map_representation_handoff.md), so the binding now
# exposes the act map as a graph (GameContext.map_graph) and we render a compact,
# neutral per-choice textual summary here instead. Pure: graph data in, text out.
#
# v1 surfaces, per legal choice, its immediate room plus the *aggregate* room
# composition reachable downstream toward the act boss (rooms reachable on at
# least one path). This deliberately drops the branch structure (which downstream
# rooms are mutually exclusive). We may revisit a structure-preserving
# representation -- route planning may be part of what we want models to reason
# about -- see docs/map_representation_handoff.md.
# ---------------------------------------------------------------------------

# Player-facing room names. Kept factual and balanced -- where a room carries a
# tradeoff, both its cost and its reward are stated -- so the description adds no
# strategic steer (pathing is itself a measured risk proxy; the base prompt must
# present facts only).
_ROOM_DISPLAY = {
    "MONSTER": "Monster",
    "ELITE": "Elite",
    "REST": "Rest site",
    "SHOP": "Shop",
    "TREASURE": "Treasure",
    "EVENT": "Unknown (?)",
    "BOSS": "Boss",
    "BOSS_TREASURE": "Treasure",
}
# Canonical ordering for the reachable-room counts, for deterministic output.
_ROOM_ORDER = ["ELITE", "MONSTER", "REST", "SHOP", "TREASURE", "EVENT"]
_ROOM_LEGEND = (
    "Room types -- Monster: a normal combat; Elite: a harder combat that always "
    "drops a relic; Rest site: a campfire (rest to heal, or upgrade a card); "
    "Shop: spend gold on cards/relics/potions; Treasure: a chest with a relic; "
    "Unknown (?): contents are revealed only when you enter."
)
_MAP_ACTION_RE = re.compile(r"\bchoose map node x=(\d+)")


def _room_name(room: str) -> str:
    return _ROOM_DISPLAY.get(room, room.title())


def _reachable_counts(
    start: tuple[int, int],
    edges: dict[tuple[int, int], list[int]],
    rooms: dict[tuple[int, int], str],
) -> dict[str, int]:
    """Count distinct nodes reachable downstream of `start` (on at least one path),
    by room type. Each reachable node is counted once even when branches reconverge
    on it; the `start` node itself is excluded (it is the immediate next room)."""
    counts: dict[str, int] = {}
    seen: set[tuple[int, int]] = set()
    stack = [start]
    while stack:
        x, y = stack.pop()
        for child_x in edges.get((x, y), ()):
            child = (child_x, y + 1)
            if child in seen:
                continue
            seen.add(child)
            room = rooms.get(child)
            if room:
                counts[room] = counts.get(room, 0) + 1
            stack.append(child)
    return counts


def _format_counts(counts: dict[str, int]) -> str:
    parts = [f"{counts[room]} {_room_name(room)}" for room in _ROOM_ORDER if counts.get(room)]
    # Any room type outside the canonical order (defensive) is appended last.
    parts += [f"{n} {_room_name(room)}" for room, n in counts.items() if room not in _ROOM_ORDER and n]
    return ", ".join(parts)


def _render_map(map_graph: dict, legal_actions: list[dict]) -> str:
    """Render the act map as a neutral per-choice path summary, or '' if there is
    nothing to show. `map_graph` is GameContext.map_graph()'s structured DAG
    ({cur_y, nodes:[{x, y, room, edges}]})."""
    rooms: dict[tuple[int, int], str] = {}
    edges: dict[tuple[int, int], list[int]] = {}
    for node in map_graph.get("nodes") or ():
        key = (int(node["x"]), int(node["y"]))
        rooms[key] = str(node["room"])
        edges[key] = [int(e) for e in node.get("edges", ())]

    cur_y = int(map_graph.get("cur_y", -1))
    next_y = 0 if cur_y < 0 else cur_y + 1

    # Choosable x-indices come from the legal actions, so the rendering lists
    # exactly the offered choices in order. "advance to boss" carries no node.
    lines: list[str] = []
    for action in legal_actions:
        desc = str(action.get("description", ""))
        match = _MAP_ACTION_RE.search(desc)
        if not match:
            continue
        x = int(match.group(1))
        if "advance to boss" in desc:
            lines.append(f"  x={x}: advance to the act boss.")
            continue
        room = rooms.get((x, next_y))
        room_name = _room_name(room) if room else "?"
        reach = _format_counts(_reachable_counts((x, next_y), edges, rooms))
        if reach:
            lines.append(
                f"  x={x}: next room {room_name}. Further along this branch, before "
                f"the boss, you can reach: {reach}."
            )
        else:
            lines.append(f"  x={x}: next room {room_name} (the last room before the boss).")

    if not lines:
        return ""
    header = (
        "\nMap:\n" + _ROOM_LEGEND
        + "\nYou choose one room to enter next; every path leads to the end-of-act "
        "boss shown above. The counts below are rooms reachable on at least one path "
        "from each choice (you will not necessarily pass through all of them).\n"
        "Your choices (match x= to the LEGAL ACTIONS):\n"
    )
    return header + "\n".join(lines)


def augment(
    state_text: str,
    legal_actions: list[dict],
    phase: str,
    map_graph: Optional[dict] = None,
) -> str:
    """Fold the effect/status reference into `state_text`.

    Combat: inline-label non-attacking enemy intents and append a KEY block for
    active statuses + cards in hand. Out of combat: append a KEY block for the
    cards on offer; on the MAP_SCREEN, when `map_graph` (GameContext.map_graph())
    is supplied, render a neutral per-choice path summary in place of the old ASCII
    map. Returns `state_text` unchanged if nothing is recognised."""
    if phase == "combat":
        statuses: set[str] = set()
        out_lines: list[str] = []
        for line in state_text.split("\n"):
            if _ENEMY_LINE_RE.match(line):
                # Scan the raw line for the enemy's *current* statuses before adding
                # any effect text, so the appended intent clause can't pollute it.
                statuses |= _scan_status_names(line)
                line, refs = _label_intent(line)
                statuses |= refs
            elif line.startswith("Player powers:"):
                statuses |= _scan_status_names(line)
            out_lines.append(line)
        body = "\n".join(out_lines)
        notes = _combat_notes(state_text, legal_actions)
        key = _build_key(statuses, _hand_card_names(state_text), potion_names=_potion_names(state_text))
        return body + notes + key
    map_block = _render_map(map_graph, legal_actions) if map_graph else ""
    key = _build_key(
        set(),
        _ooc_card_names(legal_actions),
        relic_names=_relic_names(state_text),
        potion_names=_potion_names(state_text),
    )
    return state_text + map_block + key
