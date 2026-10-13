# Gemma combat template audit — 6 September 2026

The saved episodes contain real information gaps and misleading statements. This is an audit of the existing interface, not a measurement of how much fixing it would improve Gemma. No prompt, serializer, simulator, or sampled trajectory was changed by this audit.

Scope: **3,686 decision records across 202 trajectories**: 201 finished attempts (including invalid-action stops and a timeout) plus 23 decisions from the interrupted attempt. The sampled arm is `gemma4_t1_p095_k64_v1`, using `combat_public_v3`, the `action_text` contract, temperature 1.0, top-p 0.95, top-k 64, and thinking disabled. Archived attempts and preflight runs are excluded. Sampling remains paused.

Evidence directory: [`../outputs/gemma_template_audit_2026-09-06/`](../outputs/gemma_template_audit_2026-09-06/). It contains inventories, targeted checks, a source-file SHA-256 manifest, and 22 representative records with reconstructed full user/chat prompts, raw responses, and following records. The current prompting/glossary hashes match those recorded for the examples. Decision numbers below are zero-based; JSONL line = decision + 1.

## Missing or ambiguous information

### 1. The interaction does not explain action versus turn sequencing

The instruction says “Choose one legal action from the list,” but never explicitly says the model will receive another decision in the same turn after playing a card. Gemma visibly questions this in several episodes:

- `seed_4016_r0_d40`, decision 1: “But the prompt asks me to pick *one* legal action from the list. I cannot play two cards in one turn if the action doesn't lead to the next.”
- `seed_4040_r0_d55`, decision 4: “Since I can't "play 3 Defends" in one action, "play Demon Form" is the only one that represents a complete turn.” It ultimately selects Defend; this quote establishes confusion, not that it actually chose Demon Form.
- `seed_4064_r0_d3`, decision 10: questions whether combinations are allowed because the menu supplies single moves; this response ends in an invalid action.

These are three verified examples, not an exhaustive count of confused responses. The recent-action history provides indirect evidence of sequencing, but explicit interface instructions are missing.

Suggested repair: explain that each response executes one action; the model is queried again with the updated state, usually within the same turn; `end turn` yields to the enemy phase. Source: `src/sts_ai/prompting.py`, `NEUTRAL_FRAME` and `render_action_prompt`.

### 2. Parenthesized card values have no legend

The KEY says `Defend: Gain 5 (8) Block` and `Strike: Deal 6 (9) damage`. These mean base/upgraded values, but the user-facing text never explains that convention. Upgraded cards do have a `+` label: the upgrade flag itself is present.

In `seed_4094_r0_d77`, decision 34, Gemma tries to reconcile the parentheses with Dexterity and Strength, concluding: “6 is base, 9 is with strength.” That interpretation is wrong. Other responses correctly infer the notation, so it is not universally misunderstood.

Suggested repair: state explicitly that parentheses are upgraded base values and `+` means upgraded, or render the rule for the particular card instance. Keep those values distinct from the already-modified `(deal N)` action annotations. Source: `glossary.py`, `CARD_DB` and `_card_definition`.

### 3. Punctuation drops existing enemy-intent explanations

`_INTENT_RE` captures the next non-whitespace token, including its trailing comma. An enemy with powers can therefore produce:

```text
[0] SENTRY HP 30/40, block 0, intent SENTRY_BOLT, (no attack) Artifact 1
```

The lookup receives `SENTRY_BOLT,`, which fails, even though `INTENT_DB["SENTRY_BOLT"]` correctly says it adds two Dazed cards to the discard pile. “No attack” leaves out the actual consequence.

**763 records in 135 trajectories** contain an active intent whose existing dictionary explanation is absent anywhere in the state text. This is a conservative record-level count: if another enemy's annotation supplies the same explanation elsewhere, that effect is not counted as absent. It spans 29 intent types. Examples include Sentry Bolt (143 records/10 trajectories), Green Louse Spit Web (34/11), and Lagavulin Siphon Soul (41/8).

Example: `seed_4050_r0_d95`, decision 0. The evidence manifest also contains a minimal input/output reproduction. Source: `glossary.py`, `_INTENT_RE` and `_label_intent`.

Suggested repair: parse only the move identifier, preserving punctuation in the rendered line, with regression cases for nonattacking enemies that also have powers.

### 4. Gremlin Nob's Skull Bash omits its Vulnerable rider

This is separate from the punctuation bug: `GREMLIN_NOB_SKULL_BASH` has no intent definition. It occurs in **18 records across four trajectories**. The prompt reports attack damage but omits that it applies **2 Vulnerable**.

Example: `seed_4015_r0_d77`, decision 10, shows `GREMLIN_NOB_SKULL_BASH (deal 6), Enrage 2`. The simulator's `MonsterSpecific.cpp` Skull Bash case explicitly queues the attack and `DebuffPlayer<VULNERABLE>(2, true)`.

Suggested repair: add the rider to the intent glossary. Gemma kills Nob in this particular example, so this is evidence of absent information, not evidence that the omission caused damage there.

### 5. Several owned relics have no effect description

The ownership names are present, but these combat-relevant rules are absent from the KEY. Counts describe ownership exposure, not the number of times a trigger fired or affected the decision.

| Relic | Missing rule, checked against this simulator | Records / trajectories |
| --- | --- | ---: |
| Blue Candle | Allows playing a Curse, losing 1 HP and exhausting it | 180 / 7 |
| Captain's Wheel (`Captains Wheel`) | Gain 18 Block on the third turn | 12 / 1 |
| Charon's Ashes (`Charons Ashes`) | Deal 3 damage to all enemies when a card is exhausted | 64 / 3 |
| Gambling Chip | Opening-hand discard/redraw selection | 1 / 1 |
| Ginger | Prevents Weak | 103 / 3 |
| Incense Burner | Every six turns, gain 1 Intangible; the counter alone does not explain this | 20 / 1 |
| Ornamental Fan | Every third Attack played in a turn gives 4 Block | 78 / 4 |
| Paper Phrog | Enemy Vulnerable multiplies attack damage by 1.75 rather than 1.5 | 77 / 4 |
| Stone Calendar | Deal 52 damage to all enemies at the end of the seventh turn | 14 / 1 |
| Turnip | Prevents Frail | 99 / 4 |

For Paper Phrog, current `(deal N)` values already account for the relic, but the generic Vulnerable explanation still says 50%, and the modifier is unexplained when considering future plays. Blue Candle also conflicts with the blanket “Unplayable” curse descriptions.

The inventory contains seven other undefined relic names: Darkstone Periapt, Girya, Old Coin, Pantograph, Peace Pipe, Potion Belt, and Thread And Needle. I have **not** counted them as equally strong current-combat gaps: their effects are out of combat, already applied, or represented directly by other visible state. In particular, Thread And Needle's Plated Armor is shown and explained.

Suggested repair: cover the active combat relic rules and their interactions. Sources: `glossary.py` `RELIC_DB`; native `BattleContext.cpp`, `Player.cpp`, and `Player.h`. The inventory and evidence files give concrete episodes for each name.

### 6. Active Duplication has no definition once its potion is consumed

In `seed_4053_r0_d50`, decision 2, the prompt shows `Player powers: Duplication 1`, but the KEY does not define it. All potion slots are now empty, so the Duplication Potion description is no longer included. The previous action name is visible, but its rule is not.

This occurs in **one saved record**. Gemma chooses Defend; the next record has 10 Block rather than 5 and the power is consumed. The native implementation duplicates the next card played. The agent sends a fresh single-user-message prompt each decision, so it cannot rely on the potion explanation from an earlier conversation message.

Suggested repair: define active powers independently of whether their source item remains held. Sources: `glossary.py` `STATUS_DB`, `agents.py` `_apply_chat_template`, and the native Duplication handling.

## Incorrect or misleading supplied information

### 7. Flame Barrier+ has the wrong retaliation value

The KEY states “deal 4 damage to any enemy that attacks you” for both versions. The upgraded card actually grants **6** retaliation damage. Flame Barrier+ appears in hand in **four records across two trajectories**.

In `seed_4012_r0_d213`, decision 21, Gemma plays Flame Barrier+; decision 22 shows the resulting `Flame Barrier` power at 6. The native card implementation uses `up ? 6 : 4`. This is a wrong effect description, beyond the ambiguous parentheses in finding 2.

Suggested repair: include the upgraded retaliation value or render the instance-specific effect. Source: `glossary.py` `CARD_DB`; `BattleContext.cpp` Flame Barrier case.

### 8. “Not lethal this turn” can be false

TURN MATH combines attacks at their current costs and displayed damage. It does not search sequences involving skills, drawing, energy generation/refunds, or changing modifiers. Its maximum-damage line has a qualifier, but the subsequent “not lethal this turn” verdict is categorical.

There are **26 such verdicts in 17 trajectories where the actual subsequent trajectory wins before ending that same turn**. For example, `seed_4009_r0_d43`, decision 32, declares a 7-HP Mad Gremlin not lethal, then Gemma plays Seeing Red, Flex, and an 8-damage Strike and wins that turn.

This is a directly observed counterexample, not speculative search. Some winning continuations elsewhere involve draws; the result does not imply their cards could have been predicted. It does show the categorical negative is unjustified. Gemma successfully overrides it in these examples, so the count is not a count of induced mistakes.

Suggested repair: label this as damage from the currently considered attacks, with no general negative lethal verdict, or implement an appropriately scoped calculation. Source: `src/sts_ai/turn_math.py`.

### 9. End-turn damage projections omit visible effects

The projection subtracts current Block and Metallicize from incoming attacks, then says “you would take N damage.” It omits other visible end-turn effects. Three checked transitions:

| Episode / decision | Projection | Observed transition | Explanation |
| --- | ---: | --- | --- |
| `seed_4015_r0_d77` / 6 | 14 | HP 72 → 62: loss 10 | Plated Armor supplies 4 Block |
| `seed_4002_r0_d208` / 35 | 4 | HP 42 → 36: loss 6 | Burn in hand also deals 2 damage |
| `seed_4012_r0_d213` / 16 | 0 | HP 21 → 20: loss 1 | Combust costs 1 HP |

Those underlying cards/powers are visible and defined. The problem is a misleading summary, not absence of the raw rules. The automated scan found 62 candidate projection/HP-delta mismatches; **that is not a validated bug count**, because death clipping, healing, and other transition effects can also create differences.

Suggested repair: make the estimate explicitly limited to covered incoming attacks, or account for the relevant end-turn effects. Source: `turn_math.py` and `lightspeed.py` `turn_math_inputs`.

### 10. Card-selection prompts contain misleading warnings; Gambling Chip is particularly opaque

**129 of 132 selection records**, across 69 trajectories, show the warning that cards cannot be played because of insufficient energy or no playable cards. In these states the immediate reason is that a card-selection action must be resolved. Most selection menus still provide useful context; this is not a claim that all 129 choices are unintelligible.

The clearest problematic case is `seed_4129_r0_d213`, decision 0. It has 3 energy and cards costing 0–2, but the pending selection is labeled `GAMBLE`, the relic is unexplained, and the only action is the malformed-looking literal:

```text
{ GAMBLE none
```

Gemma returns `{"action":"GAMBLE none"}`, omitting the opening brace, and the attempt stops invalid. The exact-copy instruction **was present**: failing to copy it is a model error. Nevertheless, the label and missing selection semantics are concrete presentation defects. Whether the binding should expose additional discard subsets is a separate action-enumeration question; this audit does not establish that behavior.

Suggested repair: describe the pending selection and offer a readable “Keep opening hand / discard no cards” action for this choice; make the warning phase-aware. Sources: `glossary.py` `_combat_notes`; binding `describeCardSelectChoice` / `MULTI_CARD_SELECT` fallback in the maintained native patch.

## What the evidence does not justify claiming

- The scan found no encountered combat card name or occupied potion name lacking a glossary definition. That establishes name coverage, not correctness of every definition.
- Card types, upgrade labels, energy, Block, HP, target identities, attack damage, public unordered pile contents, potion slots, and current-turn action information are supplied. Mistakes involving these are not automatically missing-information bugs.
- A pure attack without an intent glossary entry can be adequately described by its displayed damage. I did not count every unfamiliar move identifier as a gap.
- Vanilla Slay the Spire knowledge is not sufficient evidence for this harness. For example, the inspected Hexaghost Inferno implementation does not implement the remembered Burn rider; I excluded that proposed omission.
- Hidden draw order, RNG state, future intents, and privileged search results should not be exposed as a repair.
- There is no causal estimate yet of how many losses or invalid responses these issues caused. Versioned repairs and a paired rerun on the same starting states would be needed to measure that.

The first repair priorities are intent parsing and coverage, the categorical TURN MATH statements, and the action/turn explanation. Then complete active relic/power definitions, correct card-instance effects, and clean up selection rendering. Preserve this sampled arm as the original baseline and version any changed observation/prompt interface.
