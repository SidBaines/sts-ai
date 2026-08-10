# Human-public combat state contract

**Contract identifier:** `combat_public_v2`  
**Status:** implemented and cohort-validated; teacher dataset migration in progress  
**Last updated:** 2026-07-23

`combat_public_v1` is retained only for artifact replay. It is rejected for new
competence evaluation or training because it omitted card type in combat. During
the stopped COMP-002 run, Gemma explicitly classified Strike/Wild Strike as
Skills and Defend as not triggering Gremlin Nob's Enrage. Existing v1 model
rollouts and teacher datasets are diagnostics, not valid competence evidence.

## Purpose

This document defines the combat observation that may be shown to the policy. It
is the boundary between simulator state and model input: the policy should receive
everything an attentive human player can inspect in Slay the Spire, but it must
not receive shuffle order, future randomness, search values, or other simulator
internals.

The contract is deliberately independent of the exact prose layout. A formatting
change that preserves all fields still changes the policy interface and must be
versioned or hashed, but it does not change what information is allowed. The
implementation keeps the legacy and v1 state text available behind explicit
version switches while v2 adds compact, stable card-type labels.

## Normative rules

The words **must**, **must not**, and **should** are requirements for
`combat_public_v2`.

1. The observation must be a pure read of the pending player decision. Rendering
   it twice must not mutate the battle or change legal actions.
2. Every collection whose underlying C++ representation has a hidden order must
   be sorted or aggregated before exposure. In particular, the draw pile is a
   multiset, never a sequence.
3. Card identity must include upgrade state. Cards whose public behaviour depends
   on an additional instance value must include that public value as well.
   Card identity must also include the human-visible gameplay type: Attack,
   Skill, Power, Status, or Curse. Rarity is not required in combat.
4. Dynamic values shown to the player, such as current energy cost and enemy
   intent damage, take precedence over base card/enemy definitions.
5. The observation and the legal-action menu describe the same pending decision.
   Card upgrade markers and targets must agree in both representations.
6. Search output is teacher provenance, not policy state. It must never be added
   to this observation.
7. The contract is additive to the existing fail-closed action parser. It does not
   authorize a model to emit simulator bit fields, card indices, or enemy indices;
   the policy still chooses a displayed `action_index`.

## Required fields

### Encounter and turn

- Encounter/enemy identities and the public turn number.
- The current input or selection state.
- Public counters for the current turn: cards played, attacks played, Skills
  played, and cards discarded, when maintained by the simulator.
- Actions already executed during the current turn in displayed, human-readable
  form. This history resets when the turn changes and when combat ends. It is
  tracked by the Python harness because the simulator does not retain an action
  log.

The action history is observational memory, not a suggested plan. Plan persistence
is a separate experiment and must use a separately versioned prompt arm.

### Player

- Current and maximum HP.
- Current block.
- Current energy and maximum energy for the turn.
- Current stance, normal draw-per-turn value, and orb-slot count.
- Public powers, statuses, and their visible amounts.
- Any other combat counter that the game exposes directly and that changes the
  meaning of a legal action.

### Enemies

For every enemy, in stable slot order:

- identity, alive/targetable state, current and maximum HP, and block;
- displayed intent, including hit count and the simulator's visible post-modifier
  damage where applicable;
- public powers and statuses with visible amounts.

Enemy move RNG, future moves beyond the displayed intent, and AI-internal move
history must not be exposed unless the base game makes them visible.

### Hand and pending selection

For every card in hand:

- displayed card name with upgrade state;
- gameplay type (`Attack`, `Skill`, `Power`, `Status`, or `Curse`);
- current cost for this decision, including cost-for-turn changes and `X` cost;
- compact public effect text or a glossary entry sufficient to interpret it.

If a card-selection input is pending, the observation must include the selection
kind, the allowed number of choices, and all selectable public candidates with
upgrade state. Candidate indices may appear in the observation for readability,
but policy output continues to use the displayed legal-action index only.

### Card piles

- Draw-pile size and an unordered multiset of its publicly known card instances.
- Discard-pile size and complete contents.
- Exhaust-pile size and complete contents.

Repeated identical instances should be aggregated as `N x Card`; sorting must be
deterministic. Upgrade state is part of identity, so `Strike` and `Strike+` are
different multiset entries. The legacy line
`Piles: draw N, discard N, exhaust N` remains present for parser compatibility.

The implementation must not expose vector position, top/bottom-of-pile order, a
shuffle permutation, or RNG state. If a game effect makes one position publicly
known, that fact requires an explicit future contract revision rather than an
implicit leak through container order.

### Relics

- Complete owned relic inventory.
- Public relic counters and readiness state that a human can inspect, including
  combat-local counter changes.

Relic ownership comes from the run state copied into combat, not merely from a set
of relic-effect bits: a bitset is an implementation detail and may omit relics
whose effects are not active at the current instant.

### Potions

- Potion slots in stable slot order, including empty slots.
- Potion identity and any public usability/target context needed for the current
  decision.
- Targeted potion actions must disambiguate living enemies with the same name in
  exactly the same way as targeted card actions.

## Explicitly forbidden fields

The policy observation must not contain:

- draw-pile or discard-pile container order;
- the future shuffle permutation;
- simulator, encounter, card, or enemy RNG state/seeds;
- future random outcomes (targets, generated cards, damage rolls, and so on);
- an enemy's unshown next move;
- search visits, values, rollouts, recommended actions, predicted HP, or winning
  sequences;
- C++ addresses, enum numeric values with hidden significance, event queue
  contents, or other engine-only state;
- privileged state hashes from which hidden state can be recovered.

A public-observation hash may be supplied in training metadata, but it must be
computed from the rendered/structured public observation only.

## Stable representation

The text serializer should retain the compact legacy player, enemy, hand, pile
count, and potion lines, then append stable public sections. Exact labels are
implementation-defined until the first snapshot is frozen, subject to these
rules:

- collections are deterministically sorted or retain only a genuinely visible
  slot order;
- empty piles and empty relic/power sets are represented explicitly;
- no field is omitted merely because its numeric value is zero if zero and absent
  have different game meanings;
- card names use one canonical upgrade notation throughout state, actions, and
  selections;
- card type is rendered immediately after the canonical name as `[Attack]`,
  `[Skill]`, `[Power]`, `[Status]`, or `[Curse]` in hands, public piles, card-play
  actions, and card-selection candidates;
- structured Python summaries contain the same information as the text used for
  prompting, so dataset construction need not parse prose.

The environment exposes an explicit observation mode:

- `legacy`: the pre-contract serializer, retained only as a COMP-002 control;
- `combat_public_v1`: the superseded public serializer without combat card types,
  retained only for exact replay of rejected artifacts;
- `combat_public_v2`: this contract.

New competence collection uses `combat_public_v2`. Run metadata must record the
mode and a serializer/interface fingerprint. Silent fallback to `legacy` is an
error once a run requests `combat_public_v2`.

The current implementation also sorts Secret Technique and Secret Weapon
candidate actions by their public descriptions while retaining their raw
execution bits. Their native engine enumeration indexes the hidden draw vector;
showing that order would leak precisely the information this contract forbids.
The structured `public_cards()` result carries the same selection type and
candidate descriptions as the text observation.

## Required verification

Before freezing the interface:

1. Snapshot at least one ordinary combat decision containing upgraded cards, a
   non-empty draw pile, a non-empty discard or exhaust pile, multiple enemies or
   powers, relics with counters, and potions.
2. Snapshot card-selection decisions and verify upgraded candidate names.
3. Assert that changing only hidden draw order or RNG state cannot change the
   rendered observation or its public hash.
4. Assert that rendering leaves state, legal-action bits/descriptions, and battle
   outcome unchanged.
5. Assert parity between structured cards/relics and their text representation.
6. Assert current-turn action history is ordered, resets on a new turn, and never
   carries across battles.
7. Measure fully augmented prompt tokens at p50, p90, and p99 on the Nob task
   before teacher-data collection. Record the tokenizer, reasoning mode, and
   maximum completion budget with the measurement.
8. Run matched `legacy` and `combat_public_v2` arms using the same model, held-out
   windows, policy sample indices, generation parameters, and code revision.

Items 1–6 are enforced by binding/replay integration tests. The old v1 start
signatures, token report, COMP-002 partial rollouts, and COMP-005 labels were
invalidated by the type omission. Revalidation writes signature schema 2 and a
new manifest rather than mutating the frozen v1 evidence. Items 7–8 must be rerun
against v2 before training or interpreting prompt-only competence.

## Teacher boundary

The search teacher is allowed to inspect a private cloned `BattleContext`, so its
decision may depend on information unavailable in `combat_public_v2`. Every
teacher row therefore records `teacher_privilege = simulator_full_state` (or a
more precise future value), plus search budget and stability measurements. Labels
that change under public-equivalent hidden-state perturbations are ambiguous and
must be filtered, consensus-labelled, or explicitly weighted; they must not be
presented as exact targets solely because search is strong on the unperturbed
state.
