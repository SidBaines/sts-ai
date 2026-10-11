# Public candidate representation

The training adapter uses `public_candidate_v2`, a 256-dimensional state and
128-dimensional vector per currently legal candidate. These vectors are
float32 and work identically on CPU/MPS. The encoder never consumes `info`,
native action bits, seed identity, hidden draw order, or teacher values. It
uses an allowlist of structured public fields and controlled public text
sections. The policy's only use of menu indices is routing its selected local
row back to the environment.

State features include health, gold, act/floor/ascension, combat energy/block,
incoming damage, player powers, pile sizes, turn counters, potion availability,
event public amounts, deck size, and future map room counts. Signed hash bags
encode card identities, types, upgrades/current costs and multiplicities;
enemy identities, HP and intent/power names; relic ownership/counters, potions,
OOC deck/relics, selection purposes, offers, and the visible act boss.
Unordered draw/discard/exhaust groups stay unordered. Hashing uses fixed
Blake2b instead of Python's process-randomized hash.

Each action contains semantic flags, current cost, publicly annotated damage,
known card block calculated from public Dexterity/Frail/No Block, card type,
upgrade, public rule draw/energy amounts, shop price/affordability, HP cost and
event probability. Targeted actions join to the matching enemy's HP, block,
intent damage, Weak/Vulnerable, powers and kill threshold. Duplicate enemy names
use the public `[enemy i]` identity. Card and potion rule definitions come from
the same public glossary the environment renders; these are static game rules,
not a privileged simulator query. Multi-selection masks use a bag of selected
card names/rules and selected count; there is no global `2**hand_size` output
head. Map choices encode their next room and distance-weighted room bags over
only routes reachable from that choice, using the public map edges.

The models score every candidate through a shared state MLP, action MLP,
state/action products, and a final scorer. `context=mean` adds masked mean
candidate embeddings, a cheap DeepSets context. The critic shares the state
encoder and optionally that menu context. Menu permutation permutes logits and
preserves the critic; padding is ignored by context pooling and receives logit
`-1e9`. Terminal batches retain one masked slot so the critic can be evaluated;
trainers must never sample a terminal action distribution.

| Size | Hidden width/depth | No menu context | Mean menu context |
| --- | --- | ---: | ---: |
| small | 128 / 2 | 148,482 | 181,250 |
| medium | 256 / 2 | 493,570 | 624,642 |
| large | 512 / 3 | 2,298,882 | 2,823,170 |

For the first ten-hour campaign, compare these sizes plus mean/none on fixed
validation partitions and measure wall-clock speed. Repeated process IPC and
small MPS kernels can dominate; larger batches should be measured before
assuming a larger GPU model is expensive. This representation avoids quadratic
attention over menus that can exceed a thousand multi-select masks. A later
entity attention architecture is worth comparing if supervised loss saturates
at enough data, since the bag features compress relations and collide.
Separate combat and OOC checkpoints can specialize without changing this
contract. A shared model can pool data and transfer deck semantics, but needs
phase-balanced batches to avoid abundant combat decisions drowning out OOC.

Limitations are explicit: hashed features collide and do not reconstruct the
whole game state. Compound event effects are hashed plus selected scalars;
there is no general text comprehension. Damage annotations can be missing for
special attacks; missing damage remains zero, not an invented numeric estimate.
Rule-based block is exact only for supported cards; fallback numbers in complex
rules are descriptive approximations and may be conditional. Non-card relic
interactions are largely categorical and must be learned. OOC public text is
parsed only from controlled sections; absent fields remain absent. We report
coverage for present OOC deck/relic/potion lines, recognized action types,
known/unknown card/potion definitions, and explicitly unavailable event effects.
Unknown symbols are retained in hash bags rather than silently discarded.
Encoder changes after dataset/checkpoint creation require a new version and
re-encoding rather than silently mixing representations.

Version 2 corrects the actual native `drink potion NAME` label format (version 1
mistakenly included the word `potion` in the lookup name), and excludes the shop
card-removal service from card-name coverage. Potion rule features therefore
change intentionally; do not reuse version 1 encoded caches/checkpoints as
version 2. Dimensions and numeric combat phase feature 7 are unchanged.

CPU profiling on 811 benchmark observations from 38 battles found menus of
mean 3.9 candidates, 95th percentile 8 and maximum 13. A combined controlled
public-line scan and one-pass map-room counts improved five paired-run median
encoding time from 0.198 to 0.139 seconds (1.42x); all state/action/index arrays
and coverage counters were exactly equal across the full sample before the
separate version 2 semantic correction. Version 2 takes approximately 0.167
seconds for this sample: mean 0.201 ms/combat observation and 0.214 ms/OOC.
Its coverage report contained 2,102 known card/potion rules and zero unknown
rules; 42 event/other labels lacked the coarse action-type flags but retained
their hashed semantics. This historical tiny sample is performance evidence,
not campaign training/evaluation data. Static room labels and rule lookups by
card/potion name are cached; mutable observation dictionaries and world IDs are
not cached. Fixed supervised datasets may be preencoded once by the trainer.

Verification: `tests/unit/test_policy_encoding.py` tests private-field
invariance, menu permutations, unordered piles, exact block and enemy matching,
selection purposes and unknown coverage. `test_policy_model.py` tests both menu
contexts, padding invariance, terminal critic evaluation, finite gradients,
model sizes and metadata/state-dict reconstruction. These are CPU tests; the
campaign driver benchmarks device throughput separately.
