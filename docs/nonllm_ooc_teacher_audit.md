# OOC teacher audit: matched scripted versus lookahead labels

**Quarantined historical evidence:** subsequent backend diagnosis found that the
old automatic native combat resolver default-constructed `BattleContext bc;`,
leaving `Player.cc` uninitialized. These OOC rollouts and the lookahead continuation
labels used that affected automatic path. The clear counts below document the
matched observation; they must not be used to select or compare teacher quality
until both labels and evaluation are repeated on the isolated corrected binary.
The label/parser/fit diagnostics remain useful, but the contaminated simulator
can contribute to the apparent proxy gains and regressions. No historical
artifacts are rewritten; native digests distinguish the retest conditions.

The matched relabeling comparison favored the scripted-label model: 19 Act 1
clears versus 12 for lookahead, from 32 requested development worlds. Each had
31 valid completed episodes, zero truncations, and one separately reported native
timeout on a different world. On the 30 common clean worlds the counts remain
19 versus 12. Faults are not defeats. These are development comparisons, not final
evaluation or a claim that the scripted teacher is strategically optimal.

Both variants start from `ooc_specialist_bc.pt`, fit for 90 seconds at learning
rate 0.0003, and use the same 256 public observations in the same order. An audit
verified equality of observations/world IDs and legality of every label. Only
89 selected labels differ (34.77%). None of those changed action pairs collapse
to identical candidate features; card-selection purposes are visible and encoded.
There is no evidence here of menu-index misrouting or an ordering/parser failure
causing the result.

| OOC context | Sampled states | Changed labels |
| --- | ---: | ---: |
| Rewards | 154 | 56 |
| Map | 37 | 12 |
| Card select | 17 | 9 |
| Rest room | 17 | 4 |
| Event / Neow | 16 | 7 |
| Shop | 7 | 0 |
| Treasure | 5 | 1 |
| Boss relic reward | 3 | 0 |

The sample covers 148 training worlds; menus average 5.33 actions and reach 18.
Lookahead evaluates at most four candidates, including the scripted choice.
Card-selection changes include 3/3 removal, 1/1 transform and 5/12 upgrade states.
Shop labels did not change, so this comparison cannot establish a shop-teacher
improvement. Rewards dominate the changes: 15 gold-to-card choices, nine
gold-to-skip-all-rewards choices, four card-to-skip-all and four card-to-skip-card
choices. Choosing to skip free gold because a short continuation scores better
is evidence of a noisy/misaligned continuation target, not proof that skipping
gold is a generally good strategy.

The proxy has material limits. It evaluates one fixed hidden future with native
combat search at 50 simulations and scripted OOC continuations, for 30 external
decisions. Actual OOC evaluation resolves combats at 200 simulations. Its score
adds terminal success, endpoint HP fraction and floor progress, without explicit
gold/deck-quality value. Of 889 branch continuations, 435 reached the horizon.
For the 89 changed labels, endpoint HP increased in 77, floor progress in 47 and
terminal outcome in 27; these counts overlap. Thus every label's apparent local
proxy gain does not establish improved run-level policy value.

Endpoint HP is particularly misleading at act completion. All 190 recorded
`act_clear` branches had HP proxy 0.25, its full-health maximum. The pinned
simulator heals to full when transitioning between acts at ascension zero
(`GameContext::transitionToAct`); the observed endpoint therefore does not measure
HP preserved before the boss/transition. This is a real proxy-design limitation,
not evidence that the simulator violated its documented rules.

| Policy checkpoint | Training fit on its labels | Scripted dev accuracy, choice menus | Scripted dev accuracy, forced menus | Scripted dev cross-entropy |
| --- | ---: | ---: | ---: | ---: |
| Broad warm start | 89.82% | 72.32% | 100% | 2.35 |
| Matched scripted fit | 99.98% | 68.10% | 100% | 13.28 |
| Matched lookahead fit | 98.28% | 51.05% | 100% | 9.19 |

Development contains 1,505 OOC decisions: 1,044 genuine choices and 461 forced
menus. Forced accuracy is automatically perfect and should not obscure choice
regression. The 256-state datasets were replayed for about 7,174 and 7,133 full
passes respectively during the 90-second fits. Both models nearly memorize their
small target set and lose broad warm-start behavior. Scripted agreement is not
the right sole metric for a deliberately different teacher. The apparent matched
whole-run regression motivated this audit, but the subsequently discovered
automatic-combat corruption prevents attributing it to teacher quality.

No source/campaign changes were made for this audit. Retain and quarantine the
matched ablation; it is not valid selection evidence after the backend finding.
Future bounded fixes should
mix new labels with broad replay or distill the warm policy, reduce fit duration,
and validate held-out whole-run outcomes. A revised teacher should separate
pre-transition HP from automatic healing, avoid proxy incentives to skip free
rewards, evaluate multiple continuation/search draws or abstain when rankings are
unstable, and cover the menu more systematically. These are experimental changes
that need a new teacher identity and matched retest; they are not silently applied
as implementation fixes.

The encoder also lacks explicit base energy costs on OOC reward/shop card labels
when that cost is absent from the public description; categorical names/rules
remain available. This existing representation limit affects both variants and
does not by itself explain the differential regression.

Evidence: `data/nonllm_curriculum/ooc_lookahead_v1/{teacher.jsonl,manifest.json}`,
the matched scripted teacher in `data/nonllm_v1/campaign/ooc_matched_scripted_teacher.jsonl`,
and `ooc_{scripted,lookahead}_matched_{bc,dev}.metrics.json`. Audit calculations
were CPU-only and used training/development material; no GPU jobs or final reads.
