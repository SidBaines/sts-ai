# Earlier strategic start collection

13 October 2026. `scripts/collect_early_strategic_frontiers.py` records exact
executed prefixes from fresh TRAIN games controlled by the frozen original
public OOC policy and explicitly privileged native combat. This is a source
curriculum, not a public-agent benchmark or a teacher target bank.

The parent worker records a choice before executing its current action, so the
saved prefix contains only preceding actions and their actual preview markers.
Only completed, valid source worlds are eligible. A later source fault removes
every candidate from that world; unfinished/faulted candidates stay preserved
in quarantine. Every accepted start then reconstructs the exact public
observation and legal menu in a separate native process. A replay failure
excludes its whole source world. Deadline-limited unverified starts remain
separate from accepted data.

The initial v1 probe completes all 16 fresh TRAIN games in **20.49 seconds**
using two workers, native combat at 1,000 simulations, no faults/cuts and no
source victories. It records and replay-verifies **89/89** starts from 14 source
worlds: 84 Act 2 starts from 14 worlds, five Act 3 starts from **one world**.
Different screens and acts from one world are correlated. This is successful
collection and reconstruction, not broad Act 3 coverage or a useful teacher.

The source is `data/nonllm_v4/banks/early_original_native1000_probe16`; its
manifest, complete source outcomes, exact candidates, accepted starts and
quarantine/unverified files are preserved. Latest v1 validation runs 553
host/native tests with one optional artifact skip. The accepted environment
remains 0.3/public-v3/native-API-3; no core or model weights change.

The probe exposes a selection limitation: the first reward menu still contains
gold and can omit the actual later card-only decision. Next versions should
distinguish public menu-kind/selection-purpose stages and include the Act 1
boss-relic choice, which shapes later deck viability. More frontier rows cannot
substitute for more independent late-act worlds. Future sampling must choose
worlds before starts so a world with many recorded screens does not dominate
the training mix.

## Menu stages and native-controller scope

Frontier v2 records the first public menu-kind/selection-purpose stage in Acts
2/3, plus the late Act 1 boss-relic decision. This captures actual card-only
rewards after gold/potions are collected and distinguishes removal, upgrade and
bottling purposes. Repeated stages are skipped before copying their observations.
V1's exact source remains in commit `b58d0fb` and its original data are preserved.

Collector v6 removes combat scoring that native control immediately discards,
and reports actual externally native-controlled decisions separately from the
environment's automatic `native_steps`. A fixed 16-world timing comparison
preserves all 16 control traces, rewards and endpoints: policy CPU time falls
4.065→2.753 seconds, but total wall time is 19.679→20.894 seconds. That pair does
**not** demonstrate an end-to-end speedup; native/step wait variation matters.

The per-action native comparator also differs from the simulator's built-in
agent: it discards the rest of a searched winning sequence after each action,
whereas the built-in agent retains a winning plan. Earlier native-control
results therefore characterize the per-action comparator, not all native
controllers. This could understate competence and inflate cost.

A separately named `privileged_winning_plan_reuse_v2` controller retains only
winning sequences with a matching first action. Every cached step checks current
legal bits, native description and turn; policy deviations/settings changes,
resets and battle boundaries clear the cache. Its diagnostics count queries,
cache hits, invalidations, exhausted plans and unexpected defeats. This is a
private controller diagnostic, not an exact reproduction of the built-in agent
or a public expert oracle.

The first cache prototype compared native descriptions with enriched public
display labels, causing 2,577 conservative invalidations on the 16-world panel.
Those v1 results stay preserved. The v2 guard uses the same native description
format as search. On the same 16 worlds it has **zero invalidations, zero
winning-plan defeats, 690 searches and 3,981 cache hits**. It reaches 15/16 Act 1
clears and 3/16 Act 2 clears, versus 14/16 and 1/16 for per-action replanning;
both have zero full wins. It finishes in 15.006 seconds, but changed trajectories
make this a controller comparison rather than a matched-workload speed benchmark.
All 259 new starts replay exactly. No policy or target is promoted.

The short prototypes initially saved source hashes without source bytes.
Their executed sources are now preserved with every file verified against its
original launch hash, including reconstructed timing-control sources. New
frontier and counterfactual launches automatically snapshot environment/policy/
script sources, vocabulary, project metadata and the native patch. Runtime,
checkpoint and native-binary identities remain separately recorded; snapshots
do not replace preservation of the runtime/models/build.

## Expanded source bank

The independent 64-world panel at world range 431000000 finishes every game with
zero faults/cuts in 45.53 seconds, using two workers and the v2 plan controller.
It reaches **60 Act 1 clears and 17 Act 2 clears**, with **zero full wins**.
This is explicitly privileged source play, not a public trained-policy result.
Its full source outcomes remain in `early_stage_native1000_plan_v2_64`.

The initial 45-second replay window verifies 990/1,051 starts and preserves the
61 remaining candidates. `complete_frontier_replays.py` verifies the rest into
a **new child bank**, preserving the first output and parent hashes. All
**1,051/1,051** candidates now replay exactly, with no failures/unverified starts.
The complete bank is `data/nonllm_v4/banks/early_stage_native1000_plan_v2_64_complete/starts.jsonl`:

- 60 late Act 1 boss-relic starts, from 60 worlds.
- 780 Act 2 starts, from 60 worlds.
- 211 Act 3 starts, from **17 independent worlds**.
- 77 card-only reward decisions across the recorded acts.

The original current-profile late bank had nine Act 3 source worlds. These 17
are new, independent TRAIN worlds; multiple menus within each remain correlated.
This establishes broader/earlier source coverage, not useful gradients or a
teacher. World-balanced sampling and outcome validation are still required
before fitting a condition. No original campaigns or final cohorts are opened.

Final acceptance runs **563 host/native tests**, one optional generated-bank
skip, and the mandatory bounded smoke without error. A separate fresh process
repeats all **16 full plan-controller traces, rewards, endpoints and diagnostic
counters**, and reconstructs the same 259 starts. The 64-world panel has 2,634
search queries, 15,298 cached actions, **zero invalidations, zero exhausted plans
and zero predicted-winning-plan defeats**. These are sampled execution checks,
not complete simulator certification. Parent/child bank hashes and every saved
source file (116/117 files respectively) verify. No generated teacher target or
learning condition is accepted from this collection.

## World-first card-choice pilot

`strategic_world_then_start_v1` chooses source worlds before choosing a start
within each world. Stable independent RNG streams and canonical row order make
selection reproducible under input reordering; duplicate rows are removed.
Adding many menus to one world does not change world selection probability.
The diagnostic CLI retains its historical first-source default and explicitly
opts into world-balanced selection, act/family filters and plan-controller use.

A matched six-world pilot selects three Act 2 and three Act 3 card-only reward
menus from the complete bank. Both frozen-public and privileged-plan
continuations evaluate **30/30 legal branches**, with all six control traces
matching and zero faults/cuts/unstarted sources. The selected task definitions
match exactly between controllers. Every reference and alternative loses the
full game; no menu supplies a new win. Four public-continuation menus tie on all
paired outcomes, versus five with privileged continuation.

Skipping the card in world 431000032 improves public-continuation floor reach
24→29; that gain disappears under privileged continuation. Thus even the lone
positive progression signal is controller-dependent. These fixed-world,
single-choice probes do not identify a reliable strategic teacher. No targets
are accepted. Broader sequential strategy, continuation competence and hidden-
future uncertainty remain unresolved rather than being replaced by proxy labels.

Evidence: `data/nonllm_v4/counterfactual/early_cards_world6_public` and
`early_cards_world6_plan1000`. Counterfactual protocol v3 explicitly records
per-action versus plan-reuse continuation roles. Source snapshots accompany
both launches; earlier data and controller conditions remain preserved.
