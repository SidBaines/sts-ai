# Curriculum task and data adapters

The environment release remains unchanged. `workers.CurriculumWorker` wraps a
private base episode only inside a persistent, bounded native subprocess. The
parent owns inference, learning and model memory. Each worker supplies reset,
step, teacher_action, and asynchronous submit methods. On any native fault the
worker process is destroyed; reset creates a clean one. Faulted transitions and
labels are never ordinary defeats.

`curriculum.TaskSpec` identifies a world, task kind, exact public action prefix,
optional observation hash, horizon and act limit. Battle tasks end immediately
at the fight outcome; OOC tasks resolve combat with native search; run tasks
allow control of all actions. Limits truncate, game/fight outcomes terminate.
The base records max_act completion as a truncation. This explicit curriculum
adapter promotes it to terminal `act_clear` success when `goal_act_clear=True`
(default). Full-game victory and individual-battle victory remain distinct
outcomes; task horizons remain bootstrap truncations. Set goal_act_clear=False
to retain the base cutoff semantics.

Non-obvious replay invariant: native OOC action previews are pure queries, but
executing their chosen action commits the native policy RNG. Replaying only
menu indices would omit that commit. `ooc_previews` records which prefix actions
were preceded by a native OOC preview; all replay banks must preserve it, along
with max_act, ascension, seed and search settings. `task_start_ooc_previews`
accompanies the worker's `task_start_prefix` in diagnostic info. Policy inputs
are exclusively the `observation` object, never these diagnostics or raw native
bits. Native teacher labels are privileged and kept separate; the OOC teacher
is a scripted baseline, not a strategic expert.

`data.collect_dataset` generates fresh labels and battle banks against the
new base native binary. Every world shares one stable train/dev/final split,
including all fights from that world. Atomic per-world journals preserve
completed worlds after interruption. Resume rejects changes to collection
settings or adapter source. Errors discard the entire world's labels. Summary
aggregation streams worlds to keep memory bounded. Use a new output directory
for different collection settings or code identities.

Reward version `sts_curriculum_reward_v2` belongs to this experiment adapter.
Worker config `reward_mode` selects potential (default), sparse, or progress.
Gamma defaults to 1.0; training must log/configure its matching discount. Progress
mode adds progress_floor_weight (default 0.05) per new floor; this changes the
objective and is a curriculum hypothesis, not invariant potential shaping.
Terminal HP bonus is optional via terminal_hp_weight (default 0.0). Reward
settings/components are recorded in info, separately from policy observations.

Potential shaping It uses public player HP
and enemy HP progress, with gamma*Phi(next)-Phi(previous), terminal Phi zero.
Block is deliberately excluded because rewarding block directly invites
endless block cycles. Horizon cutoffs retain potential for value bootstrap;
native faults never produce reward targets.

Optional `fast_observations=True` installs process-local precompiled status
regexes and bounded immutable-text caches. This addresses the observed glossary
rendering bottleneck without editing base source or changing observations.
Paired whole-run public observations, encoded state/action arrays, teacher
actions, rewards and diagnostic info must remain exactly equal. The collection
CLI accepts --fast-observations; dataset identity includes this adapter's source.

Bank replay validates Python public-interface/runtime identity and exact native
provenance before executing a prefix. Resume probes the live build and validates
the same native/interface provenance for all valid saved worlds. A public start
hash alone cannot establish compatibility across changed native builds.

11 October 2026 Mac CPU panel (32 identical world seeds, native 50 simulations):
uncached workers 1/4/8/12 yielded 743/2267/3638/3927 valid records per second;
process-local exact caches yielded 1343/3597/5002/5021. Both produced exactly
5562 valid labels and the same one simulator fault. Eight cached workers achieve
99.6% of twelve-worker throughput, leaving CPU capacity for policy encoding.
This is a short native-collection benchmark, not an end-to-end GPU-training
measurement. Native teacher search consumed about 4% of profiled runtime;
glossary status scanning was the dominant removable cost.

## Optional privileged OOC lookahead teacher

`CounterfactualOOCWorker` in lookahead.py exactly replays a full-mode prefix and
evaluates up to four sampled public menu candidates from common seeded starts.
Native search handles combat during each continuation; the seeded native OOC
policy handles subsequent OOC choices. The default horizon is 30 OOC decisions.
The scripted baseline is always included, and wins exact score ties to avoid
creating label noise from arbitrary tie breaks. Any branch native fault rejects
the entire source label. Candidate-order invariance and repeatability are tested.

Explicit proxy score: act-clear/full-victory +1, defeat -1, final public HP/max HP
times 0.25, and new floors times 0.05. This teacher uses privileged continuations
and evaluates a partial menu; it is not a strategic-expertise claim. Logs retain
individual score terms and whether the continuation horizon was reached. The
engine heals at the next act, so final HP saturates on act-clear outcomes; this
proxy is useful for an ablation, not a certificate of stronger overall play.
Only the original public observation and chosen index are training inputs.

`collect_ooc_lookahead.py` samples train-world OOC states from atomic source-bank
journals and emits separately identified labels, with independent source/code/
settings validation on resume. Eight tested sources produced two strict proxy
improvements over scripted labels, no faults, in 0.81 seconds of evaluation;
that does not establish any held-out policy win-rate improvement.

## Native automatic-combat validity limitation

The pinned binding's resolve_current_battle creates a local BattleContext, calls
ScumSearchAgent2::playoutBattle, then exitBattle. It does not check the context's
undefinedBehaviorEvoked flag. BattleContext::exitBattle copies player/game state
but does not retain that flag in GameContext. Exposed GameContext and Agent APIs
have no last-battle/context/UB status accessor; Agent.playout operates on an
entire GameContext and also destroys its local battle context. An adapter cannot
recover this flag after automatic resolution using the current native API.

Therefore OOC/automatic-combat comparisons report surfaced exceptions/timeouts
separately but cannot claim parity with full-combat live-flag detection. Zero
reported faults does not certify their continuation labels are free of simulator
UB. Unflagged corruption can also escape full-combat detection. Original labels
and scientific settings are preserved; this finding does not silently change
the native controller or overwrite prior banks.

A minimal future environment-base maintenance patch should check
bc.undefinedBehaviorEvoked after bc.init(gc), and again after
agent.playoutBattle(bc), throwing before bc.exitBattle(gc) if either is true.
Keep the native controller/seed/search settings unchanged. Add native regression
coverage for automatic-resolution flag rejection and ordinary deterministic
trajectory parity, rebuild, publish a new release tag, and collect separately
identified comparison data. Explicit micro-action search in an experiment
adapter would expose live flags, but changes the controller and must be treated
as a separate experiment instead of a transparent repair.

## Optional prefix replay acceleration

Worker config `fast_replay=True` skips intermediate observation text generation
when reconstructing a saved action prefix. It still executes each action through
the v5 public semantic wrapper, preserves OOC preview RNG commits and counters,
advances native state identically, and checks sticky UB, terminal state, and a
nonempty legal menu after every prefix action. It renders once at the final
frontier; default replay remains unchanged.

Paired generated trajectories, ten random first-bank starts, twenty Act 2/3
starts, and a known fault prefix verified exact reset observations/diagnostics,
teacher labels, subsequent rewards/observations/diagnostics, and identical fault
rejection. Matched persistent RPC benchmarks over twenty later-act starts and
7852 prefix actions measured reset totals 4.118 seconds standard vs 1.579 seconds
fast (2.61x), with all semantics checks passing.

Initial native banks retain collection adapter hash
6438978542e95701d505d273534209a35d84140bb5918580de059642e8a37499.
Adding this optional replay path changes the collection source identity to
08e37bd55722543cc64a0f81a7f5b37eafc91a28b3f194761bdc3056f6984f4d.
Old banks remain replay compatible because native/public Python identities are
unchanged and still validated. Never resume old collection under changed source:
use a new dataset directory so the recorded collection identity remains true.

## Balanced panels and rehearsal reservoirs

`build_curriculum_panels.py --banks ... --output ...` assembles preserved battle
starts without executing any held-out learner. Training uses three genuine act
files cycled by round rather than invented duplicate IDs. Development/final
panels request equal act counts, select across worlds before adding extra fights
from a world, and never oversample evaluation starts. Metadata records actual
world/floor coverage and source bank manifest identities. Publish new cohorts
to a versioned directory: changing an active development file mid-screen would
make candidate scores incomparable.

`aggregation.aggregate_teachers` streams a deterministic train-only reservoir
over native/learner label files. It validates world splits and legal menu labels,
retains source-file/line hashes, and never overwrites a source. A fixed native
rehearsal sample avoids repeatedly scanning gigabytes of immutable native labels
for every DAgger correction; subsequent aggregation adds all prior learner
corrections. The saved rehearsal file and its manifest provide the provenance
chain back to original native banks. Compression and source mixing are explicit
recorded settings, not silently equivalent to retaining every native example.

The first Act 3 expansion used 2048 separate worlds, native 200 simulations,
four CPU workers and a 480-second collection ceiling; all worlds finished in
337 seconds, with 2025 valid and 23 flagged-invalid worlds. The expanded panel
contains 16 Act 3 development fights from 15 worlds and 16 Act 3 final fights
from 12 worlds, replacing the original one-world-per-split coverage only after
its separate publication. Original banks/panels remain preserved.
