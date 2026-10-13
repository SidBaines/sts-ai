# Curriculum task and data adapters

Current semantic profile is environment 0.3.0/public-v3/native-API-3 with
`public_entities_v4`, OOC model v3 and combat residual v2. Deck selections join
their precise public inventory source; reward offers retain group membership;
top-level combat gold/HP/max-HP are live. Entry snapshots support the explicit
historical bridge. See `docs/public_entities_v4.md` for acceptance and migration.
The earlier v3 encoder/model sections below describe the preserved 0.2 study.
Old typed checkpoints/banks must use `external/reference_code_v020` plus the
pinned v020 native build; current loaders intentionally reject them.

Inference batches must preserve phase separation. Mixed OOC/combat padding can
change near-tied greedy choices through floating-point rounding, even with a
zero residual. Collector v2 uses separate phase batches and compact legacy
combat choices in both micro and macro frozen-controller modes; its task RNG
stream remains v1. Exact preservation requires trajectory checks, not only
matching aggregate outcomes.

`combat_dagger.LabelingWorker` supplies privileged search labels while executing
the public policy's original choices. Completed/fault/cut worlds have separate
journals; faults discard every label from that world. Consolidated banks require
matching no-query traces and native/interface provenance. `teacher_panel` adds a
separate experimental RPC for search RNG and cloned hidden draw interventions;
its results never enter policy observations. A panel RPC contains six native
conditions, so panel RPC counts must not be called individual search counts.

One-hot and consensus losses are different conditions. Consensus averages the
negative log probability mass of each recorded model-equivalent label group;
it does not expose private root values or draw permutations as neural features.
Search disagreement may reflect near-equivalent good choices. Native root means
are search diagnostics, not calibrated public action values.

The combat fitter verifies the completed bank manifest and data digest before
creating an output. Publication writes the consolidated data before its final
digest manifest, so an in-flight bank correctly fails this gate; wait for the
collector's completed result rather than fitting a partial publication.

`reporting_v3`/`scripts/report_nonllm_v3.py` read curated JSON summaries only and
need no Torch import. Historical non-finite fields become explicitly flagged
unavailable values; source hashes preserve the original evidence. The portable
HTML snapshot is read-only and refreshes by rerunning the exporter. It does not
infer promotion or completion from a live PID or checkpoint filename.

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

The later direct-guard optimization removes the remaining per-prefix v5 summary
copy. Inside the isolated adapter, it reads exactly the base summary's predicate:
raw backend `_undefined_behavior_evoked`, OR a live battle context's
`undefined_behavior_evoked`. The sticky flag remains checked after battle exit;
terminal/legal-menu checks, OOC previews and all counters remain unchanged.

Landed-source verification repeated generated prefixes, ten random first-bank
starts, twenty Act 2/3 starts, and a known invalid prefix. Public observations,
all diagnostics, native teacher labels, subsequent rewards, and rejection errors
were identical. Matched persistent RPC resets over 7852 prefix actions measured
1.613 seconds with the old summary guard versus 0.915 seconds with direct flags
(1.763x additional reset speedup). No native binary or environment-base source
changed. This optimization landed after both health-ablation conditions finished.

The current collection adapter identity is
1f2e599fe97c57be233c3d66e1bb10682d9079394676c20657bd625781b41607.
Earlier banks preserve their original collection identities and remain replay
compatible under their unchanged native/Python provenance; new collection must
use a new directory rather than resume an old source identity.

The spawn-based simulator cannot start from a Python stdin entry point. Use the
guarded official CLI or a saved script with `if __name__ == "__main__"`; do not
interpret a missing `<stdin>` child entry point as a simulator or policy fault.
# Repaired strategic candidate

The new protected study uses `public_entities_v3` and native environment 0.2.0;
legacy campaign checkpoints use `public_candidate_v2`. Loaders are separate.
See `docs/public_entities_v3.md` for inputs/known limits and
`docs/nonllm_v3_summary.md` for measured progress. Old banks fail new core
provenance; historical artifacts remain in their original worktrees.

Entry points (run with this worktree's `.venv`, `PYTHONPATH=src` and its pinned
`STS_LIGHTSPEED_BUILD`):

- `scripts/run_strategic_v3.py initialize|audit|equivalence|evaluate`
- `scripts/prepare_strategic_v3.py` collects original-policy TRAIN calibration
  and rehearsal, then fits only the separate critic. A fitted checkpoint is not
  automatically accepted: inspect independent TRAIN holdout calibration.
- `scripts/run_protected_strategic_v3.py` runs one finite micro/macro condition.
  It requires a fresh output directory and the matching common initialization,
  rehearsal buffer and permanent DEV baseline. Sampling and optimizer seeds are
  distinct. Checkpoints, samples, updates and evaluations stay immutable; only
  `status.json` is the live atomic pointer. Its run ID/PID are published before
  workers/caffeinate launch. The controller only closes processes it owns.

Non-obvious contracts: typed checkpoints pin model/encoder/bridge code digests,
so changes need explicit migration; macro aggregation currently requires gamma
one; native-prefix source decks are a separate distribution; AC presence can
coexist with battery discharge. Cuts and faults remain separate from defeats.
Current controller resource reserves are 25% battery when not charging and 8 GiB
free disk, with a finite UTC deadline. A resource pause of the run does not mark
the broader implementation goal complete.

MPS flushes subnormal float32 probabilities to zero. Final policy KL/TV/entropy
diagnostics therefore transfer logits to CPU and compute in float64. Never use
MPS `p * log(p)` with tiny reference probabilities as a stopping guard: it can
produce NaN and silently bypass comparisons. Non-finite diagnostics fail closed.

Model v2 feeds controlled legacy features into the trainable residual as well
as the immutable prior; model v1 remains reproducible at `bab055a`. A checked
zero-residual migration can add a new rehearsal reference without relabeling its
actual source trajectories. Learned residuals require a separate migration.

Use `scripts/build_strategic_starts_v3.py` to construct/hash-replay OOC starts
before later fights. OOC teacher records have `simulations=None`; prefix task
settings come from the run-level journal configuration. Boss-only continuation
wins can have zero actor rows and must not count as useful strategic learning
experience. Mixed/repeated continuation counts do not get a binomial interval.
