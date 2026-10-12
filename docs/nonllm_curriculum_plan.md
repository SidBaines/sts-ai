# Non-LLM Slay the Spire curriculum: first local campaign

## Objective, scope, and resource contract

**Allocation amendment, 12 October 05:03 UTC:** before any final evaluation, extend the three-act training stage to approximately 06:00 UTC, reserving 30 minutes before the unchanged 06:30 cutoff. Measured completed DEV checks take 20 seconds for 48 full-control three-act worlds, 12.3 seconds for 48 native-reference worlds with only two workers, and 4.5 seconds for 32 later-act fights. Twenty-two final panels plus CLI/report checks fit a deliberately larger reserve than these observed costs. This changes time allocation only; frozen cohorts, seeds, horizons, rewards and no-final-tuning rules remain unchanged. Preserve the earlier plan/history below.

**Execution amendment, 12 October 03:05 UTC:** the branch still originates at immutable `env-v0.1.0`, but active native-assisted training/evaluation uses signed `env-v0.1.1` after an uninitialized-character bug was repaired. Earlier native-auto evidence is quarantined. Retained full-control combat evidence was checked separately; [revision evidence](nonllm_native_revision.md) records the scope. The planned final embargo has one diagnostic exception: an early learned-agent seed-3 smoke touched the broad hash-final bucket outside the designated benchmark. See [the protocol audit](nonllm_protocol_audit.md). These amendments preserve the initial plan below while making the executed conditions explicit.

Train and evaluate a non-language policy controlling both combat and out-of-combat
decisions, using the immutable `env-v0.1.0` environment release. Work lives on
`codex/nonllm-curriculum`; the original research checkout and environment release
remain untouched. The first campaign has ten hours on the local Apple M5 Pro,
15 CPU cores and 48 GB unified memory, with exclusive GPU use.

The allocation starts at 2026-10-11 20:08:46 UTC (21:08:46 BST) and ends at
2026-10-12 06:30:00 UTC (07:30:00 BST). Setup, profiling, training, validation, and
the final evaluation all count against that window. No training is allowed past
the deadline. Checkpoint and summarize incomplete work honestly rather than
calling a pilot a completed competence result. Reports may be finalized afterward
without extending the GPU allocation.

Initial scope is Ironclad, ascension 0, ordinary Acts 1–3. Start with Act 1 and
expand full-run horizons after the policy can play constituent fights. Act 4 and
other characters require additional observation/encoder work. This is a first
empirical campaign, not a claim that ten hours is sufficient to master the game.

## Data and validity invariants

- Inputs are exclusively the public observation: structured state, public text,
  and current legal action indices/descriptions. Never encode world/policy seed,
  raw native action bits, native rollout values, future draw order, teacher
  outputs, checkpoint outcomes, or diagnostic `info` as policy features.
- Privileged native combat search supplies supervised labels, clearly separated
  from observations. It is a ceiling/reference rather than a human-information
  comparator. Native scripted OOC labels are warm-start baselines, not expert
  strategic supervision. OOC lookahead labels are also privileged and explicitly
  conditioned on their continuation policy/search budget.
- Generate fresh banks under the released simulator binary. The old Gemma/fight
  banks have incompatible native/interface identities and must not be reused.
- Split by world seed before training: hash-based 80% train, 10% development,
  10% final embargo. Every fight, prefix and decision from one world stays in its
  split. Freeze the seed cohort in the campaign config and never tune on final.
  Bootstrap uncertainty by world, not by correlated decisions or fights.
- Simulator crashes, timeouts, flagged undefined behavior, replay mismatch, and
  sanity failures are separate failures, not ordinary game deaths. Quarantine
  the affected episode/world and retain evidence. Do not silently replace a
  failed held-out world or compare incomplete cohorts as complete ones.
- Battle starts store the world seed, exact canonical action prefix, public hash,
  native/Python identities, and OOC teacher-preview flags. Those preview flags
  are essential: native OOC policy RNG advances only when its preview is executed.
  Replay all starts used in a pilot, then sample/check every larger bank family.
- All training conditions have immutable configs, distinct output directories,
  seed identity, source hashes, checkpoints, and resumable progress. Use atomic
  checkpoint/JSON replacement; preserve a partial batch on interruption.

## Observation and action representation

A fixed output unit per menu position is a poor main representation because
indices change meaning across screens. The main actor instead scores every
currently legal candidate using shared parameters, then normalizes over the
legal set. A permutation of the menu must produce the same permutation of
scores and the same semantic choice. Padding never contributes to pooling,
probability, loss, entropy or value targets. Single-action menus are valid but
reported separately because they inflate imitation accuracy.

The first implemented encoder has 256 state features and 128 features per legal
candidate. It combines normalized scalars with deterministic categorical/hash
bags and controlled parsing of public fields. It includes player HP/energy/block,
current enemy HP/block/intent/powers, current card costs/types/upgrades/effect
amounts, draw/discard/exhaust counts and identities, active relics/potions,
selection purpose, public deck/offer information, prices/HP costs, and visible
map reachability and routes. Candidate features resolve damage/block/cost and
target-specific enemy facts from the same public state. Multi-card choices are
represented by their public selected cards and aggregate features, not native
subset bit masks. Log parsing/unknown-coverage rates by phase; richer encoders
must be added if absent features visibly limit learning.

| Representation | Strength / limitation | First-campaign decision |
| --- | --- | --- |
| Global fixed-index policy | Cheap, but meaning changes with menu ordering | Diagnostic baseline only, not the main learner |
| Shared candidate MLP | Legal, permutation-equivariant, inexpensive | Main first implementation |
| Candidate MLP + DeepSets menu context | Can compare relative options without positional leakage | Direct ablation against independent scoring |
| Structured entity DeepSets | Models card/enemy multisets more explicitly | Next encoder if collisions/coverage become bottlenecks |
| Set Transformer / cross-attention | Captures interactions and target matching; more compute | Small bounded pilot if simpler models plateau |
| Recurrent GRU/LSTM policy | Useful when the public view is not fully Markov; history complicates replay/training | First diagnose history-sensitive errors, then add if justified |
| Factorized card/target/subset policy | Scales large subsets, but legality and joint effects become harder | Compare if subset enumeration dominates measured time |
| Enumerated legal subsets | Exact legality and simple learning; up to roughly 1024 choices | Keep initially; bucket/chunk expensive menus |

The first sizes are approximately 0.15–0.18 M, 0.49–0.62 M, and 2.3–2.8 M
parameters, depending on menu context. These are small purpose-built networks,
not LLMs and not pretrained language models. An encoder collision or a missing
mechanic cannot be repaired simply by increasing parameter count.

## Learning methods and alternatives

| Method | Why consider it | Decision under ten hours |
| --- | --- | --- |
| Behavioral cloning | Dense supervised signal; establishes representation and throughput | Combat and OOC warm start; compare phase-balanced/shared versus separate models |
| DAgger | Corrects distribution shift when learner reaches states absent in native trajectories | Query combat teacher on learner states; aggregate without leaking labels into inputs |
| Masked PPO + critic | On-policy, discrete actions, flexible curriculum and shaping | Main online improvement route after warm start |
| REINFORCE/A2C | Simpler updates, but higher variance / less reuse | Small diagnostic if PPO overhead or clipping obscures a failure |
| DQN / distributional / double Q | Replay reuse may be attractive | Requires candidate-action Q scorer and legal next-action maximization; first compare sample efficiency if PPO stalls |
| Discrete SAC | Off-policy entropy and replay reuse | Later if collection is the dominant cost; do not apply continuous SAC directly |
| Offline CQL/IQL | Potentially reuse banks without more simulation | Teacher coverage and changing legal sets make extrapolation a concern; baseline only if online budget is constrained |
| Search distillation / AlphaZero-style policy-value search | Strong labels and planning feedback | Native teacher distillation now; learned public-state search needs a separate belief/chance model |
| MuZero/world model | Could amortize simulation and handle long planning | Too much implementation/validation risk for the first ten hours |
| Evolution / random search / linear models | Cheap baselines that can expose feature quality problems | Random/first/public tactical heuristic and simple linear scorer where useful |

PPO must store the generating policy's log-probabilities and values at collection
time. Ratios must be against that frozen behavior policy; stop-gradient of the
current forward pass is not an old policy. Test non-unit ratios, clipping, KL,
masking and weight changes. Use a scalar critic, normalized advantages, gradient
clipping, target-KL early stopping, entropy diagnostics and a held-out live panel.

Teacher agreement is a diagnostic, not the primary outcome: multiple actions can
be good, native search is privileged, and BC can imitate a weak OOC script.
Measure win rate and HP cost on fights, and floor/act clears/full-game victories
with all failure and truncation rates on integrated runs.

## Curriculum and decision-point strategies

1. **Mechanics and easy isolated battles.** Collect replayable fight starts from
   native trajectories and a small exploratory mixture. Stratify by enemy family,
   floor, HP and deck size. Start with regular single-enemy fights/starter-like
   decks; include mid-turn/card-selection states in imitation rather than only
   opening positions. Evaluate on held-out worlds and enemy strata.
2. **Diverse battles and distribution correction.** Add multi-enemy rooms, elites,
   bosses, potions and selection mechanics. Mix earlier examples to prevent
   forgetting. DAgger queries a teacher on student-visited states; use bounded
   teacher budgets and report how labels/outcomes change at 50/200/1000 simulations.
   Do not declare different seeds 0/1 independent on libc++.
3. **Combat RL.** Fine-tune a selected checkpoint on independent fight tasks,
   comparing warm-start PPO with a small scratch-PPO control. Match task starts
   across conditions. Use success/failure reward and explicit public potential
   shaping; no repeated positive block reward that encourages cycling. Report
   HP cost beside win rate and time/action counts.
4. **Out-of-combat policy.** Warm-start from clearly labeled scripted choices,
   then optimize Neow, routes, card/relic/potion rewards, campfires, shops and
   events with a fixed native combat controller. Evaluate per-screen entropy,
   agreement and live outcomes. Where the baseline is poor, query bounded
   counterfactual continuations at replayable public frontiers with common world
   identities and multiple search/controller seeds; record teacher privilege.
   Gold-spending, card synergy and route planning should be learned from outcomes,
   not hand-coded strategic framing. Automatic forced decisions get no inflated
   policy-learning credit.
5. **Integrated traces.** Compare a shared model with a phase-selected composite
   of separately fine-tuned combat/OOC models. Train increasingly long horizons
   (short segments -> complete Act 1 -> multi-act runs), mixing isolated combat
   refreshers if forgetting appears. Train OOC with the learned combat model as
   soon as it is reliable enough; native-combat OOC performance alone does not
   establish useful joint play. Include full-control held-out runs in the final
   panel regardless of which specialization won development selection.

For shaped rewards use `r + gamma*Phi(next)-Phi(previous)`, setting terminal Phi
to zero for a policy-invariant finite-episode potential. Genuine time-limit
truncations retain bootstrap value; deaths do not. The initial worker potential
uses public HP and enemy progress and is bounded. Its optimization target is
success/failure; HP preservation is separately measured. Any added final HP bonus,
progress reward, or length penalty changes the objective and gets a new logged
reward config/ablation. Match worker and trainer gamma exactly. Native-resolved
combat is a macro transition: log the discount unit and evaluate horizon effects,
rather than silently treating it as a combat micro-action.

## Empirical screens and allocation

All times are ceilings, not invitations to run a slow setting for hours.
Promote candidates by development evidence per wall time, preserving controls.

| Allocation from request | Work and decision gate |
| --- | --- |
| 0–45 minutes | Finish plan/adapters/tests; benchmark native workers 1/4/8/12, CPU/MPS/MLX batches, source/encoder coverage; collect fresh bank concurrently on CPU |
| 45–120 minutes | Six size/context BC candidates, learning rates 1e-4/3e-4/1e-3 using successive halving, batch 64/256/1024 where efficient; retain two best live-performing candidates; include initialization seeds |
| 2–4 hours | Combat DAgger and PPO: teacher budget pilot, easy-to-hard mixture, scratch control, clip 0.1/0.2, entropy 0.005/0.02, gamma 0.99/1.0 matched shaping, learning-rate screen; select by development fights/HP cost/throughput |
| 4–6 hours | OOC shared vs specialized warm starts, fixed-native combat PPO and selective lookahead; compare per-screen deficits; start learner-combat coupling |
| 6–9 hours | Joint full-control PPO and curriculum horizon expansion, shared vs composite; protect a simpler control; compare scaling only when learning curves support it |
| 9–10 hours | Freeze candidates before final embargoed evaluation, full-run panel with random/heuristic/native references, calibration/replay/checkpoint reload, uncertainty and failure audit; write final results and stop by deadline |

Short pilots have strict operation and campaign deadlines. Reserve integrated
training and final evaluation time even if an earlier stage is slow. Do not spend
the entire window on combat-only accuracy or on implementation. At each phase
save a usable checkpoint and development result; later stages must actually run,
not merely appear in the plan. If failures reduce the schedule, record the exact
missing comparisons and the achieved scope without redefining the objective.

Benchmark accelerator training and inference separately. Tiny rollout batches
can be faster on CPU while large updates benefit from MPS/MLX. Time GPU work
with explicit synchronization/evaluation. Warm up compilation before timing.
Limit large batch × candidate products to avoid unified-memory pressure; inspect
RSS/device memory and early throughput. Use persistent isolated simulator workers
and one central batched policy learner; workers do not each load GPU models.

Compare CPU/PyTorch MPS/MLX on equivalent candidate architectures with copied
weights, finite forward/loss/gradient checks, then elapsed-time throughput.
Prefer one supported backend for the campaign after measurement. If MLX has a
large advantage, port the selected operations with output/loss parity rather
than adopting it on marketing claims. MLX and PyTorch versions are recorded and
kept in optional experiment extras; the environment core stays dependency-free.

## Operational controls and acceptance criteria

Use one supervisor with a ten-hour UTC deadline, atomic progress/metrics and
phase checkpoints. Keep an idle-sleep assertion while the authorized run is
active; AC/thermal interruptions remain infrastructure events. The supervisor
checks actual processes and artifact progress, uses bounded retries, and never
changes a frozen scientific condition in place. Restart from checkpoints;
archive partial output before replacing it. Stop worker/GPU processes at the
allocation deadline, not just the top-level launcher.

The first campaign is complete only when there is a tested branch, a detailed
executed schedule, trained checkpoint(s), isolated combat and OOC results,
integrated learned control of both phases, held-out evaluation/baselines,
hyperparameter/architecture/throughput evidence, and an honest final report with
remaining uncertainties. A live PID or a training launch is not completion.
Winning improvements are an empirical target, not guaranteed by the time budget.

## Primary references

- [PPO](https://arxiv.org/abs/1707.06347): behavior-policy clipped updates.
- [DAgger](https://arxiv.org/abs/1011.0686): learner-state distribution correction.
- [Deep Sets](https://arxiv.org/abs/1703.06114) and
  [Set Transformer](https://arxiv.org/abs/1810.00825): set representations.
- [Potential shaping](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf):
  conditions for policy invariance; explicit terminal handling.
- [PyTorch MPS](https://docs.pytorch.org/docs/stable/mps.html) and
  [Apple Metal PyTorch](https://developer.apple.com/metal/pytorch/): local backend
  support. Performance selection comes from this machine's measurements.
