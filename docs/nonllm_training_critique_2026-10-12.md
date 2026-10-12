# Slay the Spire nonlanguage training critique

**12 October 2026.** The current system can learn useful Act 1 play, but its longer training runs do not yet provide a dependable route to full-game improvement. The next substantial run should follow repairs to action identity and public feature coverage, a short experiment on out-of-combat credit assignment, and much earlier checks for cumulative policy drift. Simply extending the current PPO run is poorly justified.

This review covers `codex/nonllm-curriculum`, `codex/long-curriculum`, and `codex/long-curriculum-v2`: their experiment histories, relevant implementation, original discussion, saved metrics, and additional bounded CPU investigations. Its scope is Ironclad, ascension 0, ordinary Acts 1–3. Proposals below have not been adopted into the paused campaigns. New evidence and executable probes are preserved on the separate `codex/nonllm-critique` branch. The remaining v2 training allocation is still 14,252.174 active seconds, about 3.96 hours.

## What the experiments establish

| Work | Evidence | Interpretation |
| --- | --- | --- |
| First campaign | 20/21 held-out Act 1 clears; 145/160 isolated Act 1 fight wins; 25/32 later-act fight wins | Useful local and Act 1 competence |
| First campaign full games | 0 victories, 20 defeats, one simulator fault on 21 requested worlds | Full-game competence remains unestablished |
| Longer v1 | Selected ancestry retained six new chunks and 511 updates after 526 jobs and 36,217 updates | Repeated rollbacks discarded most parameter learning |
| Longer v2 | 129 chunks, 4,030,949 retained transitions, 13,514 updates; two confirmations scored 0/192 versus initial 2/192 | Continuity was repaired, but retained updates regressed |
| Existing phase crossover | New OOC alone reduces Act 1 clears 151→130 and Act 2 clears 13→5 on 190 common-valid DEV worlds | OOC changes cause the broad progression loss on this panel |

Changing combat alone preserves 151 Act 1 clears and improves Act 2 clears to 22, but loses both original full victories. Isolated saved combat also improves slightly, 24→25/32. Therefore combat is neither the main broad regression nor certified safe on all learner-induced states. Mean floor and isolated battle wins cannot replace full-game evaluation. These are repeated development panels, with much thinner evidence for population win rate than for exact paired progression changes. The newly consolidated [paused handoff](/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI/docs/experiment_handoff.md) was also reviewed after its documentation commit 453c768; it preserves the same learner condition and correctly distinguishes the selected critic encoder sharing. [First final evaluation](/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/docs/nonllm_final_evaluation.md), [v1 audit](/Users/sidbaines/.codex/worktrees/long-curriculum/SlayTheSpireAI/docs/long_curriculum_learning_audit.md), [v2 crossover](/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI/docs/long_curriculum_v2_forgetting_diagnosis.md:34).

The environment extraction, masked candidate output, public-only policy inputs, isolated simulator workers, world-level splits, paired identities, atomic recovery, and explicit fault accounting are useful foundations. The first campaign also measured and fixed substantial overhead: repeated PPO encoding fell from roughly 40 seconds to less than one second per chunk. These parts should be preserved.

## Repair the legal action abstraction first

**A native replay confirms that the public interface removes mechanically different legal choices.** In TRAIN world 12022 at decision 14, the hand contains a free-once Strike and a normal Strike. Both native descriptions say cost 1. Playing the free card leaves energy 3; playing the normal one leaves energy 2 and the free card in hand. Their remaining hands and public next-state hashes differ. The public menu retains only the first description.

`_action_views()` deduplicates by description and explicitly assumes identical descriptions imply interchangeable actions. That assumption is false. Teacher raw-action mapping also falls back to equal descriptions, so supervision can be routed to the other instance. See [deduplication](/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI/src/sts_env/_backend/lightspeed.py:170) and [native descriptions](/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI/patches/sts_lightspeed_python_api.patch:943).

Expose separate public card instances, or deduplicate only after comparing complete public instance mechanics and target identity. Include effective cost, free-once, retention, upgrade/special values, and any other public effect-changing flags. Keep routing identity separate from features used for learning; arbitrary instance IDs need not become learned ordinal features. Add regressions comparing resulting energy and remaining cards, rather than checking description equality alone.

This requires a new observation/menu version and deliberate replay/data migration. Old local action indices can shift. Preserve the old release and banks; regenerate affected banks or provide verified semantic translation where sufficient source identity exists. Do not silently reinterpret old checkpoints or traces.

The observed free/normal collision is rare in the audited bank: eight states in the first 150,000 TRAIN rows. It is not an explanation for the entire v2 regression. It is still a correctness defect that additional training cannot solve. [Native reproduction](/Users/sidbaines/.codex/worktrees/nonllm-critique/SlayTheSpireAI/scripts/investigations/nonllm_critique_2026_10_12/nonllm_action_dedup_native_probe.py), [exact evidence](/Users/sidbaines/.codex/worktrees/nonllm-critique/SlayTheSpireAI/docs/assets/nonllm_critique_2026_10_12/nonllm_action_dedup_native_results.json).

## Preserve public mechanics before increasing model size

The selected composite has **two separate large, context-none networks**, roughly 2.3 million parameters each. The representation is only 256 state features and 128 features per candidate. Hashing compresses many public facts, while the word tokenizer drops numbers and selected regexes recover only some amounts. This is more serious than ordinary hash collisions: several strategically different inputs are identical even before learning.

| Controlled public difference | Encoder result |
| --- | --- |
| Enemy block 0 versus 20 with a Whirlwind/Defend/end-turn menu | Identical state and every action vector |
| Large Slime maximum HP 65 versus 69, current HP 50, attack damage 16 | Identical vectors although resulting HP 34 crosses only one split threshold |
| The Bomb countdown 2 versus 3 turns | Identical vectors |
| OOC deck Searing Blow+1 versus +5 | Identical vectors |
| Unparsed option text granting 100 versus 250 gold | Identical vectors |
| Offered Searing Blow+1 versus +5 | Identical candidate vectors |
| Synthetic map with an elite-free path versus an unavoidable elite | Identical candidate vectors |

These counterexamples demonstrate information loss, not their frequency among live errors. Synthetic fixtures were not all verified as reachable paired native states, particularly the Searing Blow+5 reward offer and toy enemies with block 20. In the separate first-30,000-row Act 1 TRAIN audit, positive enemy block occurs in 1,769/20,941 combat observations, 8.45%. The multi-choice map scan found 423 duplicate candidate-vector pairs among 4,126 map observations, but most routes were equivalent or similar. Eight pairs had different path counts; none differed in minimum/maximum elite or rest counts in that sample. Do not treat every duplicate vector as an error: equivalent duplicate Strikes and Defends are common.

The mechanisms are visible in [tokenization](/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI/src/sts_policy/encoding.py:28), [card fields](/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI/src/sts_policy/encoding.py:185), [enemy pooling](/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI/src/sts_policy/encoding.py:202), [numeric action parsing](/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI/src/sts_policy/encoding.py:289), and [reachable-route histograms](/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI/src/sts_policy/encoding.py:322).

Introduce a versioned typed encoder. Start with explicit entities and a small set encoder rather than replacing the whole learner with a large transformer:

- Cards: stable categorical identity, upgrade count, dynamic/special values, base/current/effective cost, free-once, retain, type and pile; preserve multiplicities and source-instance associations.
- Enemies: identity, current/maximum HP, block, intent damage and hit count, powers and counters together. Keep the association between an enemy's intent and its HP/powers.
- Candidate actions: action kind, precise source and target entities, typed effects/amounts, X-cost status and multi-hit semantics. Give missing or conditional quantities explicit availability flags; missing damage should not mean known zero.
- OOC: structured deck entries, relic counters, potions and event effects, including gold/max-HP changes. Avoid relying on parsing display prose for numerical mechanics.
- Maps: at least nearest rest/shop, mandatory elites, and feasible path summaries. A small DAG encoder is an alternative if inexpensive dynamic-programming summaries prove insufficient.

Keep the variable legal-candidate output and permutation tests. They solve an appropriate problem; permanent global action IDs would not repair missing semantics. Small DeepSets encoders provide a principled way to pool entities while preserving their internal fields. Bounded entity attention is a later comparison if set pooling misses important relations. [Deep Sets paper](https://arxiv.org/abs/1703.06114).

Compare representations on the same train data and live DEV tasks, with both matched samples and matched wall time. Include specific mechanics panels and learning curves, not only teacher accuracy. Distill the original public-state policy into the new encoder if useful, without pretending old feature weights are directly compatible. [Encoding probes and observations](/Users/sidbaines/.codex/worktrees/nonllm-critique/SlayTheSpireAI/docs/assets/nonllm_critique_2026_10_12/nonllm_encoding_critique_results.json), [map audit](/Users/sidbaines/.codex/worktrees/nonllm-critique/SlayTheSpireAI/docs/assets/nonllm_critique_2026_10_12/nonllm_map_alias_act3_results.json).

## Give the critic enough information

With the selected `context=none`, the critic uses only the state embedding. It receives hashed offer semantics, but cannot use quantities present only in candidate features. In a new probe, adding a targeted Strike lets the actor distinguish enemy block 0 versus 20; its Strike logit changes from −7.07 to −8.60. The critic remains exactly −0.211613685 in both states. The information needed for useful bootstrapping is unavailable to that critic. [Model implementation](/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI/src/sts_policy/model.py:45), [probe evidence](/Users/sidbaines/.codex/worktrees/nonllm-critique/SlayTheSpireAI/docs/assets/nonllm_critique_2026_10_12/sts_critic_menu_probe.json).

Test typed entity/menu context for the critic independently of changing the actor. A separate critic encoder can also prevent value updates moving actor features, but current evidence does not establish critic interference as the cause of forgetting. The actual selected critics share the **state encoder only**; they do not backpropagate through the action encoder. The historical v2 diagnosis incorrectly generalizes both-encoder sharing from the optional mean-context architecture.

Bounded backward probes found weighted critic encoder-gradient norms about 1–5% of actor norms in sampled stochastic original/current states, with small positive or negative cosine. That weakens a claim that critic gradients currently dominate. Track these measurements during updates, because relative effects can change and a small gradient can still matter in sensitive directions.

## Shorten strategic credit assignment

Gamma 1 preserves undiscounted future reward, but GAE lambda 0.95 is applied at every external microdecision. Full control inserts many combat plays between strategic choices and outcomes. A fresh 32-world TRAIN probe of the original stochastic policy collected 9,995 transitions, with no faults/cuts. Choice OOC states had a median **177 microdecisions to termination**; the direct coefficient on the final TD residual is `0.95^177 = 0.000114`. Neow states had a median 343 microdecisions, versus 97.5 OOC decisions if combat were compressed. Corresponding median direct coefficients are about 2.3e−8 versus 0.0068.

Bootstrapping can transport terminal information through the critic, so these coefficients do not mean eventual reward disappears. They do explain why an inadequate critic is especially consequential. This is the bias/variance tradeoff inherent in GAE, not evidence of an obvious terminal or truncation implementation bug. [GAE implementation](/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI/src/sts_policy/losses.py:39), [GAE paper](https://arxiv.org/abs/1506.02438).

The same probe compared values against completed realized returns: empirical explained variance is approximately 0.19 in combat and 0.20 in OOC despite a sampled weighted GAE value loss around 0.002. Current CP128 has lower realized-return error and EV around 0.48/0.38, yet loses the original probe's win. These measurements contain rollout noise and many correlated decisions per world; they are not exact expected-value calibration or a causal critic comparison. They demonstrate why small self-bootstrapped loss cannot certify useful policy improvement.

The most promising next learning experiment is **OOC training with the frozen learned combat policy**, using transitions from one OOC choice to the next. Sum the intervening rewards, preserve the same game trajectories, and bootstrap at the next OOC state. With gamma 1, reward aggregation can preserve the undiscounted objective. GAE now operates at the strategic timescale. This is different from previous native-combat OOC training: the continuation controller is the combat policy actually intended for deployment.

Keep a protected microdecision arm and compare it with an otherwise matched macro arm. Use the same frozen combat controller, sampling or greedy semantics, task-bound random draws, horizons and reward in both arms; otherwise a change in combat behavior confounds credit assignment. Log both primitive decisions and strategic training decisions. Add deterministic equivalence tests for trajectory, reward sum, terminal/truncation handling and action identity. A separate lambda 0.99/1.0 or completed-return advantage ablation can test credit without changing aggregation; expect a variance tradeoff. The old 60-second lambda controls were useful screens, not enough evidence to reject all longer-credit methods.

## Make the reward match the intended outcome

For completed monotone-floor fresh starts, the current reward is:

`return = 0.05 × final floor + 2 × victory − 1`.

Two extra average floors are worth the same expected reward as five percentage points of extra win probability. This is an explicit survival/progress objective; ranking champions by wins does not change the gradients optimizing that objective.

A synthetic calculation using the actual reward function illustrates the mismatch: always dying at floor 50 yields return 1.50; winning 40% of games at floor 51 and otherwise dying at floor 10 yields expected return 1.12. These are hypothetical policies, not observed results, but the current objective prefers the zero-win policy in that example. [Reward code](/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI/src/sts_policy/curriculum.py:106), [arithmetic evidence](/Users/sidbaines/.codex/worktrees/nonllm-critique/SlayTheSpireAI/docs/assets/nonllm_critique_2026_10_12/sts_reward_objective_probe.json).

Treat floor reward as a deliberate curriculum aid with an eventual transition toward victory. Compare smaller/tapered floor weight or sparse victory reward on easier near-completion starts once positive examples exist. Another option is a separate auxiliary progress prediction objective with a victory-based actor advantage. Proper potential shaping is different from awarding accumulated floor progress; zero-terminal potential must be handled correctly. Do not add ad hoc card, damage or block bonuses before confirming the simpler credit/reward hypotheses, because such proxies can create new strategies unrelated to winning.

## Protect competence without preventing improvement

V1's rollback mechanism repeatedly selected an Act 2 leader and erased short branches; it even rejected a two-win full-game candidate because its Act 2 clears were lower. The learner/champion separation in v2 is correct. Its remaining weakness is that a small local PPO change can accumulate into large original-policy drift. Existing fixed TRAIN probes measured original-to-learner KL up to 0.79 in combat and 0.29 in OOC while local logged mean KL was roughly 0.0049.

For the next short pilot, restart from the original, freeze the **whole combat subtree**, and optimize OOC conservatively. One PPO epoch and LR 1e−5 or 3e−5 are reasonable bounded controls against the current LR 1e−4/four-epoch condition, not established optimal settings. Use separate phase optimizers and explicit phase losses/normalization. Add a frozen-original distribution rehearsal loss on fixed TRAIN observations stratified by act and OOC screen. Measure mean/tail KL, total variation and changed greedy choices; monitor value and actor gradients separately. PPO's local ratio objective is not a constraint against the original policy. [PPO objective](/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI/src/sts_policy/losses.py:64), [PPO paper](https://arxiv.org/abs/1707.06347).

Log final post-update policy KL separately by phase/act and on fixed TRAIN probes. The current before-update guard observes a minibatch before its next optimizer step; it does not measure the whole-buffer or final policy change after the last step. Its aggregate mean should not be presented as a hard trust-region guarantee.

Anchoring protects a starting point but can also preserve its weaknesses. Relax it only when independent outcomes justify the change; do not permanently require imitation of every original choice. Never use the two known DEV winning trajectories as gradient/rehearsal data. Their losses are valuable diagnosis, not a training curriculum.

Current severe-regression thresholds are too permissive for a pilot: a loss of at least 29/192 Act 1 clears or eight mean floors, on two distinct confirmations, is required. The first observed paired retention loss was already 21 clears. Earlier engineering pauses should trigger investigation well before that loss, without reverting to v1's automatic rollback after ordinary fluctuations. Maintain a continuous learner and a separate deployable champion; pause a damaging pilot while preserving all its evidence.

## Allocate experience to the real bottlenecks

The 90/25/13 request mixture is not a 70/20/10 training-gradient mixture. Across v2, fresh-start episodes contribute **79.85%** of retained transitions, Act 2 starts 13.87%, and Act 3 starts 6.28%. A fresh original-policy probe is 70.5% combat and 29.5% OOC decisions. Moreover, 31.3% of OOC decisions are forced choices. Forced choices have no useful actor choice gradient, but still enter value fitting and global statistics.

Normalize and report actor losses on actual choice menus; retain appropriate critic transitions. Separate phase/act/start-family sampling and optimizer exposure. Decide weights explicitly rather than inferring them from episode quotas. This is particularly important with independent heads coupled through one global advantage normalization and gradient-norm cap.

V2 training recorded 12 victories from fresh starts, three from Act 2 continuations and 54 from Act 3 continuations. Positive examples are scarce even after thousands of requests. Saved continuations also inherit the deck/relic/HP distribution of their source policy; many native-generated decks originate from scripted/random OOC decisions. Isolated later-act fight wins do not establish that those decks can beat subsequent bosses.

Build a stratified TRAIN start bank covering boss-adjacent positions, elites, AoE encounters, weak decks, strong decks, low HP and learner-induced states. Start close enough to completion that both successes and failures occur, then move earlier as competence improves. Preserve some fresh starts and old skill rehearsal throughout. Promote curriculum difficulty using measured learning and retention rather than fixed hours alone. Report conditional start success separately from fresh-game wins.

For combat, native DAgger remains useful on learner-induced states after action identity is repaired. For OOC, the previous 256-frontier lookahead teacher is insufficient to call expert supervision: it samples at most four candidates, uses a short horizon, one hidden future and scripted OOC continuation. Its 20-update corrections regressed live play. Explore longer paired action continuations under frozen learned combat, multiple public-consistent hidden futures and measured action-value gaps. Validate the teacher through live improvement before scaling imitation. [Lookahead](/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI/src/sts_policy/lookahead.py:25).

A compact public-history tracker is a later targeted experiment for known draw placement and enemy-move history. Replay starts must reconstruct that history or provide burn-in. No recurrent architecture can recover a choice removed by action deduplication or a public amount discarded before it reaches the model.

## Measure learning and explain failures

The next dashboard should make the following views prominent:

| Question | Measurement |
| --- | --- |
| Is deployed play improving? | Matched fresh full-game victories, paired gains/losses, Act 1/2/boss reach, floor survival curves; original/learner/champion separately |
| Is competence being forgotten? | Frozen TRAIN original-policy drift by phase/act/screen; matched retention; mechanics panels |
| Is reward teaching the intended behavior? | Terminal/floor/shaping reward components and advantage distributions by phase/act/start family |
| Is the critic useful? | Completed-return bias/error/EV, repeated-sample calibration, terminal-distance buckets, bootstrapped versus realized returns |
| What gradients actually ran? | Actor/critic gradient norms/cosine, pre/post clipping, phase optimizer steps, accepted minibatches and unique/repeated sample exposures |
| Why do runs die? | Room/enemy/boss, fight entry/exit HP, deck/relic/potion summaries, changed strategic choices and representative paired trace differences |
| Is execution healthy? | Requested/finished/faulted/truncated/unstarted, faulty mechanics, excluded transitions, actual wall throughput and process/checkpoint age |

Include forced-versus-choice metrics and entropy normalized or stratified by legal-menu size. Log a small representative reservoir of public transitions/trajectories, including actions, probabilities, rewards and values; per-episode outcomes alone cannot identify the first harmful strategic decision. Use compact static IDs and reservoir sampling to contain storage. Freeze representative failure cases and compare counterfactual choices on TRAIN worlds.

The existing whole-run HP cost is net starting minus ending HP. Healing at act transitions and terminal death make it a poor health-management diagnostic. Log battle/room entry and exit HP, healing, and cumulative damage separately. Deck growth, removal, scaling, AoE/block access and potion use should be read conditional on available choices, rather than as unqualified good/bad action counts.

Current `samples=len(transitions)` reports collected rows even when KL stops early. The audit infers about 13,630,113 actual optimizer sample exposures, 3.381 passes per retained transition rather than four. Four chunks leave 32,732 rows entirely unused. Log exposure directly; count collection, optimization and retained ancestry separately. [Update loop](/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI/src/sts_policy/rollout_training.py:368).

Evaluation is affordable. Twelve v2 eval jobs took 190 seconds, about 4.7% of train-plus-eval child wall time. A 192-world confirmation costs about 31 seconds, with another 19 seconds for paired Act 1 retention. Use earlier paired checks during a short pilot and fixed TRAIN drift checks every few chunks. Show stochastic-policy evaluation separately from greedy deployment; four fresh 32-world probes did not indicate that their difference alone explains regression.

Use larger fresh confirmation cohorts once a candidate looks promising, repeat learning across at least several optimizer seeds, and keep new final worlds untouched until selection freezes. Repeated DEV access is model development. For absolute small-world success rates, Wilson/Jeffreys intervals avoid the misleading zero-width bootstrap interval at all-zero successes: zero of 21 has a two-sided Wilson upper bound around 15.5%. Preserve world-cluster bootstrap for correlated fights and paired continuous comparisons. Match horizons across future DEV/final panels.

## Improve throughput where measurement supports it

Across the 129 v2 training children, actual end-to-end rate is about **1,038 valid collected transitions/second**, versus 1,116 using only the CLI's elapsed field. These buffers belong to retained learner ancestry, but some rows never entered an optimizer step. The rate is substantial; lack of useful learning is the more urgent problem.

| Measured component | Seconds | Share of training-child wall |
| --- | ---: | ---: |
| Actor encoding/forward/sampling | 1,286 | 33.1% |
| PPO preparation/update | 1,038 | 26.7% |
| Native step waits | 861 | 22.2% |
| Reset waits | 271 | 7.0% |
| Outside CLI elapsed measurement | 274 | 7.1% |

These named timers do not cover every part of child wall time. Native waits are parent wait time, not a pure measure of simulator CPU work.

Prioritize three changes:

1. **Avoid repeated string parsing for every actor decision.** A 512-observation TRAIN fixture takes roughly 0.112 seconds with fresh encoding versus 0.039 seconds with exact prepared features. Profiling counts 24,977 regex searches and 90,671 numeric-helper calls. The 2.86× difference is a fixture ceiling, not an achieved rollout speedup. Typed numeric entities and cached static card semantics can improve fidelity and reduce this cost together.
2. **Keep the learner and native pool alive across chunks.** Current chunks restart Python/Torch and eight workers roughly every 31 seconds. Outside-CLI overhead averages 2.13 seconds per chunk. A persistent learner can still write atomic checkpoints, accept explicit task batches, and remain externally killable/recoverable. Preserve native fault isolation.
3. **Benchmark ready-worker collection and larger inference batches.** Current collection synchronizes all workers before each actor batch; mean batch size is only 7.62. Measure straggler/reset cost and IPC payloads, then compare asynchronous batching or more workers on identical tasks. Sampling currently uses global Torch RNG; introduce task-bound random streams or explicitly record any sampling-order change, so asynchronous completion cannot silently alter trajectories. CPU inference/MPS learning remains a measured sensible default; a larger GPU alone will not remove parsing and IPC costs.

Pack PPO metric transfers and remove redundant GPU validation synchronizations after CPU validation, while retaining the before-update KL decision. Benchmark real updates and outcomes. Revisit candidate padding/bucketing only if measured menu-size tails justify it; prior mixed-menu tensor caching was slower and should not be enabled by assumption. [Operations evidence](/Users/sidbaines/.codex/worktrees/nonllm-critique/SlayTheSpireAI/docs/assets/nonllm_critique_2026_10_12/nonllm_operations_probe.json).

## Simplify recovery and reporting

**Preserve every confirmed or diagnostic checkpoint.** CP114, the second confirmed regressed model, was deleted by normal retention while its metrics survived. This permanently weakens further attribution. Current protection covers hourly anchors, current/best/latest-two and active jobs, not all completed confirmations. Archive their model weights by content hash; optimizer state is unnecessary for evaluation-only copies. [Retention code](/Users/sidbaines/.codex/worktrees/long-curriculum-v2/SlayTheSpireAI/src/sts_policy/long_run_control.py:307).

Fix the pause race between process creation and ownership publication before another sustained run. Defer signals across publication, reap the direct child, and verify descendants. Keep the existing unrelated-process protections. Make one authoritative current-state record feed status views: at the audited snapshot journal/progress said `operator_paused`, while generated live Markdown still said `attention_required`.

The generated live page is 118 KB over 33 lines because it prints full event outcome dictionaries. Replace these with concise typed summaries and links. The growing journal is rewritten/fsynced every two seconds; separate a small heartbeat/current state from immutable job records and append-only history. This is a scaling/clarity improvement, not the largest present runtime bottleneck.

Maintain one compact current-state document with: latest verified performance, chosen condition, rejected hypotheses, unresolved issues, next bounded experiment, and links to evidence. Keep the detailed chronological ledger separate. Expand compressed prose and avoid putting PIDs, hashes and test counts ahead of the scientific conclusion. Rename the model role `shared` to `joint_composite`: its two heads have independent encoders. Publish effective PPO parameters in one resolved configuration, including currently implicit value weight, target KL, gradient cap and AdamW settings.

Correct the historical sharing statement via a dated correction rather than silently changing frozen evidence. Tests and parameter checks establish execution properties; they do not establish game mastery or whether a policy improved.

## A concrete route to the next useful experiment

1. **Repair action identity and the highest-value public feature omissions.** Publish an explicit environment/encoder revision and preserve old artifacts. A changed entity architecture cannot directly load the old feature weights. Use a deliberate compatibility adapter, distillation or retraining to establish one common repaired warm start, then verify its competence and record a new baseline. Add transition-based regressions and the representation counterexamples as tests of distinguishability. Confirm simulator mechanics on the frequent and generated-card cases used by training; adopt the separate Discovery fix only through an explicit new pinned condition.
2. **Add the small diagnostic contract.** Phase/act choice counts, consumed samples, original-policy drift, completed-return diagnostics, per-fight HP, permanent confirmation snapshots and unambiguous pause state are the minimum. Run an instrumented no-update baseline and a short update probe before spending substantial learner time.
3. **Compare two short protected OOC pilots from the same compatible baseline.** Freeze combat, use the same reward/anchor/conservative optimizer condition, and compare microdecision versus learned-combat macro credit. Keep enough successes/failures through stratified continuation starts; retain fresh starts. Run a few optimizer seeds, saving every evaluated snapshot. Hold the encoder, initialization and combat sampling semantics fixed within this comparison. An earlier bounded credit diagnostic can retain the legacy encoder/binary and original checkpoint, but its result must remain separate from learning under the repaired interface.
4. **Continue only a retaining, improving condition.** Require no material paired early-survival regression, bounded original-policy drift, and a credible full-run or late-boss improvement signal confirmed on fresh DEV. A low loss, healthy PID, more updates or higher conditional start success is insufficient. If neither pilot improves, prioritize typed representation/strategic supervision instead of another long unchanged PPO allocation.
5. **Reintroduce combat learning carefully.** Use separate phase budgets and critic controls, learner-induced DAgger states, and whole-game retention. Study victory-focused reward/taper as a separate condition. Only after these learning curves are credible should a longer allocation or off-policy replay algorithm become the main investment.

There is no evidence that replacing PPO wholesale with SAC, DQN or a world model is currently the shortest route. Those methods can be useful later, especially for replay reuse, but would inherit the same missing actions, compressed public mechanics and weak strategic objective. The next run can have better reasons to succeed by making each of those bottlenecks measurable and testing one learning mechanism at a time.

## Investigation evidence

All new rollouts were bounded CPU-only TRAIN diagnostics; no new final cohort was opened, optimizer update performed, campaign resumed, source runtime edited, or pinned native build replaced. Native action probes use disposable subprocesses. Backward probes compute gradients only. Synthetic examples establish representational counterexamples; they are explicitly separate from live outcome estimates.

The evidence directory contains script/source/model identities, recorded native provenance, cohort worlds, raw summaries and the exact public fixtures. The [reproduction guide](/Users/sidbaines/.codex/worktrees/nonllm-critique/SlayTheSpireAI/scripts/investigations/nonllm_critique_2026_10_12/README.md) explains local artifact dependencies and commands. The [learning probe summary](/Users/sidbaines/.codex/worktrees/nonllm-critique/SlayTheSpireAI/docs/assets/nonllm_critique_2026_10_12/sts_learning_probe_consolidated.json) preserves the qualification that empirical Monte Carlo errors include noise and correlated within-world observations.

Verification on the isolated review branch: all 471 inherited tests completed successfully with one optional artifact skip on the host, and the mandatory bounded core smoke completed 21 decisions with an ordinary defeat and no error. The first restricted execution encountered only sandbox process/socket permissions; the host rerun passed. All copied script syntax, JSON outputs, manifest digests and local report links were checked. The parent independently reproduced every synthetic alias and the exact native free-once replay. Passing the inherited suite alongside this newly confirmed action bug illustrates why the next repair needs transition-based mechanics regressions.
