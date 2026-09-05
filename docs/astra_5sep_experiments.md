# Astra experiments — 5 September 2026

Branch: `astra-5sep`. Session starts 22:08 UTC, approximately one hour budget.
This exploratory session preserves production interfaces and frozen artifacts.
Raw outputs live under `data/experiments/astra_5sep/`.

## Questions and protocol, recorded before model results

1. On new diagnostic worlds 1000–1031, how much do elementary OOC rules and
   combat search budget affect floor? Paired random/heuristic × 50/500 search
   simulations, one rollout/world/arm, all starts accounted for, per-process
   timeout. Bootstrap over worlds; report errors and wins as well as floor.
2. How often do those trajectories encounter missing public information?
   Observe prompts and public map DAGs without changing policy decisions.
3. Does the base model or an RL checkpoint make different decisions when exact
   legal JSON is enforced? Freeze 133 OOC prompts from random/search50 traces,
   stratified by screen (up to16 each, selection seed905). Compare free and
   finite-menu constrained decoding on exactly the same observations. Validity
   under constraints is guaranteed by construction, not evidence of competence.
   Base, pre-instability RL (`iter_1/adapter`), and later RL (`iter_5/adapter`)
   are selected before these probe results. Greedy generation, no thinking;
   `action_text` and compact `action_only` contracts. Additional live rollouts
   will assess selected interventions; any adaptive selection remains exploratory.
4. Can cheap OOC rollout-value labels provide an informative diagnostic?
   Replay each frozen prompt, verify the exact public frontier, try legal actions,
   and continue with fixed simple policies. Label values are continuation-policy
   dependent and privileged, not optimal values or policy-visible information.
5. How much do format-dead episodes and extreme KL estimates influence the saved
   RL updates? Reconstruct existing trainer row order and analyze coefficient
   allocation and counterfactual group baselines. This is descriptive sensitivity
   analysis, not an alternative unbiased PG estimator.

The embargoed final Nob cohort and historical final eval protocol are untouched.
New diagnostic worlds become development data through this session.

## Results

Results and conclusions will be filled after the experiments complete.
