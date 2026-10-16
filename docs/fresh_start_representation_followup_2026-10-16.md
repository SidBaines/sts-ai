# Representation follow-up — 16 October 2026

This is a source/configuration diagnosis after the completed fresh-start study. No new training, whole-game evaluation or confirmation access occurred. Original results and their verdicts are unchanged. The question is why the structured model did not outperform the old representation, and whether historical training was replicated.

## Distinguish the historical champion from the fresh old-encoder control

The champion has a long lineage: original combat behavioural cloning/DAgger/PPO, then successful outside-combat macro-credit PPO, smith/elite rule distillation and further lambda 1 PPO. See [training review](training_review_2026-10-14.md). That history was deliberately not replayed in the fresh-start study.

Both overnight combat students were initialized from scratch, used the same 113,042 FIT rows/soft teacher distributions/world splits, and used the same supervised trainer. Neither inherited champion combat weights or its historical optimizer/curriculum. The old-encoder student's stronger exploratory result therefore cannot be attributed to having inherited the champion's training history. Its outside-combat partner was the same frozen champion policy used with the structured combat student.

For outside-combat play, the comparison is different: a fresh structured student copied the historically trained champion. There was no matched fresh old-encoder OOC control, so architecture versus imitation error is not isolated there.

## Concrete differences found

| Property | Fresh old-encoder combat student | Structured combat student |
|---|---:|---:|
| Trainable policy parameters |2,035,713|858,241|
| Architecture |512-wide, three-layer state/action MLPs with explicit state×action interaction|128-wide, two-layer entity transformer, recurrent recent-action branch, action/entity scorer|
| Public rule-derived action features |Card/potion glossary semantics, calculated block, draw/energy effects, lethal-attack flag, capped damage|Card IDs/type/upgrade/cost/free/retain and displayed attack damage; those legacy rule features are not carried over|
| Learning-rate candidates |0.0003 and0.001|0.0003|
| Selected learning rate |0.001|0.0003|
| Selected optimizer update |4,100|4,100|
| Example presentations at selection |523,083|519,328|

Both use AdamW with weight decay 0.01, batch 128, gradient-norm clipping 1, whole-world validation and validation-only early stopping. Minibatch bucketing differs because the structured inputs have variable entity counts, but the observed example budget differs by less than 1%. This is not evidence of an accidentally tiny or stalled new-model fit. More tuning could still help.

The old encoder was more than a lossy hash bag. It obtains public card/potion definitions from the glossary and supplies semantic features to the scorer. For example, a real reserved probe observation's `play Defend [Skill] (cost 1)` action supplies explicit block 5 to the old scorer. The structured action carries the Defend identity and card properties, but no explicit block-effect value. It must infer that effect from labelled examples, together with visible player powers. Displayed attack damage remains available to the new model; it is not missing all action-effect information.

These are public-rule features, not hidden simulator information. The new encoding gained entity distinctions while removing precomputed domain features and changing architecture/capacity. Card IDs can in principle let a model learn many of these mechanics, but the new model must learn relations the old scorer receives explicitly. It is not a strict information superset, and the study was not an encoding-only ablation. Missing semantics is a concrete, plausible cause, but its effect has not been measured by adding the features back.

## What the result does and does not establish

Held-out teacher agreement was almost tied:58.68% new versus58.35% old, paired difference +0.33 points[−0.27,+0.93]; validation soft CE was 1.295840 versus1.295787. The new model did learn the teacher about as well on this aggregate metric. Direct Act 2 boss panels showed more HP lost with the new model. Whole-game old/new point estimates differed strongly, but no direct paired old/new whole-game contrast was computed; the old fallback remains exploratory.

Average imitation loss can conceal different mistakes in decisive states, and labels came from champion-visited states rather than each student's own trajectory distribution. This makes different error placement and compounding off the source distribution plausible; no failure attribution by card, state or trajectory has yet established that mechanism.

A bounded new check loaded the selected trained models and compared vectorized CPU/MPS arithmetic against singleton CPU deployment on32 reserved, unlabelled 93e9 choices per phase. No argmax changed. Maximum MPS/CPU logit gaps were1.91e−6 combat and9.54e−6 OOC. This is a small plumbing check, not universal parity or an explanation of the outcome gap. An initial diagnostic-only script used the wrong OOC phase alias; it was corrected before producing this result and did not affect any original study run.

Critic interference cannot explain the supervised comparison through a value loss: neither fit optimized a critic loss. The new critic was independent and frozen; the old critic head was frozen while its shared features trained for the actor.

## Recommended controlled follow-up

Before attributing the result to structured representations, test a structured model that preserves the legacy public action features, then check actor capacity and give both architectures an equally declared validation-only learning-rate search. Reuse the same immutable FIT/validation corpus for controlled comparisons, keep checkpoint selection on validation, and use the exposed label TEST/fight panel as development diagnostics. Any final generalization claim needs fresh evaluation worlds after freezing the candidate. Separate feature, capacity and optimization changes so their effects can be identified. A fresh old-encoder OOC imitation control would isolate the outside-combat question.

The strongest existing candidate remains old-encoder combat plus champion OOC, pending untouched confirmation. These diagnostic findings strengthen the case for a controlled hybrid-feature experiment; they do not change the original gates or authorize further training.

Evidence is preserved in `data/rl_runs/fresh_start_20261015/representation_followup_v1/`: exact parameter/budget/configuration comparison, source hashes, and the bounded trained-model parity script/output. See [the completed report](fresh_start_gate_c_2026-10-16.md) for all original outcomes.
