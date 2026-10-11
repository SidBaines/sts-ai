# Composite phase dispatch optimization

`CompositeActorCritic.forward` previously evaluated both specialized policies on
every row and selected outputs with `torch.where`. It now evaluates only the
matching phase policy for a homogeneous batch. Mixed batches gather each phase's
rows, evaluate their corresponding network, then scatter logits and values into
the original row order. Public encoder feature 7 supplies the phase; action masks
and candidate order retain their existing semantics. Model architectures,
weights, encoders and task/reward definitions are unchanged.

Three CPU regression tests cover all-combat, all-OOC and mixed batches. Outputs
and every parameter gradient match the former implementation at absolute and
relative tolerance 1e-5, tested with small and large networks. Candidate
permutation and padding invariance remain intact. Forward hooks verify that no
wrong-phase network executes. For homogeneous batches with gradients enabled,
zero-valued parameter anchors preserve inactive-network zero gradients rather
than `None`; otherwise AdamW weight decay/optimizer state could change. No-grad
evaluation omits those anchors and fully skips the inactive network. The current
candidate policies are independent across batch rows; future models using
batch-dependent normalization would need their own equivalence verification.

CPU benchmark: two large mean-context policies, batch 8, up to 16 candidates,
float32 and two CPU threads. Each condition used 20 warmups and five alternating
paired timing repeats of 50 calls per implementation.

| Batch phase | Former latency | Optimized latency | Speedup |
| --- | ---: | ---: | ---: |
| All combat | 1.749 ms | 0.880 ms | 1.99x |
| All OOC | 1.931 ms | 0.876 ms | 2.20x |
| Mixed, four rows each | 1.881 ms | 1.195 ms | 1.57x |

On the measured synthetic batches, legal logits and values were exactly equal
and argmax selections matched. The tolerance tests cover broader gradients and
mask patterns; these numbers do not assert universal bitwise equivalence across
devices. No GPU jobs, native rollout changes or final-world reads were involved.
The health-study evaluation conditions use the same frozen optimized wrapper.

Evidence: `data/nonllm_v1/benchmark/composite_dispatch.json`, including every
paired timing, source fingerprints and output differences. A source comparison
verified that only this forward method changed in `model_agent.py`.

- Former method SHA256: `95b1603abe5a2c5a043e01b7076954a728518ae10907a956e58f135670644a16`
- Optimized method SHA256: `b73aebe7657eb8ab2762749e5003afe8910466a47b9425ac4d7b82084d18ba29`

Regression entry point: `PYTHONPATH=src .venv/bin/python -m unittest
tests.unit.test_composite_dispatch -v`.
