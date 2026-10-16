# D4 subset composition and comparison integrity

FIT ordering now chooses the nonempty act/room group with the largest deficit
from its full-phase fraction at the next prefix. Scores use integer numerators
`k * full_group_count - used_group_count * full_phase_count`; group names break
exact ties. Within a group, worlds and each world's rows retain their existing
hash order, and each surviving world supplies one row per round. Informative combat
FIT targets supply combat quotas. OOC quotas include all targets, including forced
single-action targets; OOC fitting is full-only. The combat 5k, 20k and full sets remain exact nested prefixes
and are shared by both encoders. Raw labels, target identities and world splits
are unchanged. Newly frozen manifests include `fit_order_method` and
`fit_subset_group_counts[phase][5000|20000|full]`, keyed by act/room group. The
changed ordering and added metadata intentionally produce a new manifest digest;
archived manifests and earlier outputs are preserved.

Final TEST rows now carry their checkpoint's manifest `data_identity`, together
with the existing `source_identity`, `split`, target and inference identities.
Label comparison requires TEST and the declared CPU singleton exact-argmax
protocol, plus one nonempty matching manifest/scientific-source identity across
the entire paired cohort. These checks, exact cohort checks and target checks
finish before any interval is computed. The primary label agreement difference
remains label weighted with source-world block bootstrap.

Full-game comparison requires nonempty equal `expected_provenance` for every
paired world and an explicit `bosses` dictionary in every record. An empty dict
means no boss reached; a missing dict is invalid. Boss entries must identify an
Act 1–3 reach with boolean pass and finite nonnegative numeric entry HP. Invalid
rows are rejected before any bootstrap work.

Fight-panel differences now bootstrap equal source-world means, matching C4's
`paired_summary` point estimand even when worlds contribute different numbers of
starts. Output records `weighting: equal_world`; every start and salt from a
source world stays in that world's block. Unsalted and resampled results remain
separate, with the existing frozen grid, coverage and selection rules.

Production config validation requires a dedicated `--probe` corpus before any
output or optimizer is created. Only `validate_training_config(...,
synthetic=True)` permits omission for plumbing. The dedicated probe loader,
strict native receipt and archived adapter AST checks are unchanged. Root will
integrate C4 and verify its recollected `probe_native_schema_v2`; this tree keeps
C3 for unit fixtures. No compatibility allowlist or relaxed AST comparison was
introduced.

Old control metrics/checkpoints now report
`critic_status: head_frozen_shared_actor_features_trained`: its critic head is
frozen, while actor updates still train the shared features consumed by that
head. Entity status remains `frozen_untrained`; explained variance remains
`not_applicable` for both. Large/none from-scratch defaults, vectorized entity
math, optimizer settings and deferred TEST selection remain unchanged. The
minimal D3 selection receipt still relies on root's predeclared candidate list
and procedural validation-only argmin freeze.

The consistency CLI rejects directories without direct `worlds/*.json.gz`
shards and instructs callers to pass actual shard directories. It does not scan
unreceipted child directories or turn a sharded group root into empty success.
An explicit input with no repeat-query rows is also rejected without a result.

D4 verification uses bounded synthetic/pure unit and combined plumbing tests.
Host native, MPS, HTTP, real C4/probe integration and production fits remain with
the orchestrator. No production fit or native job was launched here.
