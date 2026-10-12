# Corrected-native OOC learning audit

Snapshot: 12 October 2026, 01:30 UTC (02:30 BST). This report concerns the
separate `data/nonllm_native_revision/campaign` run, using environment release
`env-v0.1.1`, native module SHA256
`40ab5056c17dc481e27933c4c662f7a9fa801c3f0b72b10b446e1c0b0c8670bd`.
The original native-auto OOC results remain quarantined; see
[nonllm_native_revision.md](nonllm_native_revision.md).

## Cohort and selection verification

All reference, pilot and completed long-training development files matched their
completed journal SHA256. Every evaluation requested **the same multiset of 32
task IDs**: generated OOC worlds from 10000, split dev, seed 777, horizon 1024,
Act 1, native base search 200. Requested IDs equaled episode-plus-fault IDs;
`cohort_completed=true`, with no unstarted tasks or collector-budget truncations.
These evaluate the external OOC policy with native search resolving combat; they
are not whole-model combat-and-OOC win rates. Faults remain simulator errors,
not game defeats. Scores use clears per requested
32 tasks, with explicit conservative fault coverage and a separate fault limit.

| Condition | Clears / 32 requested | Defeats | Faults | Mean valid floor | Mean valid HP cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh OOC BC reference | 12 | 20 | 0 | 15.97 | 48.72 |
| LR 1e-4, progress 0.05 | 19 | 13 | 0 | 16.59 | 31.13 |
| LR 3e-4, progress 0.05 | 23 | 8 | 1 | 15.97 | 20.29 |
| LR 1e-3, progress 0.05 | 8 | 24 | 0 | 14.94 | 60.47 |
| LR 1e-4, sparse | 17 | 14 | 1 | 16.23 | 34.77 |
| LR 1e-4, progress 0.02 | 17 | 14 | 1 | 16.16 | 34.52 |
| Long OOC chunk 0000 | 23 | 8 | 1 | 16.61 | 18.23 |
| Long OOC chunk 0001 | 25 | 6 | 1 | 16.74 | 12.39 |

At this snapshot `ooc_0002` was marked running in the journal and is omitted from
completed-run totals; no process-liveness claim is inferred from that journal alone.

The campaign selected LR **3e-4**, progress weight **0.05**, gamma 1, clip 0.1,
entropy 0.01 and terminal HP weight zero. This now follows a completed matched
screen, rather than an assumed LR. Source checkpoint was fresh `ooc_specialist_bc.pt`
(SHA256 `9e360ae31b86d2c99ca6232ff125e2495bfc59ecba8a6d78a6666529490f33a3`).
The actual checkpoint architecture is **large/no-context**, state dimension 256,
action dimension 128; the recorded effective `model_config` confirms it.

The LR 3e-4 pilot and initial reference-to-training evaluation faulted on world
10152, with `native battle playout evoked undefined behavior`. Sparse faulted on
10070, progress 0.02 on 10049. Long chunk 0000 also faulted on 10152; chunk 0001
on 10011. All are rapidly reported simulator errors after the corrected guard,
not the original ten-second hangs. Therefore statements such as “23/31” describe
only the conditional valid-game subset; the conservative requested-cohort result
is **23/32, one fault**. No broad significance claim follows from these reused
32 development worlds. Final worlds/results were not read in this audit.

## Executed learning and throughput

Pilots used identical 90-second learner budgets and initial weights, seed 11,
64 fixed training worlds from 22000, horizon 512, CPU inference and MPS updates.
Different policies and PPO KL stopping produce different sample/update counts.
“Transitions” below means valid **collected** transitions, not unique worlds or
all samples consumed by the optimizer.

| Condition | Elapsed s | Valid collected transitions | Optimizer updates | Collector s | Update s | Fault episodes | Discarded transitions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-4 / progress 0.05 | 90.11 | 78,359 | 240 | 59.75 | 28.58 | 18 | 598 |
| 3e-4 / progress 0.05 | 90.68 | 96,577 | 108 | 63.79 | 24.74 | 5 | 133 |
| 1e-3 / progress 0.05 | 90.67 | 95,556 | 50 | 67.48 | 20.93 | 11 | 260 |
| 1e-4 / sparse | 90.52 | 81,817 | 248 | 59.86 | 28.85 | 14 | 292 |
| 1e-4 / progress 0.02 | 90.57 | 85,405 | 237 | 59.65 | 29.05 | 8 | 278 |
| Long chunk 0000 | 181.25 | 192,177 | 92 | 138.10 | 39.85 | 37 | 1,331 |
| Long chunk 0001 | 181.06 | 187,541 | 85 | 139.99 | 37.75 | 41 | 1,641 |

The selected pilot processed about 1,065 valid collected transitions/second.
Its collector timing comprised actor 26.49 s, successful step waits 30.59 s,
reset waits 5.52 s, bootstrap 0.002 s and failed waits 0.028 s. The former
40–50-second timeout loss has disappeared. Reset/process replacement and policy
inference now matter; smaller architecture is an untested throughput/quality
tradeoff, not a conclusion from this screen. Preserve the selected model while
making any later bounded architecture comparison.

Training UB faults occurred on several worlds, including repeated 22047 for the
selected pilot and recurring 22058/22069 for other pilots. Whole affected episode
transitions were excluded. Fault waits were milliseconds and explicitly counted;
no fault reward or fake defeat was added.

## PPO/deferred-bootstrap checks

Stored behavior ratios changed after optimization (selected pilot ratio means
ranged 0.9948–1.0793), so this was genuine PPO rather than ratios fixed at one.
Deferred bootstrap used the next actor value for continuing trajectories; explicit
bootstrap inference appeared only at legitimate cutoffs. No evidence of a deferred
bootstrap, mask, or non-finite loss failure appeared in these records.

One zero-update round appeared at the **end** of most pilots/chunks when collection
exhausted the time budget. For example, selected pilot round 24 collected 2,560
transitions and then spent 0.454 s preparing its buffer but ran zero optimizer
steps. Long chunk 0000's final buffer had 5,544 transitions and 0.960 s preparation.
This is a budget-boundary inefficiency, not a missing-gradient bug. Reporting those
as trained samples would be misleading; a future scheduler can reserve buffer
preparation/update time rather than collect another terminal buffer without time.

The current KL guard checks **after** `optimizer.step`, while the reported KL was
computed on the parameters **before** that step. Thus an already-over-target
minibatch still makes one extra update before stopping. LR 1e-3 reached round-mean
KL 0.327 versus target 0.03 and regressed to 8 clears. A before-update KL guard
would avoid that extra update; changing it should be a recorded algorithm revision,
not silently applied to completed matched controls. The aggregate metrics alone
cannot measure each final post-update KL, so do not claim a hard KL trust region.

## Joint composition reuse audit

The current `Campaign.main` branch for an existing `joint_initial_composite.pt`
and `.composition.json` only loads the JSON. It does **not** verify the stored
output SHA, current component checkpoint paths/SHA256, or composition/wrapper/
encoder/model source SHA256 before reuse. If the OOC best checkpoint changes on
resume, this can silently reuse an obsolete OOC head. Validate these identities
before reuse and create a new condition-specific composition when they differ;
selected/frozen artifacts must remain immutable.

`compose_checkpoint` validates source digests while creating a new composite.
When both sources contain explicit training-environment provenance, it also
rejects mismatched native SHAs. Retained clean legacy combat with missing provenance
is flagged as legacy, not certified newly trained. If an explicit old/new native
transition is required, use a verified compatibility record rather than removing
provenance checks. This audit made no campaign, model, backend or CLI changes and
launched no GPU or baseline jobs.
