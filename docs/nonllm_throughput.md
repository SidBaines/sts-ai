# First-campaign throughput and validity evidence

The local allocation runs from 2026-10-11 20:08:46 UTC until 2026-10-12
**06:30:00 UTC**, following Sid's extension. The supervisor stops jobs at that deadline. Environment release
`env-v0.1.0` and the original research checkout remain intact.

The early measurements below were taken on the first-release backend. Active native-assisted experiments now use corrected `env-v0.1.1`; old automatic-combat evidence is quarantined ([revision audit](nonllm_native_revision.md)). On that corrected backend, [public feature reuse](nonllm_encoding_reuse.md) increased observed full-control training throughput to 1,460 accepted transitions/s and removed repeated PPO encoding, with exact CPU update parity. Retain backend identities when comparing measurements.

## Measured execution choices

The Mac has an M5 Pro, 15 CPU cores and 48 GB unified memory. Torch 2.14.0,
MLX/MLX Metal 0.30.6 and NumPy 2.5.4 are isolated optional experiment dependencies.
GPU access works outside the filesystem sandbox; a sandbox-only MLX check fails
to enumerate Metal devices, which is an execution limitation rather than a GPU
hardware failure.

An identical native 50-simulation collection panel produced 5,562 valid labels
and the same one flagged fault under every worker condition. Uncached workers
1/4/8/12 achieved 743/2,267/3,638/3,927 records per second. Exact process-local
text caches achieved 1,343/3,597/5,002/5,021. Eight workers provide essentially
the same cached throughput as twelve while leaving CPU capacity for encoding.
Paired whole-run public observations, encoder arrays, teacher labels, rewards
and diagnostics match exactly. No core environment source was edited.

`scripts/benchmark_policy.py` compares synchronized CPU, MPS and MLX candidate
policies with copied weights and checked forward parity. Its 54 conditions
finished in 14.95 seconds with zero errors. These are synthetic compute results,
not end-to-end training rates:

| Training batch | Model | CPU examples/s | MPS examples/s | MLX examples/s |
| --- | --- | ---: | ---: | ---: |
| 512 | Small | 107,085 | 405,434 | 254,699 |
| 512 | Medium | 46,876 | 180,705 | 132,187 |
| 512 | Large | 13,219 | 51,322 | 61,248 |
| 2048 | Small | 127,541 | 518,631 | 362,433 |
| 2048 | Medium | 51,375 | 190,116 | 155,951 |
| 2048 | Large | 14,349 | 52,479 | 68,546 |

At batch 32 the small model's CPU inference reached 136,047 examples/s versus
54,754 on MPS. Choose CPU for small rollout batches and MPS for large updates;
verify the resulting device route on genuine PPO updates. MLX has a large-model
advantage but the first campaign uses the existing tested PyTorch learner,
rather than maintaining two online-training implementations.

The first real small-policy BC jobs processed 6.96 million/6.27 million examples
in 80.4/79.0 training seconds at batch 1024 (about 86.5k/79.3k examples/s), before
the metric-transfer optimization. Data loading, encoding and development
evaluation made total job time about 94 seconds. Synthetic GPU rates alone
would overstate actual throughput. Subsequent code batches GPU metric transfers
and avoids redundant categorical validation synchronizations in BC; PPO encodes
each immutable collected observation once before reusing it across epochs.

CPU collation measurements rejected an apparently faster tensor-cache default:
mixed-menu batches took 0.154s versus NumPy's 0.135s across 100 collations.
Same-size menus benefited (0.00848s versus 0.0261s), so tensor staging and
action-count bucketing remain explicit optional screen conditions. No samples
or legal candidates are dropped. Bucketing changes batch composition and must
be logged separately.

Readonly profiling of 20 Act 2/3 starts found 7,852 prefix actions consumed 4.63s
of reset RPC time, versus 0.025s for 20 actual steps and 0.028s for 20 native
200-simulation teacher queries. Python deepcopy/glossary work dominated replay.
A prototype that renders only the final replay frontier took 0.72s and matched
all 20 final public hashes, teacher labels and following observations. This is
not yet a matched RPC production speed claim; the optional implementation must
retain per-step undefined-behavior, termination and OOC-preview checks.

## Data and initial development evidence

The Act 1 bank contains 49,944 legal teacher decisions and 2,245 replayable
starts from 252 valid worlds; four flagged undefined-behavior worlds are
quarantined. Valid world counts are 207 train, 25 development and 20 final.
Twenty-five sampled starts reproduced both public hashes and native labels.

The separate 256-world Act 3 enrichment contains 68,520 labels and 2,983 starts
from 246 valid worlds, with ten separately flagged faults. Actual battle counts
are 2,142 Act 1, 718 Act 2 and 123 Act 3; only 15 source worlds reached Act 3.
There are eight Act 3 boss starts: seven training, one development and zero
final. This is inadequate to claim final boss generalization. Preserve the thin
coverage and use independent full-run evaluation rather than inventing a boss
test cohort.

A separate train-only OOC lookahead pilot labels 256 public frontiers by four
privileged continuations of at most 30 OOC decisions. It strictly improved its
explicit proxy over scripted continuation on 89 sources; ties retain the script.
The proxy includes act clear/death, floor progress and endpoint HP. Act-clear
healing can saturate endpoint HP, so proxy improvement is not an expert-play
claim; promote it only after true held-out outcome evaluation.

The first two small BC policies each won 42/48 matched development battles
across 21 worlds, with zero surfaced faults/truncations. On the exact same task
identity set, random won 28/48, the public heuristic 42/48 and privileged native
search 44/48. None of these runs accessed final worlds. Small development panels
have substantial uncertainty and do not establish full-run competence.

## Known fault-observability limitation

Explicit full-control battles expose the backend's undefined-behavior flag and
are quarantined when it is raised. The current released auto-native-combat
binding does not latch its temporary BattleContext flag after native playout.
Thus zero surfaced hybrid/OOC/lookahead faults does not prove that every native
playout was free of flagged undefined behavior. Keep this qualification on
hybrid results and investigate detection through existing APIs before changing
the frozen native build. No held-out failed world may be silently replaced.

## Recovery record

The first evaluation invocation encountered a missing CLI cache flag. Both
completed learned checkpoints were preserved; the dispatcher was paused while
the flag/device/reward support was implemented, then failed evaluations were
rerun successfully. Saved BC jobs were reused. The dispatcher now logs each
condition separately, stores atomic checkpoints and maintains the original
deadline. Tests before first GPU training: 228 passed plus the bounded release
smoke. Later optimization changes require their own regression and device checks.

## Real leading-model inference confirmation

An ABBA CPU/MPS/MPS/CPU experiment on the leading large checkpoint used
48 matched development fights with eight persistent cached/fast-replay workers.
All four runs produced 45 wins and 687 transitions. Warm CPU/MPS collection took
0.962/1.395 seconds; actor plus bootstrap timings were 0.375/0.787 seconds.
Cold reset/Metal initialization costs are reported separately in the raw rows.
This confirms CPU rollout inference for this architecture; large-batch updates
remain on MPS. Artifact: `data/nonllm_v1/benchmark/real_large_inference.json`.
