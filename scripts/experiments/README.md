# September 2026 diagnostic runners

These scripts are isolated experiments, not production harness entry points.
The protocol and results live in `docs/astra_5sep_experiments.md`.

`astra_ollama_probe.py` is a prepared fallback for the frozen 133-state and
56-state failure cohorts. Its default mode writes requests and source hashes
without making any HTTP calls. `--execute` sends real inference to the local
Ollama server. Use a fresh output directory; prepared directories cannot be
reused for execution. `--limit 3 --temperature 0 --total-seconds 180` bounds a
sanity run. The runner records `/api/version`, `/api/tags`, `/api/show`, every
request and exact raw HTTP response, and parses one assistant content response
with the shared action-text parser. It has no retries or constrained decoding.

The client is killed at its wall deadline and execution stops after a transport
or protocol failure. Killing the client does not prove server-side inference
was cancelled; check the server before starting another model job. Per-probe
seeds are requested and recorded, but backend adherence and correspondence to
MLX's batch random stream are not established. Keep fallback outputs separate
from primary MLX analyses: weights, quantization, backend, chat template and
sampling implementation change together. The cached Ollama NVFP4 model staged
for direct MLX loading is also a distinct bundle from the separately downloaded
Hugging Face affine-quantized checkpoint, including its custom model loader.

`astra_analyze_failure_replay.py` analyzes only completed 56-state replay arms
with matching prediction/report counts and frozen source provenance. Its output
is `docs/astra_5sep_failure_replay_results.json`; incomplete or absent arms remain
explicitly pending, and mismatched completed artifacts fail validation. This
cohort was selected because the historical final retry failed. New observations
are first attempts with one sample per arm/state, so recovery is descriptive,
not a repeated-sampling reliability estimate. Singleton-menu validity is action
communication; the three branching states are the only strategic choices.

Suite log filenames append `.log` to the entire arm name. Do not use
`Path.with_suffix('.log')`: decimal temperature names such as `base_temp0.0`
and `base_temp0.7` otherwise collide. The September session's pre-launch
collision and resumed attempt ledger are retained under `failure_replay/` and
the `failure_resume_plan*` artifacts; no model predictions were overwritten.
