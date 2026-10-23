# K2-Horizon serving benchmark

Run `benchmark.py --help` for the bounded raw-completions client. It requires
aiohttp and Transformers. Use the pinned model revision in `serve.sh`.

The client reads fixed JSONL rows with messages and legal_indices, renders the
effort-specific chat template, retains raw output token IDs and checks final
actions after a closed reasoning channel. No constrained decoding or malformed
output retries are applied. It stops the assistant message at its chat end token.

Server attention backend, graph execution, scheduler token budget and prefix
caching are positional settings in `serve.sh`; the maximum concurrent sequences
is 64. Run one server configuration at a time. Warmups must go to separate
output directories. Never overwrite completed cases or edit a running script.
Place a STOP file in the run root to prevent further case launches.

Prompt data and all results are in gitignored data/llm_runs/k2_throughput_20261023.
The manifest records source hashes and selection. Prompts are public observations
from env-v0.1.5.1; neither the old sts_ai harness nor a native simulator is needed
for this offline serving benchmark.
