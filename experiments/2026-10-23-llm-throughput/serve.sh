#!/usr/bin/env bash
set -euo pipefail
cd /workspace/llm_throughput
backend=${1:-FLASH_ATTN}
execution=${2:-graphs}
batched_tokens=${3:-8192}
prefix=${4:-off}
extra=()
if [[ "$execution" == eager ]]; then extra+=(--enforce-eager); fi
if [[ "$execution" == no_graphs ]]; then extra+=(-cc '{"cudagraph_mode":"NONE"}'); fi
if [[ "$prefix" == on ]]; then extra+=(--enable-prefix-caching); else extra+=(--no-enable-prefix-caching); fi
exec .venv/bin/vllm serve IFM/K2-Horizon-0.9B \
 --revision 7c75f2f49e74ac97c1852e62b4c860091d6a00ea \
 --code-revision 7c75f2f49e74ac97c1852e62b4c860091d6a00ea \
 --trust-remote-code --model-impl vllm --dtype bfloat16 \
 --tensor-parallel-size 1 --max-model-len 40960 \
 --max-num-seqs 64 --max-num-batched-tokens "$batched_tokens" \
 --gpu-memory-utilization 0.80 --attention-backend "$backend" \
 --generation-config vllm \
 --host 127.0.0.1 --port 8000 "${extra[@]}"
