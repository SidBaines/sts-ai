#!/usr/bin/env python3
"""Verify the harness drives Gemma-4 via its NATIVE thinking toggle, not the
prompted "reason step by step in <think>" instruction.

Tokenizer-only (no GPU / no vLLM load) — it replicates exactly what
VllmJsonAgent does: `_probe_native_thinking` (does the chat template change when
`enable_thinking` is toggled?) and the `induce_reasoning` prompt decision. The
actual native-`thought`-channel *generations* come from the benchmark run.

Usage: PYTHONPATH=src python scratch/verify_gemma_thinking.py <model_id>
"""
from __future__ import annotations

import sys

from transformers import AutoTokenizer

from sts_ai.prompting import NEUTRAL_FRAME, render_action_prompt
from sts_ai.schemas import LegalAction

model = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-4-12B-it"
tok = AutoTokenizer.from_pretrained(model)
msgs = [{"role": "user", "content": "ping"}]


def templated(enable_thinking: bool) -> str:
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )


on, off = templated(True), templated(False)
native = on != off  # this is exactly VllmJsonAgent._probe_native_thinking()

print(f"model: {model}\n")
print("[1] does the chat template honour enable_thinking? (VllmJsonAgent._probe_native_thinking)")
print(f"    enable_thinking True vs False differ? {native}")
print(f"    -> reasoning_mode would be: {'native' if native else 'prompted (no native toggle)'}\n")

# show exactly what enable_thinking=True injects (first point of divergence)
i = next((k for k in range(min(len(on), len(off))) if on[k] != off[k]), min(len(on), len(off)))
print("[2] what the native toggle injects (template divergence):")
print("    common :", repr(on[max(0, i - 50):i]))
print("    ON  -> :", repr(on[i:i + 90]))
print("    OFF -> :", repr(off[i:i + 90]))
print()

# what prompt text the harness sends in each mode
la = [LegalAction(0, 0, "event option 0"), LegalAction(1, 1, "event option 1")]
p_native = render_action_prompt("GAME STATE\n...", la, NEUTRAL_FRAME, induce_reasoning=False)   # native uses this
p_prompted = render_action_prompt("GAME STATE\n...", la, NEUTRAL_FRAME, induce_reasoning=True)   # gemma-3/llama use this
print("[3] harness prompt content (render_action_prompt):")
print(f"    native mode  (induce_reasoning=False): '<think>' present? {'<think>' in p_native} | 'step by step'? {'step by step' in p_native.lower()}")
print(f"    prompted mode(induce_reasoning=True) : '<think>' present? {'<think>' in p_prompted} | 'step by step'? {'step by step' in p_prompted.lower()}")
print()

if native:
    print("CONCLUSION: Gemma-4 has a native toggle, so the harness sets reasoning_mode=native ->")
    print("  enable_thinking=True in the chat template (the model's OWN switch) AND")
    print("  induce_reasoning=False, so NO '<think>/step-by-step' instruction is added.")
    print("  => we use Gemma-4's native thinking, exactly as intended.")
else:
    print("WARNING: no native toggle detected -> harness would fall back to prompted <think>. Investigate.")
