# WS3c spike: reuse TRL's GRPOTrainer loss, or own it?

**Question:** can we reuse TRL 1.6.0's `GRPOTrainer` for the clipped+KL policy-gradient
*update*, feeding it our externally-generated multi-step rollouts + precomputed
group-relative advantages — or do we own a small loss?

**Decision: own a small clipped+KL token loss** (a `transformers.Trainer` subclass + PEFT),
reusing `sft_format.tokenize_example` for completion masking. Use TRL/HF only for the
optimizer / PEFT / accelerate / wandb plumbing — **not** the GRPO loss path. Generation stays
ours (the streaming orchestrator; durable commitment).

## Why not reuse GRPOTrainer's loss

1. **Structural data-model mismatch.** `GRPOTrainer`'s unit is *one prompt → G completions →
   one reward each → group-relative advantage over those G completions* (grouped by
   `num_generations`). Our unit is *one world seed → K full multi-step games → one `final_floor`
   per game → group-relative advantage per game, broadcast to the hundreds of
   decision-level `(prompt, completion)` rows that make up each game*. A game is **not** one
   `(prompt, completion)`; its decisions have **different** prompts (different game states), so
   they cannot be a `num_generations` group of one prompt. We compute the advantage externally
   (per trajectory) and broadcast it to decisions — there is no clean public path to inject
   precomputed per-row advantages that bypasses GRPO's reward→grouping→advantage machinery.
2. **The external-generation hook is experimental.** TRL does expose
   `rollout_func(prompts, trainer) -> dict[str, list]`, but the docs state it "is experimental
   and this API may change or be removed at any time," and custom episode fields are **not**
   passed through to reward functions (only `prompts/completions/completion_ids/trainer_state`).
   Coupling a multi-month research effort to that is the version-fragility we set out to avoid.
3. **The loss we'd be reusing is small and we'd be fighting the trainer to reach it.** Bypassing
   generation + reward + grouping leaves only `GRPOTrainer._compute_loss` worth borrowing — a
   private method — so reuse means subclassing around private internals. Owning the loss is less
   coupling, not more code of consequence.

## What we keep from the library

PEFT `LoraConfig`/`get_peft_model`, `transformers.Trainer` (optimizer, accelerate, grad-accum,
checkpointing, wandb), the tokenizer, and the **reference formula below** (verbatim from TRL's
GRPO docs, so our owned loss matches a known-good implementation). We are not reinventing the
optimizer or PEFT — only the ~40-line loss.

## Reference formula (TRL v1.6.0 GRPO docs — implement this verbatim)

- **Advantage** (we compute it externally per trajectory, then broadcast to that trajectory's
  decision completions — i.e. `Â_{i,t} = Â_i`):
  `Â_i = (r_i − mean(r)) / (std(r) + ε)` over the group; `std`-scaling optional
  (`scale_rewards=False` ⇒ subtract mean only). This is exactly `advantage.group_relative_advantages`.
- **KL** (Schulman k3, against the frozen reference = LoRA adapter-disabled forward):
  `D_KL[π_θ‖π_ref] = π_ref/π_θ − log(π_ref/π_θ) − 1`, per completion token.
- **Loss, μ=1 (single optimizer pass per generation batch — our default):**
  `L = −(1/Σ|o_i|) Σ_i Σ_t [ (π_θ / [π_θ]_no-grad)·Â_{i,t} − β·D_KL ]`.
  The `[π_θ]_no-grad` stop-grad ratio (value 1, gradient through the numerator) means **no
  `old_logprobs` are needed at μ=1** — it reduces to advantage-weighted NLL + KL.
- **Loss, μ>1 (reuse a generation batch for several gradient epochs — clipped surrogate):**
  `L = −(1/Σ|o_i|) Σ_i Σ_t [ min( (π_θ/π_θ_old)·Â_{i,t}, clip(π_θ/π_θ_old, 1−ε, 1+ε)·Â_{i,t} ) − β·D_KL ]`.
  `π_θ_old` = an **HF-side snapshot** (a no-grad forward over the same tokenization at the start
  of the inner-epoch loop) — **never vLLM logprobs** (tokenization-alignment footgun).
- **Aggregation / length bias:** default token-mean over `Σ|o_i|` (TRL `loss_type="grpo"`);
  `dapo` / `dr_grpo` variants reduce response-length bias (divide by a constant `L`). Start with
  the token-mean; expose `loss_type` if length bias appears.
- **Config names to mirror:** `beta` (KL coeff; TRL default 0.0 — we set >0 for stability),
  `epsilon` (clip), `num_generations` (= our group size G), `num_iterations` (= μ, inner epochs),
  `scale_rewards`, `loss_type`.

## Consequence for WS3c

`train_pg_trl.py` implements the μ=1 loss first (no `old_logprobs`; offline signed-advantage =
the same loss with `Â_i = floor − baseline`), with the clipped μ>1 path behind an `inner_epochs`
knob using the HF-side `old_logprob` snapshot. CPU-testable on a tiny model: positive `Â` lowers
completion NLL, negative raises it; KL ≥ 0 and → 0 at θ=ref; clipping bounds the ratio.

Sources: TRL GRPO Trainer docs (huggingface.co/docs/trl/en/grpo_trainer, v1.6.0) and the
GRPOTrainer `rollout_func` / OpenEnv integration notes.
