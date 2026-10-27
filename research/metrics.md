# Metrics and terms

- **Scope:** Ironclad, Ascension 0, Acts 1–3. Full-game win means beating the Act 3 boss.
- **Valid decision:** a response closes its reasoning, ends normally, and selects a legal index from the current menu in final JSON.
- **Valid decisions per second:** valid decisions divided by measured batch wall time, including failed and incomplete responses in that time.
- **Generated tokens per second:** all generated completion tokens divided by batch wall time; includes reasoning and malformed/truncated output.
- **Latency:** request wall time from submission to complete response; report median and 95th percentile. Batch wall time also includes client queuing.
- **Truncation:** completion ends because the output-token limit is reached; separate from missing reasoning close and transport errors.
- **Reasoning effort:** the model's low, medium or high template setting. It is not a measured token budget.
- **Concurrency:** number of simultaneous client requests to one server; the server scheduler determines actual decode batch size.
- **Prefix caching:** server reuse of earlier prompt computations. Repeated-prompt throughput is a distinct condition from fresh-prompt throughput.
- **Teacher top-choice agreement:** a valid answer equal to the teacher's most probable action; invalid answers count as disagreement. Diagnostic only, not game correctness.
- **Confidence intervals:** bootstrap over whole source worlds when reporting paired decision differences, and no interval with fewer than 10 worlds (a percentile bootstrap over a handful of worlds is far too narrow). Throughput differences additionally need repeated timed batches.

- **Training generation:** synchronous Hugging Face generation under the current unmerged LoRA adapter, with full-support sampling. It is a separate execution path from vLLM serving.
- **Update rate:** actual AdamW forward/backward steps on a fixed completion cohort, with one warmup step excluded. Synthetic advantages test throughput, not learning.
- **Fixed token probe:** 64 synthetic requests of 2,048 input and 256 output tokens, concurrency 32, ignoring EOS. It measures execution speed without natural completion-length differences.
- **Cache hit fraction:** the change in vLLM cached prompt-token hits divided by queried prompt tokens, excluding warmups. It is not the fraction of total generation work avoided.
- **Response normalization:** either average each response's generated-token objective by its own length, or divide all response token sums by the same fixed constant. This choice affects reasoning-length weighting.

- **BF16 / FP32:** bfloat16 and float32 numerical formats. Changing precision, batch shape or kernels can change numerical outputs even under the same weights and random seed.
- **HF / PEFT / LoRA:** Hugging Face model generation, the parameter-efficient fine-tuning library, and low-rank trainable adapters over frozen base weights.
- **SDPA:** PyTorch scaled dot-product attention, selected through Hugging Face. This name alone does not identify which CUDA attention kernel ran.
- **KL:** Kullback–Leibler divergence. The training loss uses a sampled nonnegative proxy to the frozen foundation-model reference, not an exact full-vocabulary KL.
- **World / group:** one environment world seed; a group repeats either a saved state or one world under different policy samples. Whole-world splits prevent related decisions crossing train and held-out sets.

- **Reasoning tokens:** generated content tokens before the first reasoning closer, excluding control tokens and final JSON. An unclosed capped trace is an observed lower bound, reported separately.
- **Pilot panels:** training, validation and untouched-test public states split by whole world; proposed counts and sampling are defined in the medium-effort pilot page. The paired game panel uses separately recorded worlds and reports every outcome category.
- **Teacher probability score:** the frozen teacher's probability for a valid chosen action, or zero for invalid/capped evaluation output. It differs from the proposed training reward, which penalizes invalid/capped output with -1.
- **Random legal action baseline:** the teacher probability expected from choosing uniformly among the current menu's legal options; equal to the mean of one over the menu size, because the teacher's probabilities sum to one over the menu.
- **Always-first baseline:** always choosing the first option of the current menu (index 0), scored the same way. Model arms are compared with it state by state.
- **Teacher top-choice baseline:** the teacher's own most probable action on each state; the highest teacher probability score attainable there.
- **First-option rate:** the fraction of valid answers that choose the first menu option.
- **First-displayed-option rate:** with prompt version 2's shuffled, letter-labelled menu, the fraction of valid answers choosing label A. Under random menu orders it cannot earn reward. A policy that chooses by content alone picks A at the mean of one over the menu size (0.27 on the stage-1 training states), so a rate above that is position bias and a rising rate is drift towards a position shortcut.
- **Prompt version 2:** the unchanged state text plus a letter-labelled menu in a fresh random order per presentation; the answer is `{"action": "<label>"}`.
- **Pass (stage 1):** one sweep through all training states (8 iterations of 4 states, 8 samples each: 256 decisions).
- **Valid-pairs-only sensitivity:** the teacher probability averaged over valid answers only, keeping a state only when both compared arms have a valid answer. It checks whether invalid answers scored as zero drive a comparison; the zero-scored version stays the main result.
- **Zero-variance group fraction:** fraction of sampled groups whose action rewards provide no group-relative policy advantage; identical preferred actions and uninformative rewards can both produce flat groups.
- **End-to-end training rate:** updated responses divided by total measured wall time, including generation, teacher lookup, old/reference scoring, updates, adapter refresh and checkpoint saving.
- **Sampling/scoring mismatch:** difference between actual sampling log probabilities and recomputed token log probabilities under unchanged weights, measured over generated tokens with prompt/padding excluded.

- **Decisive state:** a state where the teacher's most probable action leads the second by at least 0.3 probability. In the 128 prepared states: 42 of 48 out-of-combat states and 10 of 80 combat states.
- **Top-1 agreement on decisive states:** the fraction of decisive states where a valid answer equals the teacher's most probable action; the proposed primary early-learning signal.
- **Normalised teacher score:** (teacher probability of the chosen action − mean teacher probability over the legal menu) / (teacher's top probability − that mean); 0 for a random legal action, 1 for the teacher's choice, comparable across phases. Proposed, not yet reported.
- **Menu-shuffle consistency:** the fraction of states where the model picks the same action content when the menu is presented in a different order. It detects position shortcuts. Proposed.
- **Grouped generation:** generating one state's samples together as one batch, so every prompt in the batch has the same length; on the Mac this was 2–3× faster than mixing prompt lengths in one batch (the cause inside mlx-lm is not established).
- **Equal-work probe (Mac):** the same fixed-token probe as above, run with mlx-lm with no stop tokens.

- **Broader continuation panels:** 512 candidate training states from 436 worlds; fixed evaluation subsets of 64 training-pool states from 64 worlds and 96 validation states from 73 worlds. Only 14 of the 64 training-panel states were encountered before the 21-update stop. Results compare the starting stage-1 adapter with the final adapter under identical fixed menus.
- **Active wall-time estimate:** wall time between recorded file timestamps, minus overlapping battery-pause intervals from the power log. Approximate elapsed work time, not measured GPU time. Raw monotonic/performance timers include awake pauses and may exclude system sleep.
