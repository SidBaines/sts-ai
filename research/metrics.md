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
- **Teacher agreement:** matching a separately stored teacher action; diagnostic only, not game correctness.
- **Confidence intervals:** bootstrap over whole source worlds when reporting paired decision differences; throughput differences additionally need repeated timed batches.

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
