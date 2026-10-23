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
- **Confidence intervals:** bootstrap over whole prompt identities when reporting prompt-wise differences; throughput differences additionally need repeated timed batches.
