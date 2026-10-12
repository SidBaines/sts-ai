# Public text hash cache: development-only CPU evidence

The actor still spent substantial time encoding fresh public states after PPO reused its sampled encodings. A CPU profile of 512 saved development observations found `_bag` responsible for about 29% of cumulative encoding time. The new cache stores immutable ordered `(index, sign)` terms for normalized public text. It preserves every sequential `target[index] += sign * weight` operation; it never combines hash collisions, retains observation/array objects, or caches dynamic weights.

The LRU holds at most 2,048 entries. Keys include prefix, lowercase string and target length. Text longer than 256 characters or prefixes longer than 64 characters bypass the cache. Unknown symbols remain encoded. The encoder version remains `public_candidate_v2`; the feature semantics, model, RNG, dependencies and environment are unchanged. Encoded artifact caches continue to distinguish encoder source hashes.

The fixture contains 256 combat and 256 out-of-combat observations from five DEV worlds in the corrected native011 Act 1 teacher bank, filtered by `world_split` before encoding. It differs from the earlier naturally mixed 2,048-row/11-world PPO cache fixture. Five alternating trials, three passes per trial, measured:

| Measurement | Reference | Cached | Speedup |
| --- | ---: | ---: | ---: |
| Full encoding per observation | 0.183 ms | 0.139 ms | 1.31× |
| Fresh encoding + CPU collation, batch 8, all 512 rows | 0.0925 s | 0.0737 s | 1.26× |
| Fresh encoding + CPU collation, batch 32, all 512 rows | 0.0987 s | 0.0752 s | 1.31× |

The fixture retained 1,425 cache entries: approximately 446 KB current traced allocations. Repeated public terms produced 38,921 hits and 1,425 misses. A cold full-fixture pass took 0.0737 seconds. Preencoded CPU collation took 0.00065 seconds at batch 8 and 0.00038 seconds at batch 32, so fresh encoding dominates that path. These are bounded microbenchmarks under concurrent laptop work; they do not establish an equal increase in complete episode or training throughput.

All state/action feature bytes, action indices and coverage matched the reference implementation. CPU candidate and composite networks also matched padded tensors, logits, values, greedy choices and identically seeded sampled choices exactly. Regressions cover hash collisions, negative/zero/changing weights, changing public strings, prefix/target-length keys, long-input bypass, cache eviction, array lifetime, private-field exclusion and candidate permutations. Twenty-two focused encoding/model/rollout tests passed. No simulator episodes, final worlds or GPU jobs were run.

Reproduce from the experiment worktree:

```sh
PYTHONPATH=src .venv/bin/python scripts/profile_nonllm_bag_cache.py --output docs/nonllm_bag_cache_evidence.json
PYTHONPATH=src .venv/bin/python -m unittest tests.unit.test_policy_bag_cache tests.unit.test_policy_encoding tests.unit.test_policy_model tests.unit.test_rollout_encoding_reuse
```

The adjacent JSON records exact source/encoder hashes, worlds, timings, phase counts, cache measurements and CPU parity. Encoder SHA256: `ba9cf80d896b9fb02aa4598b3c349cf96b5c0f310df8b73546630abc4b2df82e`.
