# Training worlds and starting positions — bounded snapshot

At **2026-10-12 12:04:30 UTC**, committed chunks **0–166** (167 chunks) had requested **21,376 episode attempts from 10,320 distinct TRAIN worlds**. This snapshot excludes later/live pending chunks and all DEV/FINAL requests. All 10,320 worlds had started; 10,257 had at least one healthy completed episode. There were 99 simulator-fault attempts across 92 worlds, zero truncations and zero unstarted attempts. Faults are separate from ordinary game defeats.

| Actually requested start family | Episode attempts | Distinct worlds | Exact start |
| --- | ---: | ---: | --- |
| Fresh | 9,686 | 9,686 | Act 1, floor 0, Neow/initial bonus decisions |
| Saved native frontier | 4,216 | 634 | First saved Act 2 battle, floor 18 |
| Saved learner frontier | 7,474 | 2,338 | First live Act 2 entry, floor 17; may still present reward/menu decisions |

Every requested learner-frontier world was already requested as a fresh world. Consequently the unique-world total is **9,686 fresh + 634 native = 10,320**, not a sum of all three table columns. Different prefixes/starts can also share a world: the snapshot has 12,658 distinct TaskSpec IDs, which are not 12,658 independent seeds.

Each 128-task chunk currently contains **70 Act 2 continuations, 19 fresh Act 2-goal runs, 26 fresh three-act runs and 13 fresh Act 1 rehearsal runs**. Thus it introduces exactly **58 new fresh worlds**, while continuations deliberately reuse earlier worlds. The 70 continuation positions are increasingly drawn from both native and learner pools, rather than being 70 new seeds.

## Available pools versus used coverage

| Available frontier pool | Act 2 worlds | Act 3 worlds | Overlap | World union |
| --- | ---: | ---: | ---: | ---: |
| Fixed native TRAIN pool | 634 | 86 | 86 | 634 |
| Healthy learner frontiers from these committed chunks | 3,121 | 264 | 264 | 3,121 |

The fixed native pool's starts are all floor 18 for Act 2 and floor 35 for Act 3. Learner captures are all first live entry at floor 17/34, respectively; they need not be combat menus. All 3,121 available learner worlds originated in fresh training requests. Available frontiers do not add new world coverage, and current Act 2 practice does not yet sample Act 3 continuation starts.

## Exact environment and split

All tasks use **Ironclad, ascension 0, ordinary Acts 1–3**, corrected environment **0.1.1**, with the learned policy controlling **both combat and out-of-combat decisions after the requested start**. Prefixes reconstruct saved native/learner history; they are not all fresh-game runs. Replay verifies public-state hash and native/Python provenance before training.

Native module SHA256: `40ab5056c17dc481e27933c4c662f7a9fa801c3f0b72b10b446e1c0b0c8670bd`. Python interface SHA256: `1fa75a1d3d7de01ceae2ba0461f8ce3cae4051e8e6ba94ff3ff1897a9af02205`.

Fresh TRAIN worlds start at 1,000,000. Each phase/round has a disjoint 100,000-integer seed block, from which only `world_split(seed) == 'train'` worlds are requested. The hash partitions nominally eight of ten buckets to TRAIN and one each to DEV/FINAL; it is not an exact 80% quota in every finite block. Fixed legacy frontier pools are also TRAIN-only. New DEV worlds begin at 200,000, and FINAL worlds at 300,000 remain embargoed; no final game states or outcomes were read for this audit.

The initial 10-active-hour allocation is 3 hours Act 2 continuation practice, 2.5 hours fuller Act 2 practice, 2 hours Act 3 practice and 2.5 hours mixed three-act practice. Later phases change mixture fractions/start goals while continuing rolling fresh-world sampling. Time allocation is exploratory, not a mastery threshold.

Evidence: [`seed_inventory_snapshot.json`](../data/long_rl_v1/preflight/seed_inventory_snapshot.json) records the exact timestamp/global-round bound, original journal SHA, task/manifest/metrics hashes, family world sets and healthy learner-file hashes. Its counts describe requested seed coverage, not universal simulator validity or successful game mastery. No source, runtime, GPU job or evaluation was changed by the audit.

## Later manifest-only coverage update

At 2026-10-12T12:30:47Z, 217 committed chunks had requested 27,776 attempts from 13,220 distinct worlds: 12,586 fresh and 634 native. Sampled learner frontier worlds remain a subset of the fresh worlds. This update counts requested coverage from immutable manifests; the earlier bounded audit above additionally checked actual starts, completion health and source files. Evidence: `data/long_rl_v1/preflight/seed_inventory_latest.json`.
