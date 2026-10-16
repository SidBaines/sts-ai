# Fresh-start progress — newest first

**Current state:** the approved A–C study and monitoring are complete. **G1 was not met; the full fresh composite's G2 result is inconclusive.** Keep the champion as default. The old-encoder combat fallback won 222/4,096 DEV games versus 143 for the champion; this is exploratory. All experiment workers are finished. Confirmation and Step D remain unopened pending Sid's decision.

Read the [Gate C report and plots](fresh_start_gate_c_2026-10-16.md). The [full technical journal](fresh_start_progress_detail.md) preserves every previous entry, decision, setting and receipt reference unchanged. Artifacts remain in `data/rl_runs/fresh_start_20261015/`; nothing was deleted or pushed. Hard deadline: 16 October 08:00 UTC.

## 16 October 2026, 04:06 UTC — completion audit and readable handoff

Checked the approved scope against the current source, test logs, native hashes, data manifests, completed evaluation records, figures and independent reviews. All authorized A–C/E deliverables are present; negative findings are reported honestly. The audit is saved as `data/rl_runs/fresh_start_20261015/completion_audit_v1.json`.

Added this short current-state summary and condensed dated entries. The previous journal is preserved byte-for-byte in the linked technical journal. No new experiments, confirmation access or training were started.

## 16 October 2026, 03:59 UTC — Gate C report complete

Saved nine checked PNG/SVG figure families and complete tables for teacher agreement, sensitivity, boss outcomes, HP and behavior. Independent specification, quality and scientific reviews pass. The report and code were committed locally at `6d140af` and fast-forwarded into `ws/nonllm`.

The 50k-state input audit has no encoding errors or vocabulary unknowns, but has 24 ambiguous action sources, one unresolved potion slot and 4.8% unparsed history entries. These limitations are explicit. Proposed next step: untouched 8,192-world confirmation of old combat + champion OOC, only with Sid's agreement.

## 16 October 2026, 03:36 UTC — all evaluation runs finished

All planned 4,096-world evaluations and both full fight panels completed without simulator faults or cuts. Full fresh G2 is inconclusive: 119 wins versus champion 143. The combat and OOC replacements individually pass the 1-point win margin, but that does not establish superiority or guarantee they combine successfully.

Two separately declared exploratory old-encoder combinations finished with 222 wins (champion OOC) and 196 (new OOC). Their outcomes do not change the original gates. Report and figure checks continued.

## 16 October 2026, 02:07 UTC — label advantage not established

On 28,676 paired TEST combat states, new and old agreement is 58.68% and 58.35%; difference +0.33 points, 95% CI [−0.27, +0.93]. G1's copying criterion is not met. The fight panel is still finishing.

Combat replacement: 131/4,096 wins, narrowly non-inferior. OOC replacement: 161/4,096, also non-inferior on wins but lower mean floor. The composite and remaining scores are still running. All selections were frozen before TEST.

## 16 October 2026, 01:40 UTC — shared corpus and real warmups

Completed 2,000 label-source worlds: 159,707 usable combat labels and 247,614 OOC labels, with three search abstentions excluded. Both encoders use identical raw labels and whole-world splits; full combat FIT has 113,042 rows. Act 2/3 event fights are absent from this sample and remain a coverage gap.

All 2,000 no-query champion trajectories match their teacher-query counterparts exactly. Independent teacher-repeat agreement is 71.01% [68.95, 73.08], a reference rather than a strict ceiling. Thousands of supervised updates are underway; validation catches small-data overfitting.

## 16 October 2026, 00:00 UTC — fresh baseline and frozen panel design

The environment 0.1.5 champion baseline is 143/4,096 wins (3.49%) and mean floor 35.17, with no faults or cuts. The new 1,024-world fight bank contains 16,009 starts.

Before fitting, the panel was redesigned for efficiency: one hashed fight per source world and room group, with 300 Act 2 boss worlds × eight resamples plus unsalted controls. Both students will use the same frozen 1,006-start grid. Labels are collecting in interleaved world shards; every planned evaluation will be retained.

## 15 October 2026, 23:29 UTC — baseline collection and measured performance

Launched the 4,096-world baseline and fresh fight-bank collection with detached supervisors, finite RPC deadlines, AC-power checks and STOP controls. Native control trajectories, teacher votes and resamples match after the collector repairs.

Matched collector timing is 420 decisions/s for the champion versus 336 for the structured model. Shrinking the model barely helps, so retain 128×2. A bounded optimizer comparison favors MPS batch 128. Learning-rate choices and validation-only selection rules were fixed before fitting or TEST.

## 15 October 2026, 23:11 UTC — native adoption verified

AC power returned. Fresh core and research builds pass bounded smoke, fresh-process determinism and replay. The research suite passes with two documented historical-artifact skips; the additional core suite has 275 passes and no skips. Both module hashes are recorded.

Real observations exposed native-format cases missed by synthetic fixtures. The fixes are being checked before labels or models are accepted. Original failures and corrected evidence are preserved.

## 15 October 2026, 22:32 UTC — real-format encoder repairs

Independent review rejects the first encoder prototype for boss-map edges, potion slots, public card counters, curse costs and event/reward formats. Repairs add source-faithful regression cases; no scientific fit has used the prototype.

The adapter now derives potion inventory only from the original public menu before hiding OOC drink/discard choices. Ordinary indices and combat potions remain unchanged. Native work is still paused on battery while implementation and reviews continue.

## 15 October 2026, 22:20 UTC — adoption committed; native validation pending

Committed the reviewed environment merge. All 488 pure host tests pass. Native validation is waiting on AC power, so no baseline, labels or training have started.

Separate builders implement encoder, collector and supervised tools. Review also finds an inherited build-provenance risk; a fail-closed native helper is being prepared without modifying the released environment core.

## 15 October 2026, 21:59 UTC — implementation and controls

Prepared isolated builder worktrees and independent reviews. The native pipeline is detached with power, deadline, disk and STOP controls. Existing separate build tools avoid installing into the frozen research runtime.

Fixed the whole-world split and disjoint namespaces before data collection. Corrected the fight-panel design to copy and salt the context immediately before battle initialization, preserving actual entry effects. Baseline resource checks leave a 6 GiB disk reserve.

## 15 October 2026, 21:50 UTC — pre-run decisions

Sid authorized A–C plus monitoring, with research judgment, a hard 08:00 UTC deadline and at least 50k combat labels if collection is truncated. Preserve evaluations, artifacts and worktrees; keep commits/integration local.

**G2 rule fixed before evaluation:** use a paired whole-world two-sided 95% win-difference interval and a 1-percentage-point margin. PASS if lower > −1 point; otherwise FAIL if upper < 0; otherwise INCONCLUSIVE. Separately flag clear decline even when the margin passes. Other metrics have intervals without margins. G2 informs Sid's Step D decision; it never starts Step D automatically.

**Potion rule:** hide only OOC drink/discard from both heads through Gate C; retain combat potions and ordinary action indices. Record filters in every receipt. Visibility changes are a Step D decision.
