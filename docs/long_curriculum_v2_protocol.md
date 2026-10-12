# Continuous full-game v2 protocol

This is a new study condition in `codex/long-curriculum-v2`, starting from the original morning checkpoint SHA `57f1e9613a69e31230b591ef0ff2cca27e23ddc81c3a24366cee173bc9909e70`. The paused v1 learner, checkpoints, budgets and observations remain preserved. The user authorized the remaining **18,424.127912292024 active seconds**; v1 used **17,575.872087707976 seconds**, summing to the original 36,000-second allocation. Pauses and stopped-controller gaps do not spend active time. No automatic extension is authorized.

The scientific change is the training/selection protocol: a continuous full-game learner and a separate confirmed champion replace short branches repeatedly reset to an Act 2 leader. Every TRAIN task now targets ordinary Act 3 victory. Architecture, public encoder, Ironclad ascension 0, native 0.1.1 (`40ab5056…`), Python interface (`1fa75a1d…`), floor reward 0.05, success +1, defeat −1, terminal-HP reward 0, gamma 1 and GAE 0.95 remain the initial condition. Fresh Adam starts from the original weights; subsequent normal updates retain the optimizer and latest learner weights. Logging public HP summaries does not alter the policy inputs or rewards.

## Fixed TRAIN mixture

`src/sts_policy/long_run_v2_tasks.py` exposes `sample_v2(pool, round_index=…, episodes=128, seed=11, horizon=4096, simulations=200, world_start=100000000)`. It returns canonical `TaskSpec` objects plus TaskID-bound metadata, and reexports the existing immutable task writer and validated native/learner continuation loaders.

| Component | Fraction | Default episode requests | Start | Goal |
|---|---:|---:|---|---|
| Fresh scratch | 70% | 90 | Act 1 Neow, floor 0 | Full Act 3 victory |
| Saved Act 2 | 20% | 25 | Native first battle floor 18 or healthy learner live entry floor 17 | Full Act 3 victory |
| Saved Act 3 | 10% | 13 | Native first battle floor 35 or healthy learner live entry floor 34 | Full Act 3 victory |

Quotas use exact integer largest remainders, with fresh scratch winning equal remainders; the 128-request result is exactly 90/25/13. All tasks have max act 3 and the scheduled 4,096-decision horizon. Saved prefixes keep public observation hashes, native/Python provenance, preview markers, source frontier IDs and prefix-content hashes. Native and learner source labels remain separate. Only healthy TRAIN frontiers may enter the pool; a later simulator fault invalidates captured learner frontiers from that source episode.

Fresh TRAIN worlds come from rotating disjoint 100,000-seed blocks: round `r` uses `[100000000 + 100000*r, 100000000 + 100000*(r+1))`, filtered by the existing world-level TRAIN hash partition. V2 blocks must remain below 100,000,000,000: v1 Act 2 continuation worlds were below 100 million, while its Act 2 full-start stream added a 100-billion offset (last recorded block 100,054,800,000–100,054,900,000). This bound keeps both frozen v1 streams disjoint even under an extreme later extension. Retries/resume reuse the exact task bytes and sidecar manifest. Native saved worlds can repeat across rounds and across Act 2/3 starts. Learner worlds may already belong to the fresh set: reported distinct worlds are a union, never the sum of source episode counts. Sidecars record requested episodes, distinct worlds, fresh/saved overlap, quotas, source file identities and replay provenance.

## Prespecified DEV cohorts

`fixed_v2_dev_panels(output)` creates only new DEV task definitions, not simulator states or FINAL tasks. The immutable manifest is `v2_panels.json`; each JSONL has a `.jsonl.manifest.json` sidecar. The controller protocol adds the initial checkpoint identity and original baseline metric/command hashes before training starts.

| Panel | Requested worlds | Goal | Horizon | Role |
|---|---:|---|---:|---|
| `screen_full3` | 48 | Full Act 3 victory | 4,096 | Cheap routing screen |
| `confirm_full3` | 192 | Full Act 3 victory | 4,096 | Champion confirmation |
| `retention_act1` | 192 | Act 1 clear | 2,048 | Paired safety check |

All worlds are newly declared DEV seeds starting at 400,000. Confirmation worlds are disjoint from the 48 screening worlds. Retention uses the **same ordered 192 worlds** as confirmation, with different task goals; it is not another independent 192-world sample. The total distinct DEV inventory is 240 worlds. Policy seed is 777. Initial checkpoint baselines must be recorded on all three cohorts before pilot learning. Screening alone never certifies a champion; repeated screen/confirmation access remains development selection, not held-out performance. Earlier final cohorts remain embargoed.

## Champion and fault guards

The learner always continues from its latest completed weights and warm optimizer during ordinary development fluctuations. The champion is a separately recorded confirmed snapshot. Champion ranking targets full-game victories, using paired floor progress as secondary evidence; a poor or incomplete DEV result holds the champion rather than deleting learner progress. Confirmation cadence and exact ranking/tie rules live in the signed controller protocol and decision journal.

Fault-tolerant paired checks are prespecified before observing the new baselines. For each 192-world confirmation or retention comparison, require exact requested/accounted TaskIDs, no truncations or unstarted tasks, at most `floor(0.02*192) = 3` simulator faults in each panel, and at least `ceil(0.97*192) = 187` common valid worlds. Faults are distinct invalid trajectories, not game defeats. If both panels have three disjoint fault worlds, only 186 worlds remain and the check fails. Counts of wins remain out of all 192 requested worlds, with faults separately reported. Paired win gains use common valid worlds; a supposed improvement on an original-baseline fault world cannot establish improvement over that baseline.

The severe-regression guard is an engineering pause threshold, not a confidence claim. It must occur on **two distinct learner checkpoint SHAs** with coherent valid comparisons; reusing the same SHA cannot advance the counter. The first classified severe flag immediately holds the champion, even if a floor-only tie-break would otherwise promote it; the continuous learner still advances. Pause for attention when either condition holds on two distinct flagged learner snapshots:

- **A:** paired Act 1 retention loses at least `ceil(0.15*192) = 29` clears against the fixed original baseline, and paired full-game win gain against the **pre-decision champion** is nonpositive.
- **B:** paired mean ending floor in full-game confirmation drops by at least **8 floors** against the **pre-decision champion**, and paired full-game win gain against that champion is nonpositive.

The guard evaluates the incumbent champion before any promotion, preventing a candidate from becoming its own reference. Initial full-game comparisons remain separately logged for claims against the original model. Excess-fault, censored, mismatched or otherwise invalid panels hold selection and follow bounded operational handling; they do not establish catastrophic learning regression. An attention pause preserves the candidate, optimizer, champion and active budget. It does not auto-reset or auto-resume.

## Audit and recovery

Every completed training chunk must record its actual source and output checkpoint SHA, actor update evidence, optimizer carry, TaskIDs, source family coverage and immutable task manifest. Continuous learner parentage must follow the latest accepted chunk rather than champion substitutions; normal DEV outcomes must not create cold optimizer resets. All scheduled TRAIN goals must remain 3. Source, backend, protocol, reward and cohort identities stay pinned during the condition.

Reports keep learner and champion curves separate, split fresh/native/learner outcomes by start, and retain faults/cuts/missing records in requested denominators. Logged weight changes demonstrate an update, not useful learning; compare confirmed full-game wins, paired progress and retention against the fixed original baseline, and disclose repeated-world selection. Repeated plateaus, excessive excluded transitions, parent discontinuities, cold resets or goal drift require explicit diagnosis. Do not mix v1 and v2 training curves as one condition.

Pause/resume uses the controller's explicit commands and active-time ledger; an extension requires a new user authorization. The operational controller owns process-birth proofs, isolated worker cleanup, durable atomic checkpoints, bounded retries, disk/battery pauses and retention/deletion ledgers. Root verifies these controls and launches the pilot/main run. No evaluator or dashboard may silently change learner/champion selection.


Read status or request a verified operator pause from the v2 worktree:

```sh
PYTHONPATH=src .venv/bin/python scripts/run_long_curriculum_v2.py --output data/long_rl_v2/campaign --action status
PYTHONPATH=src .venv/bin/python scripts/run_long_curriculum_v2.py --output data/long_rl_v2/campaign --action pause
```

Resume using the persisted argument vector in `data/long_rl_v2/campaign/main_command.json` from its recorded working directory. That command retains source, backend and scientific settings and removes the pilot session limit. Do not edit the journal or reset `--active-seconds`; the used allocation persists. An attention pause additionally needs explicit `--acknowledge-attention` after diagnosis. Add `--extend-hours N` only after the user authorizes that extension. Check actual learner advancement, warm-optimizer carry and parent SHA after recovery, rather than treating a live PID as proof of progress.

Focused pure CPU sampler checks:

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.unit.test_long_run_v2_tasks
```
