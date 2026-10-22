# RL tools

Training, collection and evaluation commands are listed in
[src/sts_rl/README.md](../../src/sts_rl/README.md).

Run `python scripts/rl/disk_report.py [DATA_ROOT ...] --out REPORT_DIR` before
large batches. The default root is this checkout's `data/rl_runs`; the default
free-space threshold is 30 GiB. Timestamped `disk_report_*.json` and
`disk_report_*.txt` contain disjoint folder totals, categories, birth-time growth
over 1/3/7 days, and cleanup candidates. Missing birth times are reported, never
replaced with modification times. Embedded search-detail size is skipped.

The default is read-only apart from writing reports. Repeat `--apply RULE` to
apply only named rules: `compress-fit-manifests`, `interrupted-fit-outputs`, or
`trainer-latest-duplicates`. Every action appends its time, rule, bytes saved and
path to `cleanup_log.tsv` in the report directory. Compression savings in dry
runs are estimates from at most 3 MiB per manifest; deletion sizes and totals
are logical file bytes, which may differ from allocated disk space.

Cleanup skips runs modified within two hours, active/pending/paused queue jobs,
open paths, hard-linked outputs, and runs containing symlinks or special files.
It rechecks eligibility before each action and disables cleanup if open files
cannot be inspected. Fits are identified by their training claim, best
checkpoint, fit folder name, or a manifest plus saved training configuration.
Completed trainers are identified by status/summary or the final `done` log event.

Manifest compression streams the original bytes losslessly. `gunzip
manifest.json.gz` restores the exact legacy file, including its original hash;
restore it before using old receipts that bind `manifest.json` by path and hash.
New fits write gzip manifests and readers accept either format.
