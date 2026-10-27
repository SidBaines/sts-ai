# Local LLM tools

`prepare_decisive_states.py` selects teacher-labelled decisions from saved
non-LLM games. It preserves all labelled candidates and supports selecting
again with `--candidates` without another teacher inference pass. Run `--help`
for the corpus, teacher, counts and output options.

`launch_local.py RUN_NAME SCRIPT [ARGS...]` starts a shell script detached
with a mains-power babysitter. The script receives RUN_NAME first. Use the
run's STOP file and preserve existing output directories.

`dashboard.py` serves a read-only live monitor, refreshing every five seconds:

```sh
python3 -u scripts/llm/dashboard.py --port 8767 \
  --screen broader_training_20261026 --reference stage1_screen_ps_20261026
```

Open http://127.0.0.1:8767/. The dashboard loads each run's states from its
recorded configuration and displays only configured learning-rate arms.
It shows training samples, reasoning lengths, position checks, every saved
evaluation, full responses and logs. Historical screens have different panels;
their raw scores must not be treated as paired comparisons.

The server performs no model work. Evaluation metrics publish when the complete panel is saved. New trainer
runs also publish completed-response and generated-token counters every 30
seconds; older runs lack these counters. Long-evaluation warnings mean no new
saved record, not a confirmed process failure. The power log distinguishes
battery pauses.
The existing server's PID, URL and logs are in the run's `dashboard/` directory.
