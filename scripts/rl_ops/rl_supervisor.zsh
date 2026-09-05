#!/bin/zsh
# Self-healing supervisor for a long unattended local GRPO run.
#
# Loop: find latest complete adapter -> launch an attempt resuming there with
# --policy-seed-salt = attempt*1000 (each restart re-rolls trajectories, so a
# state-dependent simulator UB hang is dodged instead of deterministically
# replayed) -> watch for stalls (no output-file progress for STALL_SECS while
# the run is not babysitter-paused) -> on stall, kill the attempt's process
# group and loop. Exits when all NUM_ITERS iteration adapters exist (writes
# the DONE file) or after two consecutive fast failures that produced no new
# adapter (persistent-error guard: needs a human).
#
# Resume is stateless: the resume point is recomputed from the adapters on
# disk every pass, so relaunching this script after any kind of death (reboot
# included) picks up where the run left off. Per-episode resume inside
# grpo_loop keeps a partially-finished iteration's completed episodes.
#
# Known benign failure mode: a machine sleep that the babysitter didn't cause
# (e.g. lid closed on AC) can look like a stall on wake; the kill+resume then
# just redoes the interrupted iteration with a fresh salt.
#
# Env overrides (all optional):
#   RL_OPS_DIR   ops state dir: status journal, run log, DONE file, rc files
#                (default data/rl_ops/<out-dir basename>; data/ is gitignored)
#   RL_OUT_DIR   run output dir      (default data/rl/ooc_grpo_v1)
#   RL_NUM_ITERS total iterations    (default 12)
#   RL_ATTEMPT_SCRIPT  per-attempt launcher (default ./ooc_rl_attempt.zsh)
#
# Launch DETACHED (harness/terminal background jobs get killed on this Mac);
# the DEVNULL handles are load-bearing, see the note on the attempt launch:
#   .venv/bin/python -c "import subprocess as sp; \
#     p=sp.Popen(['zsh','scripts/rl_ops/rl_supervisor.zsh'],start_new_session=True, \
#     stdin=sp.DEVNULL,stdout=sp.DEVNULL,stderr=sp.DEVNULL); print(p.pid)"
HERE=${0:A:h}
cd ${HERE:h:h}

OUT=${RL_OUT_DIR:-data/rl/ooc_grpo_v1}
OPS=${RL_OPS_DIR:-data/rl_ops/${OUT:t}}
ATTEMPT_SCRIPT=${RL_ATTEMPT_SCRIPT:-$HERE/ooc_rl_attempt.zsh}
NUM_ITERS=${RL_NUM_ITERS:-12}
STALL_SECS=1800
MAX_ATTEMPTS=10
FAST_FAIL_SECS=600

mkdir -p "$OPS"
STATUS=$OPS/run.status
DONE=$OPS/run.DONE
RUN_LOG=$OPS/run.log

note() { print -r -- "[$(date '+%m-%d %H:%M:%S')] supervisor: $*" >> "$STATUS" }

latest_complete_iter() {
  local last=-1 n d
  for d in $OUT/iter_*(N/); do
    n=${${d:t}#iter_}
    [[ -f "$d/adapter/adapters.safetensors" ]] && (( n > last )) && last=$n
  done
  print $last
}

newest_mtime() {
  local files
  files=($OUT/iter_*/rollouts/*.jsonl(N) $OUT/iter_*/adapter/trainer_log.json(N) $OUT/iter_*/pg.jsonl(N) $RUN_LOG(N))
  if (( ${#files} == 0 )); then print 0; return; fi
  stat -f %m $files 2>/dev/null | sort -n | tail -1
}

ensure_watcher() {
  if [[ -n "$WATCHER_PID" ]] && kill -0 $WATCHER_PID 2>/dev/null; then return; fi
  .venv/bin/python $HERE/rl_live_watcher.py $OUT $DONE >> $OPS/live_watcher.log 2>&1 &
  WATCHER_PID=$!
  note "live watcher (re)started (pid $WATCHER_PID)"
}

note "supervisor started (pid $$; out=$OUT ops=$OPS iters=$NUM_ITERS)"
attempt=0
fast_failures=0
WATCHER_PID=""

while (( attempt < MAX_ATTEMPTS )); do
  last=$(latest_complete_iter)
  if (( last + 1 >= NUM_ITERS )); then
    note "all $NUM_ITERS iterations complete -> DONE"
    touch "$DONE"
    break
  fi

  attempt=$(( attempt + 1 ))
  salt=$(( attempt * 1000 ))
  start_iter=$(( last + 1 ))
  resume="NONE"
  (( last >= 0 )) && resume="$OUT/iter_${last}/adapter"

  note "attempt $attempt: iteration $start_iter/$NUM_ITERS, salt $salt"
  rm -f "$OPS/attempt_${attempt}.rc"
  # DEVNULL handles are load-bearing: the detached child must NOT inherit the
  # command-substitution pipe, or $(...) blocks until the whole attempt exits.
  runpid=$(.venv/bin/python -c "import subprocess as sp; p=sp.Popen(['zsh','$ATTEMPT_SCRIPT','$attempt','$salt','$start_iter','$resume'],start_new_session=True,stdin=sp.DEVNULL,stdout=sp.DEVNULL,stderr=sp.DEVNULL); print(p.pid)")
  if [[ -z "$runpid" ]]; then
    note "FAILED to launch attempt $attempt; aborting"
    break
  fi

  zsh $HERE/power_babysitter.zsh $runpid $STATUS &
  ensure_watcher

  launch_ts=$(date +%s)
  fresh_ts=$launch_ts
  stall_killed=0
  while kill -0 $runpid 2>/dev/null; do
    sleep 60
    kill -0 $runpid 2>/dev/null || break
    state=$(ps -o stat= -p $runpid 2>/dev/null | tr -d ' ')
    if [[ "$state" == T* ]]; then
      fresh_ts=$(date +%s)   # babysitter-paused: never counts toward a stall
      continue
    fi
    m=$(newest_mtime)
    (( m > fresh_ts )) && fresh_ts=$m
    now=$(date +%s)
    if (( now - fresh_ts > STALL_SECS )); then
      note "STALL: no output progress for $(( (now - fresh_ts) / 60 )) min -> killing attempt $attempt (pgid $runpid)"
      kill -9 -$runpid 2>/dev/null || kill -9 $runpid 2>/dev/null
      stall_killed=1
      break
    fi
  done

  sleep 3  # let the attempt wrapper write its rc file / babysitter notice
  duration=$(( $(date +%s) - launch_ts ))
  rc="killed"
  [[ -f "$OPS/attempt_${attempt}.rc" ]] && rc=$(<"$OPS/attempt_${attempt}.rc")
  new_last=$(latest_complete_iter)
  note "attempt $attempt ended (rc=$rc, ${duration}s, stall_killed=$stall_killed, adapters through iter_$new_last)"

  if (( stall_killed == 0 )) && [[ "$rc" == "0" ]]; then
    continue  # clean exit: loop re-checks completion at the top
  fi
  if (( stall_killed == 0 )) && (( duration < FAST_FAIL_SECS )) && (( new_last == last )); then
    fast_failures=$(( fast_failures + 1 ))
    if (( fast_failures >= 2 )); then
      note "ABORT: $fast_failures consecutive fast failures with no new adapter; needs a human"
      break
    fi
  else
    fast_failures=0
  fi
done

(( attempt >= MAX_ATTEMPTS )) && note "ABORT: attempt cap ($MAX_ATTEMPTS) reached"
note "supervisor exiting"
