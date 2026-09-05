#!/bin/zsh
# Pause a long local run on battery, resume on AC. Owns the caffeinate
# assertion (only held while running on AC). Exits when the pipeline exits.
#
# The 2-consecutive-bad-reads debounce keeps a momentary pmset flap (or an
# adapter renegotiating) from stopping the run; the asymmetric low-charge
# thresholds (resume >=10%, don't pause an already-running job until <8%)
# stop it oscillating at the boundary.
PID=${1:?usage: power_babysitter.zsh <pipeline-pid> <status-file>}
STATUS=${2:?usage: power_babysitter.zsh <pipeline-pid> <status-file>}
CAF_PID=""

note() { print -r -- "[$(date '+%H:%M:%S')] babysitter: $*" >> "$STATUS" }

paused=0
bad_ticks=0
note "started (pid $$); managing pipeline $PID"
batt_pct() { pmset -g batt | grep -o '[0-9]*%' | head -1 | tr -d '%'; }

while kill -0 $PID 2>/dev/null; do
  pct=$(batt_pct); pct=${pct:-100}
  if pmset -g batt | grep -q "AC Power" && { (( pct >= 10 )) || { (( ! paused )) && (( pct >= 8 )); }; }; then
    bad_ticks=0
    if (( paused )); then
      kill -CONT -$PID 2>/dev/null || kill -CONT $PID
      paused=0
      note "AC power -> resumed pipeline"
    fi
    if [ -z "$CAF_PID" ] || ! kill -0 $CAF_PID 2>/dev/null; then
      caffeinate -i -w $PID < /dev/null > /dev/null 2>&1 &
      CAF_PID=$!
      note "caffeinate attached ($CAF_PID)"
    fi
  else
    bad_ticks=$(( bad_ticks + 1 ))
    if (( ! paused )) && (( bad_ticks >= 2 )); then
      kill -STOP -$PID 2>/dev/null || kill -STOP $PID
      paused=1
      note "battery power or low charge ($pct%, ${bad_ticks} consecutive reads) -> paused pipeline"
    fi
    if [ -n "$CAF_PID" ] && kill -0 $CAF_PID 2>/dev/null; then
      kill $CAF_PID 2>/dev/null
      CAF_PID=""
      note "caffeinate released (allow sleep on battery)"
    fi
  fi
  sleep 30
done
[ -n "$CAF_PID" ] && kill $CAF_PID 2>/dev/null
note "pipeline gone; babysitter exiting"
