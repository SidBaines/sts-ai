"""Launch run_screen.sh detached, with a power-aware babysitter.

The screen runs in its own process group (it survives the launching shell). The
babysitter, also detached, holds `caffeinate -i` while the Mac is on mains power
and pauses the whole group with SIGSTOP on battery, resuming with SIGCONT when
power returns. Paused time is spent stopped, so progress records stay valid.
Usage: python3 launch_screen.py SCREEN_NAME LR [LR ...]
"""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def on_mains_power() -> bool:
    return "AC Power" in subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout


def babysit(pgid: int, log: Path) -> None:
    caffeinate, paused = None, False
    with log.open("a") as stream:
        while True:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                break
            if on_mains_power():
                if paused:
                    os.killpg(pgid, signal.SIGCONT)
                    paused = False
                    stream.write(f"{time.strftime('%F %T')} mains power: resumed\n")
                if caffeinate is None or caffeinate.poll() is not None:
                    caffeinate = subprocess.Popen(["caffeinate", "-i", "-w", str(pgid)])
            else:
                if not paused:
                    os.killpg(pgid, signal.SIGSTOP)
                    paused = True
                    stream.write(f"{time.strftime('%F %T')} battery: paused\n")
                if caffeinate is not None and caffeinate.poll() is None:
                    caffeinate.terminate()
                caffeinate = None
            stream.flush()
            time.sleep(30)
        stream.write(f"{time.strftime('%F %T')} screen finished; babysitter exiting\n")
    if caffeinate is not None and caffeinate.poll() is None:
        caffeinate.terminate()


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--babysit":
        babysit(int(sys.argv[2]), Path(sys.argv[3]))
        return 0
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    screen, rates = sys.argv[1], sys.argv[2:]
    root = REPO / "data" / "llm_runs" / screen
    root.mkdir(parents=True, exist_ok=False)
    null = subprocess.DEVNULL
    with (root / "screen.log").open("x") as log:
        job = subprocess.Popen(["bash", str(HERE / "run_screen.sh"), screen, *rates], cwd=REPO,
                               stdin=null, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--babysit", str(job.pid),
                      str(root / "babysitter.log")], stdin=null, stdout=null, stderr=null,
                     start_new_session=True)
    (root / "pids.txt").write_text(f"screen_pgid {job.pid}\n")
    print(f"screen pgid {job.pid}; logs in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
