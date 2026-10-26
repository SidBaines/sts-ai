"""Launch a local experiment shell script detached, with a power-aware babysitter.

Usage: python3 scripts/llm/launch_local.py RUN_NAME SCRIPT [SCRIPT_ARGUMENT ...]
The script receives RUN_NAME as its first argument and writes logs beneath
data/llm_runs/RUN_NAME. The babysitter prevents idle sleep on mains power and
pauses the job group on battery. It never touches unrelated processes.
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
    screen, script, *arguments = sys.argv[1:]
    script = (REPO / script).resolve()
    if not script.is_file():
        raise SystemExit(f"Script not found: {script}")
    root = REPO / "data" / "llm_runs" / screen
    root.mkdir(parents=True, exist_ok=False)
    null = subprocess.DEVNULL
    with (root / "screen.log").open("x") as log:
        job = subprocess.Popen(["bash", str(script), screen, *arguments], cwd=REPO,
                               stdin=null, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--babysit", str(job.pid),
                      str(root / "babysitter.log")], stdin=null, stdout=null, stderr=null,
                     start_new_session=True)
    (root / "pids.txt").write_text(f"screen_pgid {job.pid}\n")
    print(f"screen pgid {job.pid}; logs in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
