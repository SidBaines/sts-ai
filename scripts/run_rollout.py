"""Bounded compatibility smoke entry point; use sts-env for normal workflows."""
from sts_env.cli import main

if __name__ == "__main__":
    import sys
    raise SystemExit(main(["run", *sys.argv[1:]]))
