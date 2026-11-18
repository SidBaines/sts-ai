"""Prepare offline, isolated teacher source trees for the production research build script."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("workstream", "teacher-source", "out", "tools-venv"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    source_tree = args.teacher_source.resolve()
    patch = Path(__file__).resolve().with_name("native-playout.patch")
    for arm in ("original", "candidate"):
        project = args.out.resolve() / arm
        project.mkdir(parents=True, exist_ok=False)
        for folder in ("scripts/rl", "patches", "src/sts_env/_backend"):
            shutil.copytree(
                args.workstream / folder,
                project / folder,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
        source = project / "external/sts_lightspeed_rl"

        def ignore(directory, names):
            return [
                s
                for s in names
                if Path(directory) == source_tree and s.startswith("build")
            ]

        shutil.copytree(source_tree, source, symlinks=True, ignore=ignore)
        (project / ".venv-native").symlink_to(args.tools_venv.resolve())
        if arm == "candidate":

            def git(*flags, env=None):
                return subprocess.check_output(["git", "-C", source, *flags], env=env)

            git("apply", str(patch))
            # Represent the candidate as a normal workstream overlay above the core patch.
            index = dict(os.environ, GIT_INDEX_FILE=str(project / "core-index"))
            git("read-tree", "HEAD", env=index)
            git(
                "apply",
                "--cached",
                str(project / "patches/sts_lightspeed_python_api.patch"),
                env=index,
            )
            overlay = git(
                "diff",
                "--binary",
                "--",
                "bindings/slaythespire.cpp",
                "include/sim/search/BattleScumSearcher2.h",
                "src/sim/search/BattleScumSearcher2.cpp",
                env=index,
            )
            (project / "patches/sts_lightspeed_rl_panel.patch").write_bytes(overlay)


if __name__ == "__main__":
    main()
