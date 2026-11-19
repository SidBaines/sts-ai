"""Shared polling and admission logic for Act 3 combat specialists.

The caller supplies the queue interface, arm-specific world rosters and journal.
Training and evaluation always use the ordinary checkpoint-consuming tools.
"""

from datetime import datetime, timedelta, timezone
import json
import time

from .paired_endpoints import compare, digest
from .fresh_study import atomic_json


def read(path):
    for attempt in range(3):
        try:
            return json.loads(path.read_bytes())
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(0.1)


def admits_candidate(fight_gain, full_gain_bounds):
    """Both gates require positive paired wins, including the worst case for unknown games."""
    return fight_gain > 0 and full_gain_bounds is not None and full_gain_bounds[0] > 0


def primary_nominee(comparison, reference, alternative):
    """Nominate the alternative only when its gain survives every unknown outcome."""
    if comparison["a"] != reference or comparison["b"] != alternative:
        raise ValueError("Unexpected comparator order")
    return alternative if comparison["full_cohort_gain_bounds_pp"][0] > 0 else reference


def confirmed_schedule_advantage(primary, repeat):
    """Require the same direction on screening and fresh, unresolved-outcome-robust confirmation."""
    if {primary["a"], primary["b"]} != {repeat["a"], repeat["b"]}:
        raise ValueError("Confirmation must compare the same two schedules")
    low, high = primary["full_cohort_gain_bounds_pp"]
    supported = low > 0 if primary["b"] == repeat["b"] else high < 0
    return supported and repeat["worst_case_ci95_pp"][0] > 0


def package(active, specialist, destination):
    """Change just the boss branch. Preserve the milestone optimizer, RNG and normalizer."""
    import torch
    from sts_policy.model_agent import Act3BossActorCritic, policy_from_payload
    from sts_rl.common import save_atomic
    from sts_rl.controlled_policy import replacement_from_payload

    base = torch.load(active, map_location="cpu", weights_only=False)
    head = torch.load(specialist, map_location="cpu", weights_only=False)
    old = policy_from_payload(base)
    replacement = replacement_from_payload(head, "combat")
    model = Act3BossActorCritic(old.combat, old.ooc, replacement)
    meta = {
        **base["metadata"],
        "model_kind": model.kind,
        "act3_boss_refit": {"active": digest(active), "specialist": digest(specialist)},
    }
    payload = dict(base, model_state=model.state_dict(), model_config=model.metadata(), metadata=meta)
    policy_from_payload(payload)
    save_atomic(payload, destination)


class BossRefitJobs:
    """Advance one arm's fixed 512-world pooled-value refit and its two gates."""

    def __init__(
        self, run, prefix, *, label_worlds, bank_worlds, gate_worlds, probe, submit, lane_prefix
    ):
        self.run, self.prefix, self.probe = run, prefix, probe
        self.label_worlds, self.bank_worlds, self.gate_worlds = label_worlds, bank_worlds, gate_worlds
        self.submit, self.lane_prefix = submit, lane_prefix
        self.result_cache = {}
        self.events = {}
        self.jobs = {}
        self.on_battery = False

    def note(self, key, message):
        raise NotImplementedError("A study must supply its progress journal")

    def job(self, name, directory, tool, args, lane, *, summary="summary.json"):
        """Retry a bounded number of interrupted jobs, retaining each attempt's outputs."""
        for attempt in range(1, 4):
            suffix = "" if attempt == 1 else f"-retry{attempt - 1}"
            key = name + suffix
            folder = directory.with_name(directory.name + suffix)
            job = self.jobs.get(self.prefix + key)
            if not job:
                if self.on_battery:
                    return None
                self.submit(key, lane, tool, [*args, "--out", folder], prefix=self.prefix)
                self.note(key + "-submit", f"Queued {key}.")
                return None
            state = job["status"]
            if state in ("pending", "running", "paused"):
                if state == "running" and folder.is_dir():
                    self.job_progress(key, folder, args)
                return None
            if state == "cancelled":
                raise RuntimeError(f"{key} was cancelled")
            path = folder if summary is None else folder / summary
            if state == "done" and path.is_file():
                if path not in self.result_cache:
                    result = read(path)
                    if result.get("status", "completed") == "completed" and not result.get("error"):
                        self.result_cache[path] = True
                if path in self.result_cache:
                    return folder
            self.note(
                key + "-incomplete", f"{key} ended without a complete result. Its outputs were retained."
            )
        raise RuntimeError(f"{name} failed three attempts; inspect its preserved records")

    def job_progress(self, key, folder, args):
        # Read only the tail: label collection writes hundreds of heartbeats.
        path = folder / "progress.jsonl"
        if path.is_file() and "--n" in args:
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 16384))
                tail = stream.read().split(b"\n")[1:-1]
            if tail:
                r = json.loads(tail[-1])
                n = args[args.index("--n") + 1]
                completed = r.get("recorded", r.get("completed", 0))
                quarter = min(4, 4 * completed // n)
                if quarter:
                    self.note(
                        f"{key}-progress-{quarter}", f"{key} returned {completed:,} of {n:,} worlds."
                    )
        status = folder / "status.json"
        if key.startswith("fit-") and status.is_file():
            info = read(status)
            step = info.get("updates", info.get("step", 0))
            if step:
                self.note(f"{key}-steps-{step // 1000}", f"{key} reached {step:,} optimizer updates.")

    def evaluate_one(self, name, label, checkpoint, start, count, lane):
        args = [
            "--checkpoint",
            checkpoint,
            "--world-start",
            start,
            "--n",
            count,
            "--processes",
            7,
            "--games-per-process",
            8,
            "--progress-every",
            256,
        ]
        return self.job(
            name + "-" + label,
            self.run / name / label,
            "scripts/rl/fast_eval.py",
            args,
            f"{self.lane_prefix}-eval-{lane}",
        )

    def paired(self, name, labels, checkpoints, start, count):
        result = self.run / (name + ".json")
        if result.exists():
            return read(result)
        folders = []
        for index, (label, ck) in enumerate(zip(labels, checkpoints)):
            folders.append(self.evaluate_one(name, label, ck, start, count, index))
        if not all(folders):
            return None
        stats = compare(*folders, start, *labels, *checkpoints, count)
        stats["checkpoint_sha256"] = {label: digest(ck) for label, ck in zip(labels, checkpoints)}
        stats["folders"] = list(map(str, folders))
        if digest(checkpoints[0]) == digest(checkpoints[1]) and (stats["a_only"] or stats["b_only"]):
            raise ValueError(
                "Identical endpoint weights produced different completed wins on paired worlds"
            )
        atomic_json(result, stats)
        self.note(
            name + "-result",
            f"{name} finished. {labels[0]} won {stats['first']['wins']:,}, "
            f"{labels[1]} won {stats['second']['wins']:,}. {stats['paired_completed']:,}/{count:,} "
            f"complete pairs. Full-cohort gain bounds {stats['full_cohort_gain_bounds_pp']}.",
        )
        return stats

    def gate(self, i, active):
        folder = self.run / f"refit{i}"
        folder.mkdir(exist_ok=True)
        decision = folder / "decision.json"
        if decision.exists():
            saved = read(decision)
            if saved["active_sha256"] != digest(active):
                raise ValueError("Active policy changed after gate decision")
            return saved
        common = ["--checkpoint", active, "--n", 512, "--workers", 6]
        labels = self.job(
            f"labels-{i}",
            folder / "labels",
            "scripts/rl/fresh_collect.py",
            [
                *common,
                "--mode",
                "collect",
                "--world-start",
                self.label_worlds[i],
                "--teacher-probabilities",
                '{"BOSS3":1}',
                "--record-search",
                "--vote-rule",
                "pooled_mean",
                "--samples",
                24,
                "--simulations",
                1000,
                "--self-consistency-every",
                0,
            ],
            f"{self.lane_prefix}-search",
            summary="completion.json",
        )
        bank = self.job(
            f"bank-{i}",
            folder / "bank",
            "scripts/rl/fresh_collect.py",
            [*common, "--mode", "eval", "--bank", "--world-start", self.bank_worlds[i]],
            f"{self.lane_prefix}-bank",
            summary="completion.json",
        )
        if bank is None:
            return None
        panel_args = [
            "--bank",
            bank,
            "--bank-world-start",
            self.bank_worlds[i],
            "--bank-n",
            512,
            "--starts-per-group",
            '{"BOSS3":192}',
            "--resamples",
            1,
        ]
        grid = self.job(
            f"grid-{i}",
            folder / "grid.json",
            "scripts/rl/fresh_panel.py",
            [*panel_args, "--freeze-grid-only"],
            f"{self.lane_prefix}-bank",
            summary=None,
        )
        if grid is None or labels is None:
            return None
        self.note(
            f"grid-{i}-complete",
            f"Update {i} froze {read(grid)['coverage']['BOSS3']} independent Act 3 boss starts.",
        )
        if not read(grid)["coverage_sufficient"]:
            raise RuntimeError(f"Insufficient saved Act 3 boss entries at iteration {i}")
        if f"labels-{i}-complete" not in self.events:
            summary = read(labels / "summary.json")
            self.note(
                f"labels-{i}-complete",
                f"Update {i} collected {summary['combat_labels']:,} boss labels "
                f"from {summary['completed']}/512 completed games.",
            )
        probe = self.probe
        if read(probe / "completion.json")["status"] != "completed":
            raise ValueError("Unlabelled movement probes are incomplete")
        deadline = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        fit = self.job(
            f"fit-{i}",
            folder / "fit",
            "scripts/rl/fresh_train.py",
            [
                "--data",
                labels,
                "--probe",
                probe,
                "--phase",
                "combat",
                "--architecture",
                "old",
                "--old-size",
                "large",
                "--lr",
                0.001,
                "--batch",
                128,
                "--target-mode",
                "pooled_values",
                "--value-scale",
                100,
                "--steps",
                12000,
                "--min-steps",
                2000,
                "--eval-every",
                100,
                "--patience",
                6,
                "--seed",
                1,
                "--device",
                "mps",
                "--defer-test",
                "--deadline-utc",
                deadline,
            ],
            f"{self.lane_prefix}-fit",
        )
        if fit is None:
            return None
        info = read(fit / "summary.json")
        self.note(
            f"fit-{i}-complete",
            f"Update {i} specialist fit on {info['actual_fit_rows']:,} "
            f"training choices. Selected step {info['best_step']:,} on {info['validation_rows']:,} validation choices.",
        )
        fitted = fit / "best.pt"
        if digest(fitted) != read(fit / "summary.json")["selected_checkpoint"]["sha256"]:
            raise ValueError("Validation-selected head changed")
        panel = self.job(
            f"fights-{i}",
            folder / "fights",
            "scripts/rl/fresh_panel.py",
            [*panel_args, "--grid", grid, "--checkpoint", active, "--combat", fitted, "--workers", 6],
            f"{self.lane_prefix}-bank",
            summary="completion.json",
        )
        if panel is None:
            return None
        fights = read(panel / "summary.json")
        if not (
            fights["grid_complete"]
            and fights["coverage_sufficient"]
            and fights["grid_hash"] == read(grid)["grid_hash"]
        ):
            raise ValueError("Incomplete or changed fight panel")
        for group, summary in fights["groups"].items():
            wins = summary["win"]
            self.note(
                f"fight-{i}-{group}",
                f"Update {i} {group} fight gate: "
                f"active {wins['champion_mean']:.1%}, specialist {wins['replacement_mean']:.1%} "
                f"on {wins['worlds']} identical entries.",
            )
        fight_gain = fights["groups"]["BOSS3:unsalted"]["win"]["paired_difference"]
        adopt = False
        candidate = folder / "candidate.pt"
        full = None
        if fight_gain > 0:
            if not candidate.exists():
                package(active, fitted, candidate)
            full = self.paired(
                f"gate{i}", ("active", "specialist"), (active, candidate), self.gate_worlds[i], 4096
            )
            if full is None:
                return None
            adopt = admits_candidate(fight_gain, full["full_cohort_gain_bounds_pp"])
        outcome = dict(
            iteration=i,
            adopted=adopt,
            active_sha256=digest(active),
            specialist_sha256=digest(fitted),
            checkpoint=str(candidate if adopt else active),
            checkpoint_sha256=digest(candidate if adopt else active),
            fight_gain=fight_gain,
            fight_panel=str(panel),
            full_game_bounds_pp=full["full_cohort_gain_bounds_pp"] if full else None,
        )
        atomic_json(decision, outcome)
        self.note(
            f"decision-{i}",
            f"Update {i}: "
            + (
                "adopted the specialist for Act 3 boss combat."
                if adopt
                else "kept the active combat policy."
            )
            + (
                " The full-game gate was skipped because the saved-fight gate did not pass."
                if full is None
                else ""
            ),
        )
        return outcome
