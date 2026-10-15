"""Fresh-study public inference and launch controls (no native imports at import time).

Both encoders receive the same public observation after menu filtering, with
OOC potion inventory recovered from the original public menu beforehand.
The potion filter was checked against env-v0.1.5's OOC ``describeAction``:
``drink potion NAME (slot N): ...`` / ``discard potion NAME (slot N)``.
The separate legacy dead-key/recall filter is applied identically to all heads.
Entity integration contract: encode_observation, collate_observations, and an
actor_logits(batch) (one dict); legacy actor_logits receives *batch (a tuple).
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import threading
import time

from sts_policy.curriculum import public_hash
from sts_rl.recording import begin_run, file_identity

PROTOCOL = "fresh_start_v1"
FILTER_ID = "ooc_potion_drink_discard_hidden_v1"
DEAD_FILTER_ID = "legacy_ooc_dead_key_recall_v1"
PUBLIC_CONTEXT_ID = "ooc_potion_public_menu_inventory_v1"
INFERENCE = "single_observation_exact_argmax_actor_only_v1"
MIN_DISK = 6 * 1024**3
_OOC_POTION_LABEL = re.compile(r"(?:drink|discard) potion (.+?) \(slot ([0-9]+)\)(?::.*)?")


def filter_observation(observation):
    """Preserve entry identity, menu order, native indices and all public text."""
    if observation.get("phase") == "combat":
        return observation
    legal = observation["legal_actions"]
    kept = [a for a in legal if not a.get("description", "").startswith(
        ("drink potion ", "discard potion "))]
    if not kept:
        raise ValueError("Potion filter emptied legal menu")
    return observation if len(kept) == len(legal) else {**observation, "legal_actions": kept}


def policy_observation(observation):
    """Derive only visible OOC slot/name pairs, without changing raw objects.

    The native OOC inventory text can stop before an occupied slot after a gap.
    Recover that slot from the raw drink/discard labels before hiding choices.
    A supplied adapter field is never evidence: always rederive it from the menu.
    """
    if observation.get("phase") == "combat":
        return observation
    inventory = {}
    for action in observation["legal_actions"]:
        match = _OOC_POTION_LABEL.fullmatch(action.get("description", ""))
        if match:
            name, slot = match[1].strip(), int(match[2])
            if not name:
                continue
            if slot in inventory and inventory[slot] != name:
                raise ValueError(f"Conflicting public potion names for slot {slot}")
            inventory[slot] = name
    observation = {**observation, "adapter_public_ooc_potions": [
        {"slot": slot, "name": inventory[slot]} for slot in sorted(inventory)]}
    observation = filter_observation(observation)
    # Match common.filter_observation's legacy behavior, including forced menus.
    if observation.get("phase") == "combat" or len(observation["legal_actions"]) < 2:
        return observation
    kept = [a for a in observation["legal_actions"] if not (
        a.get("description") == "take key" or a.get("description", "").startswith("recall"))]
    return ({**observation, "legal_actions": kept} if kept else observation)


class StudyPolicy:
    """Legacy champion plus entity or legacy candidate phase replacements, CPU only."""
    def __init__(self, champion: str, combat: str | None = None, ooc: str | None = None):
        import torch
        from sts_rl.common import load_policy
        torch.set_num_threads(1)
        champion_identity = file_identity(champion)
        self.champion, _ = load_policy(champion)
        if file_identity(champion) != champion_identity:
            raise RuntimeError("Champion changed during load")
        self._validate_finite(self.champion)
        self.champion.to("cpu").eval()
        self.replacements = {}
        self._lock = threading.RLock()
        self.identity = {"protocol": PROTOCOL, "filter_ids": [FILTER_ID, DEAD_FILTER_ID],
                         "public_context_id": PUBLIC_CONTEXT_ID,
                         "inference": INFERENCE, "champion": champion_identity,
                         "combat": None, "ooc": None}
        for phase, path in (("combat", combat), ("ooc", ooc)):
            if path is None:
                continue
            identity = file_identity(path)
            payload = torch.load(path, map_location='cpu', weights_only=False)
            if file_identity(path) != identity:
                raise RuntimeError("Replacement changed during load")
            if not isinstance(payload, dict) or not isinstance(payload.get('metadata'), dict):
                raise ValueError('Replacement requires checkpoint metadata')
            metadata = payload['metadata']
            if metadata.get('kind') == 'entity_fresh_v1':
                config = metadata.get('model_config')
                if not isinstance(config, Mapping) or config.get('phase') not in ('combat', 'ooc'):
                    raise ValueError('Replacement requires an explicit entity phase')
                from sts_policy.entity_model import policy_from_payload
                model = policy_from_payload(payload)
                actual_phase = model.config.phase
                if metadata.get('phase', actual_phase) != actual_phase:
                    raise ValueError('Replacement phase mismatch')
            elif (metadata.get('kind') in (None, 'candidate') and
                  metadata.get('encoder_version') == 'public_candidate_v2'):
                from sts_policy.model import CandidateActorCritic
                from sts_policy.model_agent import policy_from_payload
                model = policy_from_payload(payload)
                if not isinstance(model, CandidateActorCritic):
                    raise ValueError('Replacement must be a single candidate phase head')
                actual_phase = metadata.get('phase')
            else:
                raise ValueError('Unknown replacement kind/encoder_version')
            if actual_phase != phase:
                raise ValueError('Replacement phase mismatch')
            self._validate_finite(model)
            self.replacements[phase] = model.to("cpu").eval()
            self.identity[phase] = identity

    @staticmethod
    def _validate_finite(model):
        import torch
        for name, value in model.state_dict().items():
            if not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite policy weight: {name}')

    def _logits(self, observation):
        import torch
        phase = "combat" if observation["phase"] == "combat" else "ooc"
        model = self.replacements.get(phase, self.champion)
        entity = False
        if model is not self.champion:
            from sts_policy.entity_model import EntityActorCritic
            entity = isinstance(model, EntityActorCritic)
        if entity:
            from sts_policy.entity_encoding import encode_observation, collate_observations
            batch = collate_observations([encode_observation(observation)], device="cpu")
        else:
            from sts_policy.model_agent import collate_public
            batch = collate_public([observation], "cpu")
        with self._lock, torch.inference_mode():
            forward = model.actor_logits
            result = forward(batch) if entity else forward(*batch)
            logits = result[0] if isinstance(result, tuple) else result
            if logits.shape != (1, len(observation["legal_actions"])):
                raise ValueError("Single-observation logits/menu shape mismatch")
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite policy logits")
            return logits[0].detach().clone()

    def distribution(self, observation):
        """Return original display indices and softmax probabilities (for drift)."""
        observation = policy_observation(observation)
        indices = [int(a["index"]) for a in observation["legal_actions"]]
        if len(indices) == 1:
            return indices, [1.0]
        return indices, self._logits(observation).softmax(-1).tolist()

    def choose_action(self, observation):
        observation = policy_observation(observation)
        legal = observation["legal_actions"]
        if not legal:
            raise ValueError("Policy requires a legal decision")
        # Argmax is on logits, never rounded/underflowed probabilities. First tie wins.
        position = 0 if len(legal) == 1 else int(self._logits(observation).argmax())
        return int(legal[position]["index"])


def validate_world_range(start, n):
    if (isinstance(start, bool) or not isinstance(start, int) or isinstance(n, bool) or
            not isinstance(n, int) or n < 1 or start < 0 or start + n > 2**63):
        raise ValueError("Invalid contiguous world range")
    # Whole reserved namespace is embargoed, not just the 4,096-world cohort.
    if start < 92_000_000_000 and start + n > 91_200_000_000:
        raise ValueError("Confirmation namespace is embargoed")


def atomic_json(path, value, *, compressed=False):
    import gzip
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        with temporary.open("xb") as binary:
            if compressed:
                with gzip.GzipFile(fileobj=binary, mode="wb", compresslevel=1, mtime=0) as output:
                    output.write((json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())
            else:
                binary.write((json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n").encode())
            binary.flush()
            os.fsync(binary.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_run(out, config, policy, *, inputs=()):
    """Fresh-only outputs. A completed world shard is immutable; no partial resume."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Output must be empty; resume is not supported")
    # Claim the empty directory atomically if two launches accidentally race.
    with (out / ".fresh-run-claim").open("x") as claim:
        claim.write(f"{PROTOCOL} pid={os.getpid()}\n")
    for key in ("champion", "combat", "ooc"):
        identity = policy.identity.get(key)
        if identity and file_identity(identity["path"]) != identity:
            raise RuntimeError(f"{key} changed since policy load")
    checkpoints = [policy.identity[k]["path"] for k in ("combat", "ooc") if policy.identity.get(k)]
    launch = begin_run(out, config, checkpoint=policy.identity["champion"]["path"],
                       inputs=[*inputs, *checkpoints])
    receipt = {"protocol": PROTOCOL, "policy": policy.identity, "config": config, "launch": launch}
    atomic_json(out / "receipt.json", receipt)
    return receipt


def on_ac():
    if sys.platform != "darwin":
        return True
    result = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True,
                            check=True, timeout=5)
    if "AC Power" in result.stdout:
        return True
    if "Battery Power" in result.stdout:
        return False
    raise RuntimeError("Cannot determine AC status")


class Interrupted(Exception):
    """A control boundary interrupted an unfinished world/grid item."""


class RunControl:
    """Check each boundary; cache power for two seconds and pause between operations.

    Active time counts only intervals bounded by real AC polls. This deliberately
    omits uncertain transition intervals, including the interval detecting battery.
    """
    POWER_POLL_SECONDS = 2.0

    def __init__(self, out, deadline_utc=None, *, ac=on_ac, now=time.time,
                 monotonic=time.monotonic, sleep=time.sleep, disk=shutil.disk_usage):
        self.out, self.ac, self.now = Path(out), ac, now
        self.monotonic, self.sleep, self.disk = monotonic, sleep, disk
        self.deadline = None
        if deadline_utc:
            stamp = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
            if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
                raise ValueError("deadline-utc requires explicit UTC timezone")
            self.deadline = stamp.timestamp()
        self._lock = threading.RLock()
        self._closed, self._caffeinate = False, None
        self.active_seconds, self.last_tick = 0.0, monotonic()
        self.was_ac = ac()
        self._sync_awake()

    def _release_awake(self):
        process, self._caffeinate = self._caffeinate, None
        if process is not None and process.poll() is None:
            # Only the subprocess created by this RunControl is ever signaled.
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)

    def _sync_awake(self):
        if self._closed or not self.was_ac:
            self._release_awake()
        elif sys.platform == "darwin" and (self._caffeinate is None or self._caffeinate.poll() is not None):
            # -s itself applies only on AC. -w also ends the assertion if we exit.
            self._caffeinate = subprocess.Popen(["caffeinate", "-s", "-w", str(os.getpid())],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _refresh_power(self, current, *, force=False):
        if not force and current - self.last_tick < self.POWER_POLL_SECONDS:
            return
        powered = self.ac()
        if self.was_ac and powered:
            self.active_seconds += max(0.0, current - self.last_tick)
        self.last_tick, self.was_ac = current, powered
        self._sync_awake()

    def stats(self):
        with self._lock:
            if not self._closed:
                self._refresh_power(self.monotonic())
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return {"active_ac_seconds": self.active_seconds, "on_ac": self.was_ac,
                    "parent_peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
                    "free_disk_bytes": self.disk(self.out).free}

    def check(self):
        while True:
            if (self.out / "STOP").exists():
                raise Interrupted("STOP")
            if self.deadline is not None and self.now() >= self.deadline:
                raise Interrupted("deadline")
            status = self.stats()
            if status["free_disk_bytes"] < MIN_DISK:
                raise Interrupted("disk_below_6_GiB")
            if status["on_ac"]:
                return
            self.sleep(1)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._refresh_power(self.monotonic(), force=True)
            finally:
                self._release_awake()


class Progress:
    def __init__(self, out, control):
        self.path, self.control = Path(out) / "progress.jsonl", control
        self.lock = threading.RLock()
        self.counts = {k: 0 for k in ("completed", "decisions", "faults", "censored")}

    def emit(self, event="heartbeat", **fields):
        with self.lock:
            stats = self.control.stats()
            seconds = max(stats["active_ac_seconds"], 1e-9)
            row = {"event": event, "utc": datetime.now(timezone.utc).isoformat(), **self.counts,
                   "games_per_s": self.counts["completed"] / seconds,
                   "decisions_per_s": self.counts["decisions"] / seconds, **stats, **fields}
            with self.path.open("a") as output:
                output.write(json.dumps(row, allow_nan=False) + "\n")

    def add(self, **counts):
        with self.lock:
            for key, value in counts.items():
                self.counts[key] += value


def finish_run(out, summary, source, progress):
    """Publish final status/receipts for complete, interrupted and failed runs alike."""
    out = Path(out)
    summary.setdefault("cohort_complete", summary.get("grid_complete", False))
    failed = bool(summary.get("programming_error") or summary.get("grid_validation_error") or summary.get("faults"))
    summary["event"] = "done"
    summary["status"] = ("failed" if failed else "completed" if summary["cohort_complete"] else
                         "interrupted" if summary.get("stop_reasons") else "incomplete")
    if summary["status"] != "completed":
        summary["confidence_claims_allowed"] = False
    summary["run_control"] = progress.control.stats()
    atomic_json(out / "summary.json", summary)
    fields = {k: summary[k] for k in ("status", "cohort_complete", "confidence_claims_allowed",
              "faults", "cuts", "censored", "unstarted", "stop_reasons")}
    fields.update({k: summary[k] for k in ("completed", "grid_complete", "valid") if k in summary})
    atomic_json(out / "completion.json", {"protocol": PROTOCOL, "event": "done", **fields,
                "source": source, "summary": file_identity(out / "summary.json"),
                "run_control": summary["run_control"]})
    progress.emit("done", **fields)
