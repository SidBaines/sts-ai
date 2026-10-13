"""Shared worker, task, checkpoint and menu helpers for sts_rl experiments."""
from __future__ import annotations

from concurrent.futures import Future
import json
import os
import re
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import torch

from sts_policy.curriculum import TaskSpec, world_split
from sts_policy.model_agent import forward_policy, policy_from_payload
from sts_policy.training import load_checkpoint
from sts_policy.workers import CurriculumWorker

ORIGINAL_CHECKPOINT = ("/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/"
                       "data/nonllm_native_revision/campaign/joint_act3_0030.pt")
WORKER_CONFIG = {"gamma": 1.0, "reward_mode": "progress", "terminal_hp_weight": 0.0,
                 "progress_floor_weight": 0.05, "goal_act_clear": True,
                 "fast_observations": True, "fast_replay": True}
# Seed namespaces (all TRAIN split): evaluation and training never overlap, and
# none overlap the codex campaigns (1e6, 1e8, 1e11 ranges) or DEV/final cohorts.
EVAL_WORLD_START = 5_000_000_000
TRAIN_WORLD_START = 7_000_000_000
WIN_OUTCOMES = ("act_clear", "victory")
BOSS_FLOORS = (16, 33, 50)


def train_tasks(start: int, count: int, *, max_act: int = 3, horizon: int = 4096) -> list[TaskSpec]:
    """First ``count`` TRAIN-split fresh full-run tasks at or after ``start``."""
    tasks, seed = [], start
    while len(tasks) < count:
        if world_split(seed) == "train":
            tasks.append(TaskSpec(seed=seed, kind="run", seek_combat=False, max_act=max_act,
                                  horizon=horizon, battle_simulations=200))
        seed += 1
    return tasks


def make_workers(count: int, *, timeout: float = 10.0, dead_action_filter: bool = False,
                 slim: bool = True, forced_play: bool = False) -> list[Any]:
    workers = [CurriculumWorker(config=WORKER_CONFIG, timeout=timeout) for _ in range(count)]
    if not (dead_action_filter or slim or forced_play):
        return workers
    return [FilteringWorker(w, dead_actions=dead_action_filter, slim=slim, forced_play=forced_play)
            for w in workers]


def close_workers(workers: Sequence[Any]) -> None:
    for worker in workers:
        worker.close()


def load_policy(path: str | Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = load_checkpoint(path)
    return policy_from_payload(payload), payload


def save_policy(path: str | Path, model: torch.nn.Module, template: dict[str, Any], *,
                metadata: dict[str, Any], optimizer: torch.optim.Optimizer | None = None,
                step: int = 0) -> None:
    """Full loadable checkpoint (same format as sts_policy), atomic replace."""
    payload = dict(template)
    payload["model_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    payload["optimizer_state"] = optimizer.state_dict() if optimizer is not None else None
    payload["step"] = step
    payload["metadata"] = {**template.get("metadata", {}), **metadata}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def save_atomic(obj: Any, path: str | Path) -> None:
    """torch.save via a temporary file and os.replace: a kill mid-write (queue
    cancel, stall watchdog) can never leave a truncated resume checkpoint."""
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    torch.save(obj, temporary)
    os.replace(temporary, destination)


# --- Dead-action removal --------------------------------------------------------
# Act 4 keys cannot help when max_act == 3 (the run ends at the Act 3 boss), and
# PPO drifted into them. Removing them from the menu the policy sees is purely a
# policy-side filter: native indices are preserved in each legal action entry.

def is_dead_action(description: str) -> bool:
    return description == "take key" or description.startswith("recall")


# --- Forced play (combat) ---------------------------------------------------------
# Audit 2026-10-13: in Act 2 boss fights the frozen combat head ended ~11% of turns
# with unspent energy while an affordable card was playable (~1 per fight). The
# filter hides "end turn" while energy remains and some *safe* card can be played.
# Self-damaging / deck-polluting cards and curses/statuses (playable with Blue
# Candle at an HP cost) do not count as safe, so they never force the turn on.
UNSAFE_PLAY = re.compile(r"\[(?:Curse|Status)\]|\b(?:Hemokinesis|Offering|Bloodletting|Wild Strike|Reckless Charge|"
                         r"Immolate|Burning Pact|True Grit|Fiend Fire|Sever Soul|J\.A\.X\.|Second Wind|"
                         r"Combust|Brutality|Berserk|Rupture|Panacea|Purity|Violence|Chrysalis|Metamorphosis)\b")


def _energy(observation: dict[str, Any]) -> int:
    state = observation.get("state") or {}
    cards = state.get("cards") or {}
    combat = state.get("combat") or {}
    value = cards.get("energy", combat.get("player_energy"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def force_play(observation: dict[str, Any]) -> dict[str, Any]:
    """Drop "end turn" from a combat menu while energy remains and a safe card is playable."""
    if observation.get("phase") != "combat":
        return observation
    legal = observation.get("legal_actions") or []
    if not any(str(a.get("description")) == "end turn" for a in legal) or _energy(observation) <= 0:
        return observation
    safe = [a for a in legal if str(a.get("description", "")).startswith("play ")
            and not UNSAFE_PLAY.search(str(a.get("description", "")))]
    if not safe:
        return observation
    return {**observation, "legal_actions": [a for a in legal if str(a.get("description")) != "end turn"]}


_ENCODER_TEXT = re.compile(r"^(?:(?:Deck|Relics|Potions|Card select|Selection type|Event):.*|Choose a card.*|Act .*end-of-act boss.*)$")


def slim_text(text: str) -> str:
    """Keep only the text lines public_candidate_v2 reads (see encoding.py: _PUBLIC_LINES,
    the 'Choose a card' purpose and the act-boss line). Glossary/route prose is dropped:
    it is ~half of each observation and the dominant memory cost of large batches."""
    return "\n".join(line for line in text.split("\n") if _ENCODER_TEXT.match(line))


def filter_observation(observation: dict[str, Any], *, slim: bool = False) -> dict[str, Any]:
    if slim and "text" in observation:
        observation = {**observation, "text": slim_text(str(observation["text"]))}
    legal = observation.get("legal_actions") or []
    if observation.get("phase") == "combat" or len(legal) < 2:
        return observation
    kept = [a for a in legal if not is_dead_action(str(a.get("description", "")))]
    if not kept or len(kept) == len(legal):
        return observation
    return {**observation, "legal_actions": kept}


class _MappedFuture:
    def __init__(self, future: Future, mapper: Callable[[Any], Any]) -> None:
        self._future, self._mapper = future, mapper

    def result(self, timeout: float | None = None) -> Any:
        return self._mapper(self._future.result(timeout))


class FilteringWorker:
    """Wraps a CurriculumWorker so observations never offer dead actions.

    slim=True additionally drops observation text the encoder never reads
    (verified to leave public_candidate_v2 encodings byte-identical)."""

    def __init__(self, worker: CurriculumWorker, *, dead_actions: bool = True, slim: bool = False,
                 forced_play: bool = False) -> None:
        self.worker, self.dead_actions, self.slim, self.forced_play = worker, dead_actions, slim, forced_play

    def _map(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.forced_play:
            # Before slimming: the energy is read from the structured state.
            observation = force_play(observation)
        if self.dead_actions:
            return filter_observation(observation, slim=self.slim)
        return {**observation, "text": slim_text(str(observation.get("text", "")))} if self.slim else observation

    def submit(self, method: str, *args: Any) -> Any:
        future = self.worker.submit(method, *args)
        if method == "reset":
            return _MappedFuture(future, lambda r: (self._map(r[0]), r[1]))
        if method == "step":
            return _MappedFuture(future, lambda r: (self._map(r[0]), *r[1:]))
        return future

    def close(self) -> None:
        self.worker.close()


# --- Forward wrappers ------------------------------------------------------------

def phase_forward(*, combat: str = "sample", ooc: str = "sample", temperature: float = 1.0):
    """Forward that makes one or both phases greedy (logit scale 1e4).

    Scaling preserves masking and keeps log-probs finite, so the same function
    can be used for collection and for PPO ratios.
    """
    def forward(model: torch.nn.Module, batch: Any):
        logits, values, mask = forward_policy(model, batch)
        is_combat = batch[0][:, 7] > .5
        scale = torch.full_like(values, 1.0 / temperature)
        if combat == "greedy":
            scale = torch.where(is_combat, torch.full_like(values, 1e4), scale)
        if ooc == "greedy":
            scale = torch.where(~is_combat, torch.full_like(values, 1e4), scale)
        return (logits * scale[:, None]).masked_fill(~mask, -1e9), values, mask
    return forward


class JsonlLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: dict[str, Any]) -> None:
        record = {"t": time.time(), **record}
        with self.path.open("a") as output:
            output.write(json.dumps(record) + "\n")
        print(json.dumps(record)[:600], flush=True)
