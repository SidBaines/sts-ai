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

from .in_process import InProcessWorker

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
                 slim: bool = True, in_process: bool = False, forced_play: bool = False, dominated: bool = False,
                 smith_above: float | None = None, prefer_remove: bool = False,
                 elite_above: float | None = None, skip_cards_after: int | None = None,
                 save_potions_above: float | None = None, avoid_boss_relics: tuple[str, ...] = (),
                 no_skip_rares: bool = False, take_cards: tuple[str, ...] = (),
                 hide_ooc_potions: bool = False) -> list[Any]:
    workers = [InProcessWorker(config=WORKER_CONFIG) if in_process else
               CurriculumWorker(config=WORKER_CONFIG, timeout=timeout) for _ in range(count)]
    if not (dead_action_filter or slim or forced_play or dominated or smith_above is not None or prefer_remove
            or elite_above is not None or skip_cards_after is not None or save_potions_above is not None
            or avoid_boss_relics or no_skip_rares or take_cards or hide_ooc_potions):
        return workers
    return [FilteringWorker(w, dead_actions=dead_action_filter, slim=slim, forced_play=forced_play,
                            dominated=dominated, smith_above=smith_above, prefer_remove=prefer_remove,
                            elite_above=elite_above, skip_cards_after=skip_cards_after,
                            save_potions_above=save_potions_above, avoid_boss_relics=tuple(avoid_boss_relics),
                            no_skip_rares=no_skip_rares, take_cards=tuple(take_cards),
                            hide_ooc_potions=hide_ooc_potions)
            for w in workers]


def close_workers(workers: Sequence[Any]) -> None:
    for worker in workers:
        worker.close()


def load_policy(path: str | Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = load_checkpoint(path)
    return policy_from_payload(payload), payload


def save_policy(path: str | Path, model: torch.nn.Module, template: dict[str, Any], *,
                metadata: dict[str, Any], optimizer: torch.optim.Optimizer | None = None,
                step: int = 0, rng: Any = None, trainer_state: dict[str, Any] | None = None) -> None:
    """Full loadable checkpoint (same format as sts_policy), atomic replace."""
    payload = dict(template)
    payload["model_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    payload["optimizer_state"] = optimizer.state_dict() if optimizer is not None else None
    payload["step"] = step
    if trainer_state is not None:
        payload["trainer_state"] = trainer_state
    from .recording import capture_rng, file_identity
    inherited = template.get('metadata', {})
    payload['metadata'] = {'encoder_version': inherited.get('encoder_version', 'public_candidate_v2'),
                           'historical_parent_metadata': inherited, **metadata}
    parent = metadata.get('config', {}).get('checkpoint')
    if parent:
        payload['metadata']['parent_policy_checkpoint'] = file_identity(parent)
    payload['rng_state'] = capture_rng(rng)
    if 'iteration' in metadata:
        payload['iteration'] = metadata['iteration']
        payload['steps_total'] = step
    payload['python_rng'] = payload['rng_state']['python']
    payload['torch_rng'] = payload['rng_state']['torch']
    payload['mps_rng'] = payload['rng_state']['mps']
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    with temporary.open('wb') as output:
        torch.save(payload, output)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, destination)


def save_atomic(obj: Any, path: str | Path) -> None:
    """torch.save via a temporary file and os.replace: a kill mid-write (queue
    cancel, stall watchdog) can never leave a truncated resume checkpoint."""
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    with temporary.open('wb') as output:
        torch.save(obj, output)
        output.flush()
        os.fsync(output.fileno())
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


def save_potions(observation: dict[str, Any], below: float) -> dict[str, Any]:
    """In hallway (MONSTER) fights, hide potion drinking unless HP fraction < ``below``.

    Act 2 boss fights entered with 0-1 potions are won ~23% of the time vs 39-45%
    with 2-3 (correlational), and 36% start with none, so test saving potions for
    elites and bosses."""
    if observation.get("phase") != "combat":
        return observation
    state = observation.get("state") or {}
    if not str(state.get("room", "")).endswith("MONSTER"):
        return observation
    fraction = hp_fraction(observation)
    if fraction is None or fraction < below:
        return observation
    legal = observation.get("legal_actions") or []
    kept = [a for a in legal if not str(a.get("description", "")).startswith("drink potion")]
    return observation if not kept or len(kept) == len(legal) else {**observation, "legal_actions": kept}


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


# --- Dominated actions (out of combat) --------------------------------------------
# Menus at update 28 showed the policy choosing "rest" at full HP (heals 0) over
# "smith" (upgrades a card): strictly dominated unless Dream Catcher turns a rest
# into a card reward. PPO rarely samples smith (~1%), so it never learns this.

def drop_dominated(observation: dict[str, Any]) -> dict[str, Any]:
    if observation.get("phase") == "combat":
        return observation
    legal = observation.get("legal_actions") or []
    descriptions = [str(a.get("description", "")) for a in legal]
    if not (any(d.startswith("rest") for d in descriptions) and any(d.startswith("smith") for d in descriptions)):
        return observation
    state = observation.get("state") or {}
    try:
        full = int(state.get("cur_hp")) >= int(state.get("max_hp"))
    except (TypeError, ValueError):
        return observation
    if not full or "Dream Catcher" in str(observation.get("text", "")):
        return observation
    return {**observation, "legal_actions": [a for a, d in zip(legal, descriptions) if not d.startswith("rest")]}


# --- Strategy overlays (experiments, off by default) ------------------------------
# The greedy update-28 policy never smiths (0/557 rest sites), never paths into an
# elite (0/211) and never buys a card removal (0/571 affordable shop menus; 77% of
# purchases are potions). These menu restrictions measure what each habit costs.

def hp_fraction(observation: dict[str, Any]) -> float | None:
    state = observation.get("state") or {}
    try:
        return int(state.get("cur_hp")) / max(1, int(state.get("max_hp")))
    except (TypeError, ValueError):
        return None


_CARD_OFFER = re.compile(r"^take card (.+?) \[(\w+), (\w+)\]")


def apply_overlays(observation: dict[str, Any], *, smith_above: float | None = None,
                   prefer_remove: bool = False, elite_above: float | None = None,
                   skip_cards_after: int | None = None, avoid_boss_relics: tuple[str, ...] = (),
                   no_skip_rares: bool = False, take_cards: tuple[str, ...] = ()) -> dict[str, Any]:
    if observation.get("phase") == "combat":
        return observation
    legal = observation.get("legal_actions") or []
    descriptions = [str(a.get("description", "")) for a in legal]
    drop: set[int] = set()
    offers = [(i, m) for i, m in ((i, _CARD_OFFER.match(d)) for i, d in enumerate(descriptions)) if m]
    if take_cards and offers:
        # Force a listed card (upgraded copies included) whenever a reward offers it.
        wanted = {name.rstrip("+") for name in take_cards}
        keep = {i for i, m in offers if m[1].rstrip("+") in wanted}
        if keep:
            drop |= {i for i in range(len(legal)) if i not in keep}
    if no_skip_rares and offers and all(m[3] == "Rare" for _, m in offers):
        # A boss card reward: take one of the rares rather than leaving without a card.
        drop |= {i for i, d in enumerate(descriptions) if d in ("skip card reward", "skip rewards / proceed")}
    if smith_above is not None and any(d.startswith("smith") for d in descriptions):
        fraction = hp_fraction(observation)
        if fraction is not None and fraction >= smith_above:
            drop |= {i for i, d in enumerate(descriptions) if d.startswith("rest")}
    if prefer_remove and any(d.startswith("buy card remove") for d in descriptions):
        drop |= {i for i, d in enumerate(descriptions) if d.startswith("buy potion")}
    if skip_cards_after is not None and any(d.startswith("skip card") for d in descriptions):
        try:
            late = int((observation.get("state") or {}).get("floor")) > skip_cards_after
        except (TypeError, ValueError):
            late = False
        if late:
            drop |= {i for i, d in enumerate(descriptions) if d.startswith("take card")}
    if avoid_boss_relics and any(d.startswith("take boss relic") for d in descriptions):
        drop |= {i for i, d in enumerate(descriptions)
                 if d.startswith("take boss relic") and any(name in d for name in avoid_boss_relics)}
    if elite_above is not None and any("room=ELITE" in d for d in descriptions):
        fraction = hp_fraction(observation)
        threshold = elite_above
        if isinstance(elite_above, dict):  # per-act thresholds, e.g. {1: 0.7, 2: 0.9, 3: 0.9}
            try:
                threshold = elite_above.get(int((observation.get("state") or {}).get("act") or 1))
            except (TypeError, ValueError):
                threshold = None
        if fraction is not None and threshold is not None and fraction >= threshold:
            drop |= {i for i, d in enumerate(descriptions) if "map node" in d and "room=ELITE" not in d}
    if not drop or len(drop) == len(legal):
        return observation
    return {**observation, "legal_actions": [a for i, a in enumerate(legal) if i not in drop]}


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
    """Wraps either game worker so observations never offer dead actions.

    slim=True additionally drops observation text the encoder never reads
    (verified to leave public_candidate_v2 encodings byte-identical)."""

    def __init__(self, worker: CurriculumWorker | InProcessWorker, *, dead_actions: bool = True, slim: bool = False,
                 forced_play: bool = False, dominated: bool = False, smith_above: float | None = None,
                 prefer_remove: bool = False, elite_above: float | None = None,
                 skip_cards_after: int | None = None, save_potions_above: float | None = None,
                 avoid_boss_relics: tuple[str, ...] = (), no_skip_rares: bool = False,
                 take_cards: tuple[str, ...] = (), hide_ooc_potions: bool = False) -> None:
        self.worker, self.dead_actions, self.slim = worker, dead_actions, slim
        self.forced_play, self.dominated = forced_play, dominated
        self.smith_above, self.prefer_remove, self.elite_above = smith_above, prefer_remove, elite_above
        self.skip_cards_after = skip_cards_after
        self.save_potions_above = save_potions_above
        self.avoid_boss_relics = tuple(avoid_boss_relics)
        self.no_skip_rares, self.take_cards = no_skip_rares, tuple(take_cards)
        self.hide_ooc_potions = hide_ooc_potions

    @property
    def last_fault_path(self):
        return getattr(self.worker, 'last_fault_path', None)

    def _map(self, observation: dict[str, Any]) -> dict[str, Any]:
        # Menu filters run before slimming: they read structured state and relic text.
        if self.hide_ooc_potions and observation.get("legal_actions"):
            # The deployment protocol (fresh_study.StudyPolicy): recover the visible
            # potion inventory from the raw menu, then hide out-of-combat drink and
            # discard choices. With them offered, the champion wastes its potions.
            from .fresh_study import policy_observation
            observation = policy_observation(observation)
        if self.forced_play:
            observation = force_play(observation)
        if self.dominated:
            observation = drop_dominated(observation)
        if self.save_potions_above is not None:
            observation = save_potions(observation, self.save_potions_above)
        if (self.smith_above is not None or self.prefer_remove or self.elite_above is not None
                or self.skip_cards_after is not None or self.avoid_boss_relics or self.no_skip_rares
                or self.take_cards):
            observation = apply_overlays(observation, smith_above=self.smith_above, prefer_remove=self.prefer_remove,
                                         elite_above=self.elite_above, skip_cards_after=self.skip_cards_after,
                                         avoid_boss_relics=self.avoid_boss_relics,
                                         no_skip_rares=self.no_skip_rares, take_cards=self.take_cards)
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

# public_candidate_v2 action-feature columns (encoding.ACTION_FLAGS order).
_FLAG_SKIP_CARD, _FLAG_BUY, _FLAG_SMITH, _FLAG_MAP_NODE = 7, 11, 14, 19


def strategic_rows(actions: torch.Tensor, mask: torch.Tensor, *, cards: bool = False) -> torch.Tensor:
    """Rows whose menu is a rest site (smith offered), a map choice or a shop
    (optionally a card reward), read from the encoder's action flags."""
    flags = actions * mask[..., None].to(actions.dtype)
    rows = (flags[..., _FLAG_SMITH] > .5) | (flags[..., _FLAG_MAP_NODE] > .5) | (flags[..., _FLAG_BUY] > .5)
    if cards:
        rows = rows | (flags[..., _FLAG_SKIP_CARD] > .5)
    return rows.any(-1)


def phase_forward(*, combat: str = "sample", ooc: str = "sample", temperature: float = 1.0,
                  strategic_temperature: float = 1.0, strategic_cards: bool = False,
                  card_temperature: float = 1.0):
    """Forward that makes one or both phases greedy (logit scale 1e4).

    Scaling preserves masking and keeps log-probs finite, so the same function
    can be used for collection and for PPO ratios.

    strategic_temperature divides OOC logits on rest-site, map and shop menus
    (and card rewards if strategic_cards). At update 28 those logits were
    saturated (median gap of the unchosen strategic option to the best ~ -18 to
    -36), so T=1 sampling never explored smith, elites or removal. Argmax, and so
    deployment, is unchanged by any temperature.
    """
    def forward(model: torch.nn.Module, batch: Any):
        logits, values, mask = forward_policy(model, batch)
        is_combat = batch[0][:, 7] > .5
        scale = torch.full_like(values, 1.0 / temperature)
        if strategic_temperature != 1.0:
            strategic = strategic_rows(batch[1], mask, cards=strategic_cards) & ~is_combat
            scale = torch.where(strategic, torch.full_like(values, 1.0 / strategic_temperature), scale)
        if card_temperature != 1.0:
            cards=((batch[1][...,_FLAG_SKIP_CARD]>.5)&mask).any(-1)&~is_combat
            scale=torch.where(cards,torch.full_like(values,1.0/card_temperature),scale)
        if combat == "greedy":
            scale = torch.where(is_combat, torch.full_like(values, 1e4), scale)
        if ooc == "greedy":
            scale = torch.where(~is_combat, torch.full_like(values, 1e4), scale)
        logits=(logits * scale[:, None]).masked_fill(~mask,-1e9)
        if combat=='argmax' or ooc=='argmax':
            # Exact tie-breaking for a frozen phase, with a one-point categorical
            # distribution. Historical 'greedy' scaling remains readable.
            rows=(is_combat if combat=='argmax' else torch.zeros_like(is_combat)) | \
                 (~is_combat if ooc=='argmax' else torch.zeros_like(is_combat))
            chosen=logits.argmax(-1)
            fixed=torch.full_like(logits,-1e9)
            fixed.scatter_(1,chosen[:,None],0.)
            logits=torch.where(rows[:,None],fixed,logits)
        return logits, values, mask
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
