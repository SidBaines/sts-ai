"""Fight-panel deck-strength evaluation (requires the sts_rl_panel native build).

Given a decision state (world seed + action prefix), replay it into a scratch
episode, salt the combat seed, start a battle against a chosen encounter and
play it with a frozen combat policy inside the worker process. Repeating this
over a small panel (act boss, upcoming elites/hallways) and a few seeds gives a
low-variance estimate of how strong the current deck/relics/potions/HP are.

Privileged by construction (uses the run's pregenerated encounter lists and
replays hidden state); use it as a teacher/diagnostic, never as a policy input.
"""
from __future__ import annotations

from concurrent.futures import Future
import multiprocessing
import os
import traceback
from typing import Any, Sequence

from sts_policy.curriculum import TaskSpec
from sts_policy.workers import CurriculumEpisode, CurriculumWorker

_POLICY_CACHE: dict[str, Any] = {}


def _policy(path: str):
    if path not in _POLICY_CACHE:
        import torch
        from sts_policy.model_agent import policy_from_payload
        from sts_policy.training import load_checkpoint
        torch.set_num_threads(1)
        _POLICY_CACHE.clear()
        _POLICY_CACHE[path] = policy_from_payload(load_checkpoint(path)).eval()
    return _POLICY_CACHE[path]


def _choose(model: Any, observation: dict[str, Any], sample: bool, rng: Any) -> int:
    import torch
    from sts_policy.losses import masked_distribution
    from sts_policy.model_agent import collate_public, forward_policy
    with torch.no_grad():
        logits, _, mask = forward_policy(model, collate_public([observation], "cpu"))
    if sample:
        probs = masked_distribution(logits, mask).probs[0]
        position = int(torch.multinomial(probs, 1, generator=rng))
    else:
        position = int(logits[0].argmax())
    return int(observation["legal_actions"][position]["index"])


class PanelEpisode(CurriculumEpisode):
    def encounters(self, spec: TaskSpec | dict[str, Any]) -> dict[str, Any]:
        """Upcoming encounters at a decision state (names), for panel construction."""
        self.reset(spec)
        gc = self.episode.env._env.gc
        return {"boss": str(gc.boss).rsplit(".", 1)[-1],
                "elites": [str(e).rsplit(".", 1)[-1] for e in gc.upcoming_elite_encounters()],
                "hallway": [str(e).rsplit(".", 1)[-1] for e in gc.upcoming_monster_encounters()],
                "act": int(gc.act), "floor": int(gc.floor_num), "hp": int(gc.cur_hp), "max_hp": int(gc.max_hp)}

    def search_votes(self, simulations: int, draw_order_seeds: Sequence[int],
                     search_seed: int = 0) -> dict[int, int]:
        """Native search at the live combat decision, once per shuffled hidden draw
        order; returns {display index: votes}. Votes over public-consistent draw
        orders approximate a public-information choice (less privileged than one
        search on the true order). Does not mutate the live state."""
        env = self.episode.env
        raw = env._env
        votes: dict[int, int] = {}
        for d in draw_order_seeds:
            result = raw.search_best_combat_action(simulations, search_seed=search_seed + d, draw_order_seed=d)
            index = int(env.display_index_for_bits(int(result["bits"])))
            votes[index] = votes.get(index, 0) + 1
        return votes

    def search_votes_public(self, simulations: int, samples: Sequence[int], search_seed: int = 0) -> dict[int, int]:
        """Public-information search votes at the live combat decision.

        Each sample searches a deep copy of the battle whose hidden draw order is
        shuffled AND whose battle RNG streams (enemy AI, card randomness, reshuffles,
        potions, monster HP) are re-seeded, so the search cannot foresee hidden
        randomness (search_votes above keeps the true streams: privileged). Needs the
        rl build with BattleContext.copy/reseed_rngs. Does not mutate the live state."""
        from sts_env._backend.lightspeed import _TYPED_COMBAT_OBSERVATIONS
        env = self.episode.env
        raw = env._env
        raw.advance_to_decision()
        if raw.bc is None:
            raise RuntimeError("search_votes_public requires a pending combat decision")
        include = raw.combat_observation in _TYPED_COMBAT_OBSERVATIONS
        votes: dict[int, int] = {}
        for sample in samples:
            scratch = raw.bc.copy()
            scratch.reseed_rngs(int(search_seed) * 1009 + int(sample))
            try:
                result = scratch.search_best_action(simulations, search_seed=search_seed + sample,
                                                    draw_order_seed=sample, include_card_type=include)
            except ValueError as exc:
                # Draw-index card selection (Secret Technique/Weapon) cannot be searched on a
                # shuffled copy without leaking the true order: abstain, caller falls back.
                if "draw_order_seed" in str(exc):
                    return {}
                raise
            if not result.get("provenance", {}).get("hidden_pile_order_canonicalized"):
                raise RuntimeError("Canonical search requires the 0.1.4 sampling patch")
            index = int(env.display_index_for_bits(int(result["bits"])))
            votes[index] = votes.get(index, 0) + 1
        return votes

    def search_details_public(self, simulations: int, futures: list[int], search_seeds: list[int],
                              rng_base: int, options: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Canonical scratch searches with independent future and search seeds."""
        from sts_env._backend.lightspeed import _TYPED_COMBAT_OBSERVATIONS
        if len(futures) != len(search_seeds):
            raise ValueError("futures and search_seeds must have equal lengths")
        env = self.episode.env
        raw = env._env
        raw.advance_to_decision()
        if raw.bc is None:
            raise RuntimeError("search_details_public requires a pending combat decision")
        include = raw.combat_observation in _TYPED_COMBAT_OBSERVATIONS
        if options is not None and not hasattr(raw.sts, "set_search_options"):
            raise RuntimeError("Search options require a native build with set_search_options")
        try:
            if options is not None:
                raw.sts.set_search_options(**options)
            return self._search_details(raw, env, simulations, futures, search_seeds, rng_base, include)
        finally:
            if options is not None:
                raw.sts.reset_search_options()

    @staticmethod
    def _search_details(raw, env, simulations, futures, search_seeds, rng_base, include):
        details = []
        for future, search_seed in zip(futures, search_seeds):
            scratch = raw.bc.copy()
            scratch.reseed_rngs(int(rng_base) * 1009 + int(future))
            try:
                result = scratch.search_best_action(simulations, search_seed=search_seed,
                                                    draw_order_seed=future, include_card_type=include)
            except ValueError as exc:
                if "draw_order_seed" in str(exc):
                    return []
                raise
            if not result.get("provenance", {}).get("hidden_pile_order_canonicalized"):
                raise RuntimeError("Canonical search requires the 0.1.4 sampling patch")
            details.append({"future": int(future), "search_seed": int(search_seed),
                "chosen": int(env.display_index_for_bits(int(result["bits"]))),
                "selection_method": str(result["selection_method"]),
                "winning_sequence_found": bool(result["winning_sequence_found"]),
                "root_simulations": int(result["root_simulations"]),
                "best_evaluation": float(result["best_evaluation"]),
                "min_evaluation": float(result["min_evaluation"]),
                "edges": [{"index": int(env.display_index_for_bits(int(edge["bits"]))), "bits": int(edge["bits"]),
                           "visits": int(edge["visits"]), "evaluation_sum": float(edge["evaluation_sum"]),
                           "mean_evaluation": float(edge["mean_evaluation"])}
                          for edge in result["root_edges"] if edge["valid"]]})
        return details

    def menu(self, spec: TaskSpec | dict[str, Any]) -> list[dict[str, Any]]:
        observation, _ = self.reset(spec)
        self.finished = True
        return observation["legal_actions"]

    def _fights_from_current(self, fights: Sequence[tuple[str, str, int]], model: Any, sample: bool,
                             full_hp_rooms: Sequence[str], max_decisions: int) -> list[dict[str, Any]]:
        """Play each fight from an in-memory copy of the current OOC game context."""
        import torch
        raw = self.episode.env._env
        sts, base = raw.sts, raw.gc
        if raw.bc is not None:
            raise RuntimeError("panel requires an out-of-combat decision state")
        base_steps = self.episode.steps
        results = []
        original_exit = sts.BattleContext.exit_battle
        try:
            for encounter_name, room, salt in fights:
                ended: dict[str, Any] = {}

                def record_exit(bc: Any, context: Any) -> None:
                    # Never run the game's post-battle code on a scratch copy (it can
                    # call an empty native callback); record and leave BATTLE.
                    ended["won"] = bc.outcome == sts.BattleOutcome.PLAYER_VICTORY
                    ended["hp"] = int(bc.player_cur_hp)
                    context.screen_state = sts.ScreenState.MAP_SCREEN

                try:
                    gc = base.copy()
                    raw.gc, raw.bc = gc, None
                    raw._combat_recent_actions = []
                    raw._combat_history_turn = None
                    self.episode.steps, self.episode.finished = base_steps, False
                    if room in full_hp_rooms:
                        gc.cur_hp = gc.max_hp
                    hp_before = int(gc.cur_hp)
                    gc.seed = (int(gc.seed) + 1_000_003 * (salt + 1)) % (1 << 63)
                    gc.cur_room = getattr(sts.Room, room)
                    sts.BattleContext.exit_battle = record_exit
                    gc.enter_battle(getattr(sts.MonsterEncounter, encounter_name))
                    raw.advance_to_decision()
                    observation = self.episode._observe()
                    rng = torch.Generator().manual_seed(salt * 7919 + 17)
                    decisions = 0
                    while not ended and observation.get("phase") == "combat" and decisions < max_decisions:
                        action = _choose(model, observation, sample, rng)
                        observation, _, terminated, truncated, _ = self.episode.step(action)
                        decisions += 1
                        if terminated or truncated:
                            break
                    won = bool(ended.get("won", False))
                    results.append({"encounter": encounter_name, "room": room, "salt": salt, "win": won,
                                    "hp_before": hp_before, "max_hp": int(gc.max_hp),
                                    "hp_after": ended.get("hp", 0) if won else 0,
                                    "decisions": decisions, "cut": not ended,
                                    "ub": bool(raw._undefined_behavior_evoked)})
                except Exception as exc:  # report, never count as a loss
                    import traceback
                    results.append({"encounter": encounter_name, "room": room, "salt": salt,
                                    "fault": f"{type(exc).__name__}: {exc}"[:200],
                                    "trace": traceback.format_exc()[-1200:]})
                finally:
                    sts.BattleContext.exit_battle = original_exit
                    raw._undefined_behavior_evoked = False
        finally:
            raw.gc, raw.bc = base, None
            self.finished = True
        return results

    def panel(self, spec: TaskSpec | dict[str, Any], fights: Sequence[tuple[str, str, int]],
              checkpoint: str, sample: bool = False, full_hp_rooms: Sequence[str] = (),
              max_decisions: int = 600) -> list[dict[str, Any]]:
        """fights: (encounter_name, room 'MONSTER'|'ELITE'|'BOSS', salt). One result per fight."""
        model = _policy(checkpoint)
        self.reset(spec)
        return self._fights_from_current(fights, model, sample, full_hp_rooms, max_decisions)

    def afterstates(self, spec: TaskSpec | dict[str, Any], options: Sequence[Sequence[int]],
                    fights: Sequence[tuple[str, str, int]], checkpoint: str, sample: bool = False,
                    full_hp_rooms: Sequence[str] = ("BOSS",), max_decisions: int = 600) -> list[dict[str, Any]]:
        """Score each option (an action-index sequence from spec's state) on identical fights."""
        model = _policy(checkpoint)
        base = TaskSpec.from_dict(spec) if isinstance(spec, dict) else spec
        out = []
        for option in options:
            try:
                self.reset(TaskSpec(seed=base.seed, kind=base.kind, prefix=tuple(base.prefix) + tuple(option),
                                    max_act=base.max_act, horizon=base.horizon, seek_combat=False))
                if self.episode.env._env.bc is not None:
                    raise RuntimeError("option leads into combat; afterstate panel undefined")
                out.append({"option": list(option),
                            "fights": self._fights_from_current(fights, model, sample, full_hp_rooms, max_decisions)})
            except Exception as exc:
                out.append({"option": list(option), "fault": f"{type(exc).__name__}: {exc}"[:200]})
        return out


def _serve_panel(connection: Any, config: dict[str, Any]) -> None:
    episode = PanelEpisode(config)
    try:
        while True:
            method, args = connection.recv()
            if method == "close":
                break
            try:
                connection.send(("ok", getattr(episode, method)(*args)))
            except Exception as exc:
                connection.send(("error", {"type": type(exc).__name__, "message": str(exc),
                                            "traceback": traceback.format_exc()}))
                break
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class PanelWorker(CurriculumWorker):
    """CurriculumWorker with `encounters` and `panel` RPCs (long per-call timeout)."""

    def _launch(self) -> None:
        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_serve_panel, args=(child, self.config),
                                        name="sts-panel-native", daemon=True)
        self._process.start()
        child.close()

    def _ensure(self) -> None:
        with self._lock:
            if self._process is None:
                self._launch()

    def encounters(self, spec: TaskSpec) -> dict[str, Any]:
        self._ensure()
        return self._call("encounters", spec)

    def menu(self, spec: TaskSpec) -> list[dict[str, Any]]:
        self._ensure()
        return self._call("menu", spec)

    def search_votes(self, simulations: int, draw_order_seeds: Sequence[int], search_seed: int = 0) -> dict[int, int]:
        return {int(k): v for k, v in self._call("search_votes", simulations, list(draw_order_seeds), search_seed).items()}

    def search_votes_public(self, simulations: int, samples: Sequence[int], search_seed: int = 0) -> dict[int, int]:
        return {int(k): v for k, v in self._call("search_votes_public", simulations, list(samples), search_seed).items()}

    def search_details_public(self, simulations: int, futures: list[int], search_seeds: list[int],
                              rng_base: int, options: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        args = (simulations, list(futures), list(search_seeds), rng_base)
        return self._call("search_details_public", *args, *(() if options is None else (options,)))

    def panel(self, spec: TaskSpec, fights: Sequence[tuple[str, str, int]], checkpoint: str,
              sample: bool = False, full_hp_rooms: Sequence[str] = ()) -> list[dict[str, Any]]:
        self._ensure()
        return self._call("panel", spec, list(fights), checkpoint, sample, list(full_hp_rooms))

    def afterstates(self, spec: TaskSpec, options: Sequence[Sequence[int]], fights: Sequence[tuple[str, str, int]],
                    checkpoint: str, sample: bool = False, full_hp_rooms: Sequence[str] = ("BOSS",)) -> list[dict[str, Any]]:
        self._ensure()
        return self._call("afterstates", spec, [list(o) for o in options], list(fights), checkpoint, sample,
                          list(full_hp_rooms))

    def submit(self, method: str, *args: Any) -> Future[Any]:
        if method in {"panel", "encounters", "afterstates", "menu", "search_votes", "search_votes_public",
                      "search_details_public"}:
            return self._executor.submit(getattr(self, method), *args)
        return super().submit(method, *args)


def default_fights(encounters: dict[str, Any], samples: int, *, hallway: int = 2, elites: int = 2,
                   boss: bool = True) -> list[tuple[str, str, int]]:
    fights = []
    for k in range(samples):
        if boss:
            fights.append((encounters["boss"], "BOSS", k))
        fights += [(e, "ELITE", k) for e in encounters["elites"][:elites]]
        fights += [(e, "MONSTER", k) for e in encounters["hallway"][:hallway]]
    return fights


def summarize_panel(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in results if "fault" not in r and not r.get("ub")]
    def rate(room: str) -> float | None:
        rows = [r for r in valid if r["room"] == room]
        return sum(r["win"] for r in rows) / len(rows) if rows else None
    def hp_loss(room: str) -> float | None:
        rows = [r for r in valid if r["room"] == room]
        return sum(r["hp_before"] - r["hp_after"] for r in rows) / len(rows) if rows else None
    return {"boss_win": rate("BOSS"), "elite_win": rate("ELITE"), "hallway_win": rate("MONSTER"),
            "boss_hp_loss": hp_loss("BOSS"), "elite_hp_loss": hp_loss("ELITE"), "hallway_hp_loss": hp_loss("MONSTER"),
            "valid": len(valid), "faults": len(results) - len(valid)}
