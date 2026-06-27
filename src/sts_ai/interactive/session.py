"""RolloutSession: a live, interactively-driven game path, plus a registry.

A session owns one live `env` sitting at its path frontier. You sample/commit
decisions from a chosen method (`user` / `first` / `random` / `heuristic` /
`model`), edit the framing/prompt, and branch (fork a fresh env and replay the
recorded action prefix — see `replay.py`). Decisions are recorded as canonical
`DecisionRecord`s (schemas.py) and auto-saved via `SessionStore`, so the cache is
analysable by the existing tools.

Env construction and agent construction are injected (`env_factory` /
`agent_builder`) so the core is unit-testable with fakes; the real defaults lazily
import the simulator and the MLX/vLLM agents. The server (`server.py`) drives this
module; nothing here imports FastAPI.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable

from sts_ai.interactive.replay import replay_actions
from sts_ai.interactive.store import SessionStore, StoredSession
from sts_ai.interactive.templates import TemplateStore, render_template
from sts_ai.prompting import NEUTRAL_FRAME, render_action_prompt
from sts_ai.rollout import build_decision_record, build_rollout_meta, prepare_decision
from sts_ai.rollout_view import to_view
from sts_ai.rollout_view_html import render_decision_html
from sts_ai.schemas import AgentDecision, DecisionRecord, LegalAction, RolloutResult
from sts_ai.seeding import derive_policy_seed

# Methods that need no model and can be (re)built cheaply.
SCRIPTED_METHODS = ("first", "random", "heuristic")
MODEL_METHOD = "model"
USER_METHOD = "user"
ALL_METHODS = (USER_METHOD, *SCRIPTED_METHODS, MODEL_METHOD)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_env_factory(
    *, world_seed: int, ascension: int, combat_control: str, max_act: int, battle_simulations: int
):
    """Construct a real LightspeedHybridEnv (lazy import — needs the built sim)."""
    from sts_ai.lightspeed import LightspeedHybridEnv

    return LightspeedHybridEnv(
        world_seed=world_seed,
        ascension=ascension,
        combat_control=combat_control,
        max_act=max_act,
        battle_simulations=battle_simulations,
    )


def default_agent_builder(name: str, **kwargs):
    """Build a scripted or model agent (lazy import of agent_factory)."""
    from sts_ai.agent_factory import build_agent

    return build_agent(name, **kwargs)


class RolloutSession:
    def __init__(
        self,
        stored: StoredSession,
        env: Any,
        *,
        store: SessionStore,
        template_store: TemplateStore,
        env_factory: Callable[..., Any] = default_env_factory,
        agent_builder: Callable[..., Any] = default_agent_builder,
        history: list[DecisionRecord] | None = None,
    ) -> None:
        self.stored = stored
        self.env = env
        self._store = store
        self._templates = template_store
        self._env_factory = env_factory
        self._agent_builder = agent_builder
        self.history: list[DecisionRecord] = history or []
        self._agents: dict[str, Any] = {}
        self._model_build_key: tuple | None = None
        self.policy_seed = derive_policy_seed(stored.world_seed, 0)

    # --- identity / convenience ------------------------------------------
    @property
    def session_id(self) -> str:
        return self.stored.session_id

    @property
    def framing(self) -> str:
        return self.stored.framing

    @property
    def next_index(self) -> int:
        return len(self.history)

    # --- agents ----------------------------------------------------------
    def _scripted_agent(self, name: str):
        if name not in self._agents:
            agent = self._agent_builder(name)
            agent.reseed(self.policy_seed)
            self._agents[name] = agent
        return self._agents[name]

    def _model_agent(self):
        s = self.stored
        key = (s.model_backend, s.model_id, s.temperature, s.max_tokens, s.thinking)
        if self._model_build_key != key or MODEL_METHOD not in self._agents:
            kwargs = dict(
                model=s.model_id,
                temperature=s.temperature,
                max_tokens=s.max_tokens,
                thinking=s.thinking,
            )
            if s.model_id is None:
                kwargs.pop("model")  # let build_agent use its default model
            agent = self._agent_builder(s.model_backend, **kwargs)
            agent.reseed(self.policy_seed)
            self._agents[MODEL_METHOD] = agent
            self._model_build_key = key
        agent = self._agents[MODEL_METHOD]
        agent.framing = s.framing  # framing edits apply live; no rebuild needed
        return agent

    def _model_prompt_override(self, state_text: str, legal_actions: list[LegalAction]) -> str | None:
        if self.stored.use_advanced_template and self.stored.prompt_template:
            return render_template(
                self.stored.prompt_template,
                framing=self.stored.framing,
                state_text=state_text,
                legal_actions=legal_actions,
            )
        return None

    # --- views -----------------------------------------------------------
    def _prepare(self) -> tuple[str, dict[str, Any] | None]:
        return prepare_decision(self.env)

    def _frontier_record_dict(self, view: dict[str, Any]) -> dict[str, Any]:
        """A pseudo-record for the *pending* (uncommitted) frontier so the board
        renders state + legal actions with nothing chosen yet."""
        return {
            "world_seed": self.stored.world_seed,
            "decision_index": self.next_index,
            "phase": view["phase"],
            "state": view["state"],
            "state_text": view["state_text"],
            "legal_actions": view["legal_action_dicts"],
            "selected_action": {},
            "agent": {},
            "after_state": {},
        }

    def current_view(self) -> dict[str, Any]:
        """The live frontier: status, structured state, legal actions, rendered
        prompt, and board HTML for the pending decision."""
        try:
            status, view = self._prepare()
        except Exception as exc:  # noqa: BLE001 - surface sim faults to the UI, never crash
            self._mark_terminal("simulator_error", str(exc))
            return {"status": "simulator_error", "error": str(exc), "done": True,
                    "decision_index": self.next_index, **self._session_summary()}
        if status != "ok":
            self.stored.status = "terminal"
            self.stored.stopped_reason = status
            self._save()
            return {"status": status, "done": True, "decision_index": self.next_index,
                    "terminal_state": self.env.summary(), **self._session_summary()}

        legal = view["legal_actions"]
        record_dict = self._frontier_record_dict(view)
        prompt = self._render_prompt(view["state_text"], legal)
        return {
            "status": "ok",
            "done": False,
            "decision_index": self.next_index,
            "phase": view["phase"],
            "state": view["state"],
            "state_text": view["state_text"],
            "legal_actions": view["legal_action_dicts"],
            "board_html": render_decision_html(to_view(record_dict)),
            "prompt": prompt,
            **self._session_summary(),
        }

    def view_at(self, k: int) -> dict[str, Any]:
        """Render a committed decision k (read-only). k == frontier delegates to
        current_view."""
        if k >= len(self.history):
            return self.current_view()
        if k < 0:
            raise IndexError(f"decision {k} out of range")
        record = asdict(self.history[k])
        return {
            "status": "committed",
            "done": False,
            "decision_index": k,
            "phase": record["phase"],
            "state": record["state"],
            "state_text": record["state_text"],
            "legal_actions": record["legal_actions"],
            "method": self.stored.methods[k] if k < len(self.stored.methods) else "",
            "board_html": render_decision_html(to_view(record)),
            **self._session_summary(),
        }

    def _render_prompt(self, state_text: str, legal_actions: list[LegalAction]) -> str:
        """The user-message prompt the model would see for the current framing/
        template (default path == render_action_prompt, for harness parity)."""
        if self.stored.use_advanced_template and self.stored.prompt_template:
            return render_template(
                self.stored.prompt_template,
                framing=self.stored.framing,
                state_text=state_text,
                legal_actions=legal_actions,
            )
        return render_action_prompt(state_text, legal_actions, self.stored.framing)

    def preview_prompt(
        self, *, framing: str | None = None, template: str | None = None, use_advanced: bool | None = None
    ) -> dict[str, Any]:
        """Render the prompt for the current decision WITHOUT committing or
        persisting the edited framing/template (live editor preview)."""
        status, view = self._prepare()
        if status != "ok":
            return {"status": status, "prompt": ""}
        framing = self.stored.framing if framing is None else framing
        use_advanced = self.stored.use_advanced_template if use_advanced is None else use_advanced
        template = self.stored.prompt_template if template is None else template
        if use_advanced and template:
            prompt = render_template(
                template, framing=framing, state_text=view["state_text"], legal_actions=view["legal_actions"]
            )
        else:
            prompt = render_action_prompt(view["state_text"], view["legal_actions"], framing)
        return {"status": "ok", "prompt": prompt}

    def set_framing(
        self, *, framing: str | None = None, template: str | None = None, use_advanced: bool | None = None
    ) -> None:
        if framing is not None:
            self.stored.framing = framing
        if template is not None:
            self.stored.prompt_template = template
        if use_advanced is not None:
            self.stored.use_advanced_template = use_advanced
        self._save()

    def set_model_config(
        self,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        thinking: bool | None = None,
        model_id: str | None = None,
        model_backend: str | None = None,
    ) -> None:
        """Update model config; the cached model agent rebuilds lazily on the next
        model call (see `_model_agent`'s build key)."""
        if temperature is not None:
            self.stored.temperature = float(temperature)
        if max_tokens is not None:
            self.stored.max_tokens = int(max_tokens)
        if thinking is not None:
            self.stored.thinking = bool(thinking)
        if model_id is not None:
            self.stored.model_id = model_id
        if model_backend is not None:
            self.stored.model_backend = model_backend
        self._save()

    # --- sampling (no commit) -------------------------------------------
    def sample_candidates(self, method: str, *, k: int = 1, temperature: float | None = None) -> dict[str, Any]:
        status, view = self._prepare()
        if status != "ok":
            return {"status": status, "candidates": []}
        legal = view["legal_actions"]
        candidates: list[dict[str, Any]] = []
        if method == USER_METHOD:
            candidates = [{"action_index": a.index, "description": a.description} for a in legal]
        elif method in SCRIPTED_METHODS:
            agent = self._scripted_agent(method)
            draws = max(1, k) if method == "random" else 1
            for _ in range(draws):
                candidates.append(self._decision_to_candidate(agent.choose_action(view["state_text"], legal), legal))
        elif method == MODEL_METHOD:
            agent = self._model_agent()
            override = self._model_prompt_override(view["state_text"], legal)
            for _ in range(max(1, k)):
                decision = agent.choose_action(view["state_text"], legal, prompt_override=override)
                candidates.append(self._decision_to_candidate(decision, legal))
        else:
            raise ValueError(f"unknown method {method!r}")
        return {"status": "ok", "method": method, "candidates": candidates}

    @staticmethod
    def _decision_to_candidate(decision: AgentDecision, legal: list[LegalAction]) -> dict[str, Any]:
        idx = decision.action_index
        desc = legal[idx].description if 0 <= idx < len(legal) else ""
        return {
            "action_index": idx,
            "description": desc,
            "reasoning": decision.reasoning,
            "thinking": decision.thinking,
            "valid": decision.valid,
            "raw_response": decision.raw_response,
            "completion_tokens": decision.completion_tokens,
            "thinking_tokens": decision.thinking_tokens,
            "latency_s": decision.latency_s,
        }

    # --- stepping (commit) ----------------------------------------------
    def step(self, method: str, *, action_index: int | None = None, temperature: float | None = None) -> dict[str, Any]:
        """Choose + commit ONE decision via ``method``. Returns the new frontier
        view plus the committed decision summary, or a non-ok status."""
        try:
            status, view = self._prepare()
        except Exception as exc:  # noqa: BLE001
            self._mark_terminal("simulator_error", str(exc))
            return {"status": "simulator_error", "error": str(exc), "view": self.current_view()}
        if status != "ok":
            self.stored.status = "terminal"
            self.stored.stopped_reason = status
            self._save()
            return {"status": status, "view": self.current_view()}

        legal = view["legal_actions"]
        decision = self._decide(method, view, legal, action_index)
        return self._apply_decision(method, view, legal, decision)

    def _apply_decision(
        self, method: str, view: dict[str, Any], legal: list[LegalAction], decision: AgentDecision
    ) -> dict[str, Any]:
        """Validate + commit a chosen decision (shared by step and stream_step)."""
        if decision.action_index < 0 or decision.action_index >= len(legal):
            decision.valid = False
            decision.metadata = dict(decision.metadata)
            decision.metadata.setdefault("invalid_reason", "action_index out of range")

        if not decision.valid:
            # Mirror run_rollout: keep the invalid response for audit, execute nothing.
            self._commit(view, selected_dict={}, decision=decision, method=method, executed=False)
            self.stored.status = "error"
            self.stored.stopped_reason = "agent_invalid"
            self._save()
            return {"status": "agent_invalid", "decision": self._last_decision_summary(), "view": self.current_view()}

        try:
            selected = self.env.step(decision.action_index)
        except Exception as exc:  # noqa: BLE001 - unsupported combat input states, etc.
            self._mark_terminal("simulator_error", str(exc))
            return {"status": "simulator_error", "error": str(exc), "view": self.current_view()}

        self._commit(view, selected_dict=self.env.action_dict(selected), decision=decision, method=method, executed=True)
        self._save()
        return {"status": "ok", "decision": self._last_decision_summary(), "view": self.current_view()}

    def stream_step(self, method: str = MODEL_METHOD, *, action_index: int | None = None):
        """Generator that streams a model decision token-by-token then commits it.

        Yields ``{"type": "token", "text": seg}`` events as the model decodes,
        then a final ``{"type": "done", ...step-result...}``. Non-model methods
        (or an agent without streaming) fall back to a single non-streamed
        commit and just yield the ``done`` event."""
        try:
            status, view = self._prepare()
        except Exception as exc:  # noqa: BLE001
            self._mark_terminal("simulator_error", str(exc))
            yield {"type": "done", "status": "simulator_error", "error": str(exc), "view": self.current_view()}
            return
        if status != "ok":
            self.stored.status = "terminal"
            self.stored.stopped_reason = status
            self._save()
            yield {"type": "done", "status": status, "view": self.current_view()}
            return

        legal = view["legal_actions"]
        agent = self._model_agent() if method == MODEL_METHOD else None
        if method != MODEL_METHOD or not hasattr(agent, "stream_choose_action"):
            decision = self._decide(method, view, legal, action_index)
            yield {"type": "done", **self._apply_decision(method, view, legal, decision)}
            return

        override = self._model_prompt_override(view["state_text"], legal)
        gen = agent.stream_choose_action(view["state_text"], legal, prompt_override=override)
        decision: AgentDecision | None = None
        try:
            while True:
                segment = next(gen)
                if segment:
                    yield {"type": "token", "text": segment}
        except StopIteration as stop:
            decision = stop.value
        if decision is None:  # defensive: generator yielded nothing and returned None
            decision = AgentDecision(action_index=0, valid=False, raw_response="",
                                     metadata={"error": "empty stream"})
        yield {"type": "done", **self._apply_decision(MODEL_METHOD, view, legal, decision)}

    def sample_n(self, method: str, *, n: int = 1, action_index: int | None = None) -> dict[str, Any]:
        """Commit up to n decisions with ``method`` (user method commits the same
        action_index each step only makes sense for n==1; callers pass n>1 for
        autonomous methods). Stops early on any non-ok status."""
        committed: list[dict[str, Any]] = []
        final_status = "ok"
        for _ in range(max(1, n)):
            result = self.step(method, action_index=action_index)
            final_status = result["status"]
            if "decision" in result:
                committed.append(result["decision"])
            if final_status != "ok":
                break
        return {"status": final_status, "committed": committed, "view": self.current_view()}

    def _decide(
        self, method: str, view: dict[str, Any], legal: list[LegalAction], action_index: int | None
    ) -> AgentDecision:
        if method == USER_METHOD:
            if action_index is None:
                raise ValueError("user method requires action_index")
            return AgentDecision(action_index=int(action_index), raw_response="(user choice)", reasoning="user-selected")
        if method in SCRIPTED_METHODS:
            return self._scripted_agent(method).choose_action(view["state_text"], legal)
        if method == MODEL_METHOD:
            agent = self._model_agent()
            override = self._model_prompt_override(view["state_text"], legal)
            return agent.choose_action(view["state_text"], legal, prompt_override=override)
        raise ValueError(f"unknown method {method!r}")

    def _commit(
        self, view: dict[str, Any], *, selected_dict: dict[str, Any], decision: AgentDecision, method: str, executed: bool
    ) -> None:
        record = build_decision_record(
            world_seed=self.stored.world_seed,
            decision_index=self.next_index,
            state=view["state"],
            state_text=view["state_text"],
            legal_action_dicts=view["legal_action_dicts"],
            selected_action_dict=selected_dict,
            agent_decision=decision,
            after_state=self.env.summary(),
            phase=view["phase"],
            policy_seed=self.policy_seed,
            rollout_index=0,
            action_executed=executed,
        )
        self.history.append(record)
        self.stored.methods.append(method)

    # --- branching -------------------------------------------------------
    def branch_at(self, k: int, *, new_id: str | None = None, label: str | None = None) -> "RolloutSession":
        """Fork a new session at decision ``k``: a fresh env replays the recorded
        action prefix ``history[:k]`` (no model calls). This session is untouched."""
        if k < 0 or k > len(self.history):
            raise IndexError(f"branch point {k} out of range 0..{len(self.history)}")
        s = self.stored
        child = StoredSession(
            session_id=new_id or _new_session_id(),
            label=label or f"{s.label or s.session_id}@{k}",
            parent_id=s.session_id,
            branch_point=k,
            world_seed=s.world_seed,
            ascension=s.ascension,
            combat_control=s.combat_control,
            max_act=s.max_act,
            battle_simulations=s.battle_simulations,
            model_backend=s.model_backend,
            model_id=s.model_id,
            temperature=s.temperature,
            max_tokens=s.max_tokens,
            thinking=s.thinking,
            framing=s.framing,
            prompt_template=s.prompt_template,
            use_advanced_template=s.use_advanced_template,
            methods=list(s.methods[:k]),
            status="active",
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )
        env = self._env_factory(
            world_seed=s.world_seed,
            ascension=s.ascension,
            combat_control=s.combat_control,
            max_act=s.max_act,
            battle_simulations=s.battle_simulations,
        )
        replay_actions(env, [asdict(r)["selected_action"] for r in self.history[:k]])
        child_session = RolloutSession(
            child,
            env,
            store=self._store,
            template_store=self._templates,
            env_factory=self._env_factory,
            agent_builder=self._agent_builder,
            history=list(self.history[:k]),
        )
        child_session._save()
        return child_session

    # --- persistence -----------------------------------------------------
    def _mark_terminal(self, status: str, reason: str) -> None:
        self.stored.status = "error" if status == "simulator_error" else "terminal"
        self.stored.stopped_reason = reason
        self._save()

    def _build_meta(self) -> dict[str, Any] | None:
        if not self.history:
            return None
        try:
            result = RolloutResult(
                world_seed=self.stored.world_seed,
                decisions=self.history,
                terminal_state=self.env.summary(),
                stopped_reason=self.stored.stopped_reason or "active",
                error=None,
                policy_seed=self.policy_seed,
                rollout_index=0,
            )
            model_agent = self._agents.get(MODEL_METHOD)
            meta = build_rollout_meta(
                result,
                self.env,
                model_agent if model_agent is not None else _MetaShim(self.stored),
                {"agent": "interactive", "framing": self.stored.framing},
            )
            return asdict(meta)
        except Exception:  # noqa: BLE001 - meta is best-effort provenance, never fatal
            return None

    def _save(self) -> None:
        self.stored.updated_at = _now_iso()
        self._store.save(self.stored, [asdict(r) for r in self.history], self._build_meta())

    def _session_summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "label": self.stored.label,
            "parent_id": self.stored.parent_id,
            "branch_point": self.stored.branch_point,
            "n_decisions": self.next_index,
            "framing": self.stored.framing,
            "prompt_template": self.stored.prompt_template,
            "use_advanced_template": self.stored.use_advanced_template,
            "combat_control": self.stored.combat_control,
            "world_seed": self.stored.world_seed,
            "model_backend": self.stored.model_backend,
            "model_id": self.stored.model_id,
            "temperature": self.stored.temperature,
            "max_tokens": self.stored.max_tokens,
            "thinking": self.stored.thinking,
            "methods": list(self.stored.methods),
            "session_status": self.stored.status,
            "stopped_reason": self.stored.stopped_reason,
        }

    def _last_decision_summary(self) -> dict[str, Any]:
        record = self.history[-1]
        idx = len(self.history) - 1
        return {
            "decision_index": record.decision_index,
            "method": self.stored.methods[idx],
            "phase": record.phase,
            "selected": record.selected_action,
            "reasoning": record.agent.get("reasoning", ""),
            "thinking": record.agent.get("thinking", ""),
            "valid": record.agent.get("valid", True),
            "action_executed": record.action_executed,
        }


class _MetaShim:
    """Stands in for an agent when no model agent exists, so build_rollout_meta
    can read `name`/`config` for scripted/user-only sessions."""

    def __init__(self, stored: StoredSession) -> None:
        self.name = "interactive"
        self.config = {"framing": stored.framing, "model_id": stored.model_id,
                       "temperature": stored.temperature, "max_tokens": stored.max_tokens,
                       "thinking": stored.thinking}


def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]


class SessionRegistry:
    """Live in-memory sessions keyed by id, backed by the disk store. Handles env
    construction/replay (the only simulator-touching part)."""

    def __init__(
        self,
        *,
        cache_dir: str = "data/interactive",
        default_model_backend: str = "mlx",
        default_model_id: str | None = None,
        env_factory: Callable[..., Any] = default_env_factory,
        agent_builder: Callable[..., Any] = default_agent_builder,
    ) -> None:
        self.store = SessionStore(cache_dir)
        self.templates = TemplateStore(f"{cache_dir.rstrip('/')}/templates")
        self.default_model_backend = default_model_backend
        self.default_model_id = default_model_id
        self._env_factory = env_factory
        self._agent_builder = agent_builder
        self._live: dict[str, RolloutSession] = {}

    def create(
        self,
        *,
        world_seed: int,
        ascension: int = 0,
        combat_control: str = "llm",
        max_act: int = 3,
        battle_simulations: int = 2000,
        framing: str | None = None,
        prompt_template: str | None = None,
        use_advanced_template: bool = False,
        model_backend: str | None = None,
        model_id: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        thinking: bool = False,
        label: str = "",
    ) -> RolloutSession:
        stored = StoredSession(
            session_id=_new_session_id(),
            label=label,
            world_seed=world_seed,
            ascension=ascension,
            combat_control=combat_control,
            max_act=max_act,
            battle_simulations=battle_simulations,
            model_backend=model_backend or self.default_model_backend,
            model_id=model_id if model_id is not None else self.default_model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
            framing=framing if framing is not None else NEUTRAL_FRAME,
            prompt_template=prompt_template,
            use_advanced_template=use_advanced_template,
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )
        env = self._env_factory(
            world_seed=world_seed,
            ascension=ascension,
            combat_control=combat_control,
            max_act=max_act,
            battle_simulations=battle_simulations,
        )
        session = self._wrap(stored, env)
        session._save()
        self._live[stored.session_id] = session
        return session

    def get(self, session_id: str) -> RolloutSession:
        if session_id in self._live:
            return self._live[session_id]
        return self.load(session_id)

    def load(self, session_id: str) -> RolloutSession:
        """Rehydrate a session from disk: rebuild the env and replay full history."""
        stored = self.store.load_stored(session_id)
        decisions = self.store.load_decisions(session_id)
        history = [_record_from_dict(d) for d in decisions]
        env = self._env_factory(
            world_seed=stored.world_seed,
            ascension=stored.ascension,
            combat_control=stored.combat_control,
            max_act=stored.max_act,
            battle_simulations=stored.battle_simulations,
        )
        replay_actions(env, [d["selected_action"] for d in decisions if d.get("action_executed", True) and d.get("selected_action")])
        session = self._wrap(stored, env, history=history)
        self._live[session_id] = session
        return session

    def branch(self, session_id: str, k: int, *, label: str | None = None) -> RolloutSession:
        parent = self.get(session_id)
        child = parent.branch_at(k, label=label)
        self._live[child.session_id] = child
        return child

    def list_sessions(self) -> list[dict[str, Any]]:
        """Tree-ready summaries from disk (authoritative), with live overlay."""
        out: dict[str, dict[str, Any]] = {}
        for s in self.store.list_stored():
            out[s.session_id] = s.summary()
        for sid, sess in self._live.items():
            out[sid] = sess.stored.summary()
        return sorted(out.values(), key=lambda d: (d.get("updated_at") or "", d["session_id"]))

    def evict(self, session_id: str) -> None:
        self._live.pop(session_id, None)

    def delete(self, session_id: str) -> bool:
        self.evict(session_id)
        return self.store.delete(session_id)

    def _wrap(self, stored: StoredSession, env: Any, *, history: list[DecisionRecord] | None = None) -> RolloutSession:
        return RolloutSession(
            stored,
            env,
            store=self.store,
            template_store=self.templates,
            env_factory=self._env_factory,
            agent_builder=self._agent_builder,
            history=history,
        )


def _record_from_dict(d: dict[str, Any]) -> DecisionRecord:
    known = {f for f in DecisionRecord.__dataclass_fields__}  # type: ignore[attr-defined]
    return DecisionRecord(**{k: v for k, v in d.items() if k in known})
