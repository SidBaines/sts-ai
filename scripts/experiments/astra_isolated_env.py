"""Experimental spawn-process isolation for native simulator calls.

The parent retains the model and runs the unchanged rollout orchestrator. Each
environment has one CPU-only worker. A timed-out/crashed worker is killed; its
last explicitly requested summary remains available for failure finalization.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from multiprocessing.connection import Connection
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any


def _worker(connection: Connection, env_kwargs: dict[str, Any], fake: bool) -> None:
    """No model imports or model objects cross this boundary."""
    try:
        if fake:
            env = _FakeTimeoutEnv(**env_kwargs)
        else:
            from sts_ai.lightspeed import LightspeedHybridEnv
            env = LightspeedHybridEnv(**env_kwargs)
        # The sole extra getter: establish a safe initial finalization snapshot.
        connection.send(("ready", {"summary": env.summary(),
                                   "module_file": getattr(env.sts, "__file__", None)}))
        while True:
            method, args, kwargs = connection.recv()
            if method == "__close__":
                break
            try:
                value = getattr(env, method)(*args, **kwargs)
                connection.send(("ok", value))
            except Exception as exc:
                connection.send(("error", {"type": type(exc).__name__, "message": str(exc)}))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send(("error", {"type": type(exc).__name__, "message": str(exc)}))
        except (BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class IsolatedLightspeedEnv:
    """LightspeedHybridEnv-compatible RPC surface for serial/parallel rollouts.

    Instantiate only under the caller's __main__ guard. Call close_all() in a
    finally block around run_parallel_rollouts. Fresh processes can still change
    trajectories affected by native undefined behavior; this is an experiment
    condition, not a cross-process reproducibility guarantee.
    """

    _instances: set[IsolatedLightspeedEnv] = set()

    def __init__(self, world_seed: int, *, rpc_timeout_seconds: float = 8.0,
                 _fake_timeout_test: bool = False, **env_kwargs: Any) -> None:
        if rpc_timeout_seconds <= 0:
            raise ValueError("rpc_timeout_seconds must be positive")
        self.world_seed = world_seed
        self.ascension = int(env_kwargs.get("ascension", 0))
        self.combat_control = str(env_kwargs.get("combat_control", "search"))
        self.combat_observation = str(env_kwargs.get("combat_observation", "legacy"))
        self.max_act = int(env_kwargs.get("max_act", 3))
        self.rpc_timeout_seconds = rpc_timeout_seconds
        self.sts = SimpleNamespace(__file__=None)
        self._last_summary = {"world_seed": world_seed, "ascension": self.ascension,
                              "combat_observation": self.combat_observation,
                              "phase": "out_of_combat", "done": False}
        self._failure: str | None = None
        self._closed = False
        context = multiprocessing.get_context("spawn")
        self._connection, child_connection = context.Pipe(duplex=True)
        self._process = context.Process(target=_worker,
            args=(child_connection, {"world_seed": world_seed, **env_kwargs}, _fake_timeout_test),
            name=f"astra-simulator-{world_seed}", daemon=True)
        self._process.start()
        child_connection.close()
        self._instances.add(self)
        try:
            status, payload = self._receive("initialization")
            if status != "ready":
                self._abort(f"worker initialization error: {payload}")
            self._last_summary = payload["summary"]
            self.sts = SimpleNamespace(__file__=payload["module_file"])
        except RuntimeError:
            # make_env is outside the orchestrator's exception boundary. Keep a
            # failed proxy so the first advance is finalized as simulator_error.
            pass

    @property
    def isolation_metadata(self) -> dict[str, Any]:
        return {"isolated_env": True, "start_method": "spawn",
                "rpc_timeout_seconds": self.rpc_timeout_seconds,
                "worker_pid": self._process.pid, "failure": self._failure,
                "summary_after_failure": "last successful explicit summary (may be stale)"}

    def _abort(self, message: str) -> None:
        self._failure = f"isolated simulator seed={self.world_seed}: {message}"
        self.close()
        raise RuntimeError(self._failure)

    def _receive(self, method: str) -> tuple[str, Any]:
        try:
            if not self._connection.poll(self.rpc_timeout_seconds):
                self._abort(f"RPC {method} timed out after {self.rpc_timeout_seconds:g}s")
            return self._connection.recv()
        except (EOFError, OSError) as exc:
            self._abort(f"RPC {method} worker exited: {type(exc).__name__}: {exc}")
        raise AssertionError("unreachable")

    def _rpc(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if self._failure or self._closed:
            raise RuntimeError(self._failure or "isolated simulator is closed")
        try:
            self._connection.send((method, args, kwargs))
        except (BrokenPipeError, OSError) as exc:
            self._abort(f"RPC {method} send failed: {exc}")
        status, payload = self._receive(method)
        if status != "ok":
            self._abort(f"RPC {method} failed: {payload}")
        return payload

    def advance_to_decision(self) -> int:
        return self._rpc("advance_to_decision")

    def is_terminal(self) -> bool:
        return self._rpc("is_terminal")

    def legal_actions(self) -> list[Any]:
        return self._rpc("legal_actions")

    def describe_state(self) -> str:
        return self._rpc("describe_state")

    def map_graph(self) -> dict[str, Any] | None:
        return self._rpc("map_graph")

    def phase(self) -> str:
        return self._rpc("phase")

    def step(self, action_index: int) -> Any:
        return self._rpc("step", action_index)

    def _turn_math_lines(self) -> list[str]:
        return self._rpc("_turn_math_lines")

    def summary(self) -> dict[str, Any]:
        if not self._failure and not self._closed:
            try:
                self._last_summary = self._rpc("summary")
            except RuntimeError:
                # summary() also occurs outside try blocks in record building.
                # A dead worker is reported at the next guarded advance call.
                pass
        return deepcopy(self._last_summary)

    @staticmethod
    def action_dict(action: Any) -> dict[str, Any]:
        return asdict(action)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=0.25)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=0.25)
        else:
            self._process.join(timeout=0.1)
        self._connection.close()
        self._instances.discard(self)

    @classmethod
    def close_all(cls) -> None:
        for env in list(cls._instances):
            env.close()

    def __enter__(self) -> IsolatedLightspeedEnv:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class _FakeTimeoutEnv:
    def __init__(self, world_seed: int, **kwargs: Any) -> None:
        self.world_seed = world_seed
        self.sts = SimpleNamespace(__file__=None)

    def summary(self) -> dict[str, Any]:
        return {"world_seed": self.world_seed, "floor": 7, "cur_hp": 42,
                "act": 1, "done": False}

    def advance_to_decision(self) -> int:
        time.sleep(30)
        return 0


def self_check(output_dir: Path) -> None:
    from sts_ai.agents import RandomLegalAgent
    from sts_ai.lightspeed import LightspeedHybridEnv
    from sts_ai.rollout import run_rollout

    output_dir.mkdir(parents=True, exist_ok=True)
    checks = []
    try:
        env = IsolatedLightspeedEnv(9999, rpc_timeout_seconds=2, _fake_timeout_test=True)
        before = env.summary()
        env.rpc_timeout_seconds = 0.15
        started = time.monotonic()
        try:
            env.advance_to_decision()
        except RuntimeError as exc:
            assert "timed out" in str(exc)
        else:
            raise AssertionError("fake hang did not time out")
        assert time.monotonic() - started < 2
        assert env.summary() == before and not env._process.is_alive()
        checks.append({"check": "fake_timeout_killed_and_cached_summary", "passed": True})
        # Test the unchanged orchestrator's finalization path with a failed worker.
        from sts_ai.parallel_rollout import run_parallel_rollouts
        results = run_parallel_rollouts([(9999, 0)], lambda seed: IsolatedLightspeedEnv(
            seed, rpc_timeout_seconds=0.3, _fake_timeout_test=True), RandomLegalAgent(), batch_size=1)
        assert results[0].stopped_reason == "simulator_error"
        checks.append({"check": "parallel_rollout_finalizes_timeout", "passed": True})
        for seed in (3, 1000):
            kwargs = {"world_seed": seed, "max_act": 1, "battle_simulations": 50,
                      "combat_observation": "combat_public_v2"}
            direct_path = output_dir / f"seed_{seed}_direct.jsonl"
            isolated_path = output_dir / f"seed_{seed}_isolated.jsonl"
            for path in (direct_path, isolated_path):
                if path.exists():
                    raise FileExistsError(path)
            direct = run_rollout(LightspeedHybridEnv(**kwargs), RandomLegalAgent(),
                                 max_decisions=150, output_path=direct_path)
            with IsolatedLightspeedEnv(**kwargs) as proxy:
                isolated = run_rollout(proxy, RandomLegalAgent(), max_decisions=150, output_path=isolated_path)
            direct_hash = hashlib.sha256(direct_path.read_bytes()).hexdigest()
            isolated_hash = hashlib.sha256(isolated_path.read_bytes()).hexdigest()
            checks.append({"check": "random_trajectory_equivalence", "world_seed": seed,
                           "direct_sha256": direct_hash, "isolated_sha256": isolated_hash,
                           "equal": direct_hash == isolated_hash, "n_decisions": len(direct.decisions),
                           "direct_stop": direct.stopped_reason, "isolated_stop": isolated.stopped_reason})
            assert direct_hash == isolated_hash
    finally:
        IsolatedLightspeedEnv.close_all()
        (output_dir / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/astra_5sep/isolation_verification"))
    args = parser.parse_args()
    if args.self_check:
        self_check(args.output_dir)
