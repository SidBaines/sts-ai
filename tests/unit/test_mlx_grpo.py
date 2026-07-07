from __future__ import annotations

import functools
from functools import partial
import sys
import types
import unittest
from unittest.mock import patch

from sts_ai.train.mlx_grpo import (
    build_mlx_backend,
    run_parallel_as_streaming,
)
from sts_ai.train.pg_dataset import build_pg_dataset
from tests.support import requires_mlx


class MlxGrpoShimTest(unittest.TestCase):
    def test_run_parallel_as_streaming_maps_concurrency_to_batch_size(self) -> None:
        specs = [(1, 0), (1, 1)]
        make_env = object()
        agent = object()
        output_for = object()
        run_meta = {"run": "meta"}
        calls: list[dict] = []

        def recorder(*args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return ["result"]

        with patch("sts_ai.train.mlx_grpo.run_parallel_rollouts", recorder):
            result = run_parallel_as_streaming(
                specs,
                make_env,
                agent,
                output_for=output_for,
                concurrency=7,
                max_decisions=33,
                run_meta=run_meta,
                hint_cfg=None,
            )

        self.assertEqual(result, ["result"])
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["args"][0], specs)
        self.assertIs(calls[0]["args"][1], make_env)
        self.assertIs(calls[0]["args"][2], agent)
        self.assertIs(calls[0]["kwargs"]["output_for"], output_for)
        self.assertEqual(calls[0]["kwargs"]["batch_size"], 7)
        self.assertEqual(calls[0]["kwargs"]["max_decisions"], 33)
        self.assertIsNone(calls[0]["kwargs"]["max_retries"])
        self.assertIs(calls[0]["kwargs"]["run_meta"], run_meta)

    def test_run_parallel_as_streaming_rejects_hint_cfg(self) -> None:
        with self.assertRaises(ValueError):
            run_parallel_as_streaming(
                [],
                object(),
                object(),
                output_for=object(),
                concurrency=1,
                max_decisions=1,
                hint_cfg=object(),
            )

    def test_build_mlx_backend_wires_components(self) -> None:
        agent_calls: list[dict] = []

        def fake_train(*args, **kwargs):
            return None

        class FakeAgent:
            def __init__(self, **kwargs) -> None:
                agent_calls.append(kwargs)

        with (
            patch("sts_ai.agents.MlxQwenJsonAgent", FakeAgent),
            patch("sts_ai.train.train_pg_mlx.train", fake_train),
        ):
            backend = build_mlx_backend(base_model="m", framing="f", thinking=True, max_seq_len=1234)

        self.assertIsInstance(backend.agent, FakeAgent)
        self.assertIs(backend.run_fn, run_parallel_as_streaming)
        self.assertIsInstance(backend.train_fn, functools.partial)
        self.assertIs(backend.train_fn.func, fake_train)
        self.assertEqual(backend.train_fn.keywords["max_seq_len"], 1234)
        self.assertIsInstance(backend.build_dataset_fn, partial)
        self.assertIs(backend.build_dataset_fn.func, build_pg_dataset)
        self.assertEqual(
            backend.build_dataset_fn.keywords,
            {"require_no_thinking": False},
        )
        self.assertEqual(agent_calls[0]["model_id"], "m")
        self.assertEqual(agent_calls[0]["framing"], "f")
        self.assertTrue(agent_calls[0]["enable_thinking"])


@requires_mlx
class MlxAgentLifecycleTest(unittest.TestCase):
    def test_mlx_agent_config_reports_reasoning_mode(self) -> None:
        sentinel_model = object()
        sentinel_tok = object()

        def fake_load(*args, **kwargs):
            return sentinel_model, sentinel_tok

        fake_mlx_lm = types.ModuleType("mlx_lm")
        fake_mlx_lm.__path__ = []
        fake_mlx_lm.load = fake_load
        fake_mlx_lm.generate = lambda *args, **kwargs: ""
        fake_mlx_lm.batch_generate = None
        fake_mlx_lm.stream_generate = None

        fake_sample_utils = types.ModuleType("mlx_lm.sample_utils")
        fake_sample_utils.make_sampler = lambda temp: object()

        with patch.dict(
            sys.modules,
            {
                "mlx_lm": fake_mlx_lm,
                "mlx_lm.sample_utils": fake_sample_utils,
            },
        ):
            from sts_ai.agents import MlxQwenJsonAgent

            self.assertEqual(
                MlxQwenJsonAgent(model_id="x", enable_thinking=True).config["reasoning_mode"],
                "native",
            )
            self.assertEqual(
                MlxQwenJsonAgent(model_id="x", enable_thinking=False).config["reasoning_mode"],
                "none",
            )

    def test_mlx_agent_lifecycle(self) -> None:
        sentinel_model = object()
        sentinel_tok = object()
        load_calls: list[tuple[tuple, dict]] = []
        clear_cache_calls: list[None] = []

        def fake_load(*args, **kwargs):
            load_calls.append((args, kwargs))
            return sentinel_model, sentinel_tok

        fake_mlx_lm = types.ModuleType("mlx_lm")
        fake_mlx_lm.__path__ = []
        fake_mlx_lm.load = fake_load
        fake_mlx_lm.generate = lambda *args, **kwargs: ""
        fake_mlx_lm.batch_generate = None
        fake_mlx_lm.stream_generate = None

        fake_sample_utils = types.ModuleType("mlx_lm.sample_utils")
        fake_sample_utils.make_sampler = lambda temp: object()

        fake_mlx = types.ModuleType("mlx")
        fake_mlx.__path__ = []
        fake_mx = types.ModuleType("mlx.core")
        fake_mx.clear_cache = lambda: clear_cache_calls.append(None)

        with patch.dict(
            sys.modules,
            {
                "mlx": fake_mlx,
                "mlx.core": fake_mx,
                "mlx_lm": fake_mlx_lm,
                "mlx_lm.sample_utils": fake_sample_utils,
            },
        ):
            from sts_ai.agents import MlxQwenJsonAgent

            agent = MlxQwenJsonAgent(model_id="x")
            self.assertIs(agent.model, sentinel_model)
            self.assertIs(agent.tokenizer, sentinel_tok)
            self.assertEqual(len(load_calls), 1)

            agent.wake()
            self.assertEqual(len(load_calls), 1)

            agent.sleep()
            self.assertIsNone(agent.model)
            self.assertEqual(clear_cache_calls, [None])

            agent.wake()
            self.assertIs(agent.model, sentinel_model)
            self.assertEqual(len(load_calls), 2)
            self.assertEqual(load_calls[-1], (("x",), {}))

            agent.set_adapter("/p")
            self.assertEqual(agent.adapter_path, "/p")
            self.assertIsNone(agent.model)
            self.assertEqual(len(load_calls), 2)

            agent.wake()
            self.assertIs(agent.model, sentinel_model)
            self.assertEqual(len(load_calls), 3)
            self.assertEqual(load_calls[-1], (("x",), {"adapter_path": "/p"}))


if __name__ == "__main__":
    unittest.main()
