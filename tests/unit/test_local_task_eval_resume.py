from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.local_task_eval import (
    build_eval_config,
    postflight_eval_outputs,
    preflight_resume_outputs,
    validate_resume_meta,
)
from sts_ai.seeding import derive_policy_seed
from sts_ai.provenance import adapter_provenance


def _args(root: Path, *, overwrite: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        task="gremlin_nob",
        manifest=root / "source.json",
        split="holdout",
        model="example/model",
        backend="vllm",
        adapter_path=None,
        output_dir=root / "eval",
        max_decisions=80,
        combat_observation="legacy",
        max_tokens=8192,
        temperature=0.7,
        top_p=0.95,
        top_k=64,
        max_retries=1,
        thinking=True,
        output_contract="reasoning_action",
        preserve_special_tokens="auto",
        enable_prefix_caching=True,
        concurrency=12,
        rollouts_per_window=1,
        battle_simulations=50,
        max_act=3,
        overwrite=overwrite,
    )


def _window() -> dict:
    return {
        "world_seed": 8,
        "ordinal": 0,
        "window_id": "seed_8_r0_w0",
        "source_stem": "seed_8_r0",
        "split": "holdout",
    }


def _meta(config: dict) -> dict:
    return {
        "world_seed": 8,
        "rollout_index": 0,
        "policy_seed": derive_policy_seed(8, 0),
        "model_id": config["model_id"],
        "framing": config["framing"],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        "thinking": config["thinking"],
        "max_retries": config["max_retries"],
        "battle_simulations": config["battle_simulations"],
        "combat_control": "llm",
        "n_decisions": 1,
        "extra": {
            "local_task_eval": True,
            "task_id": config["task_id"],
            "split": config["split"],
            "source_manifest_sha256": config["source_manifest_sha256"],
            "adapter_path": config["adapter_path"],
            "combat_observation": config["combat_observation"],
            "competence_interface_version": config["combat_observation"],
            "output_contract": config["output_contract"],
            "orchestrator": "streaming",
            "concurrency": config["concurrency"],
            "local_task_eval_config": config,
            "agent_config": {
                "model_id": config["model_id"],
                "framing": config["framing"],
                "temperature": config["temperature"],
                "top_p": config["top_p"],
                "top_k": config["top_k"],
                "max_tokens": config["max_tokens"],
                "thinking": config["thinking"],
                "reasoning_mode": "native",
                "max_retries": config["max_retries"],
                "backend": "vllm",
                "preserve_special_tokens": True,
                "enable_prefix_caching": True,
                "output_contract": config["output_contract"],
                "adapter_path": config["adapter_path"],
            },
            "interface_provenance": {
                "combat_observation": config["combat_observation"],
                "competence_interface_version": config["combat_observation"],
                "output_contract": config["output_contract"],
                "prompt_probe_sha256": "1" * 64,
                "chat_template_probe_hash": "2" * 64,
                "python_serializer_sha256": "3" * 64,
                "glossary_sha256": "4" * 64,
                "prompting_sha256": "5" * 64,
                "simulator_patch_sha256": "6" * 64,
                "simulator_binary_sha256": "7" * 64,
            },
            "local_task": {
                "task_id": config["task_id"],
                "window_id": "seed_8_r0_w0",
                "source_stem": "seed_8_r0",
                "split": config["split"],
                "reward": 0.5,
                "metrics": {"reward": 0.5},
            },
        },
    }


class LocalTaskEvalResumeTest(unittest.TestCase):
    def test_adapter_provenance_is_content_addressed(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp) / "adapter"
            adapter.mkdir()
            weights = adapter / "adapters.safetensors"
            weights.write_bytes(b"first")
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            first = adapter_provenance(adapter)
            weights.write_bytes(b"second")
            second = adapter_provenance(adapter)
            self.assertNotEqual(first["identity_sha256"], second["identity_sha256"])

    def _setup(self, root: Path, *, overwrite: bool = False):
        args = _args(root, overwrite=overwrite)
        args.output_dir.mkdir()
        config = build_eval_config(
            args,
            task_id="gremlin_nob",
            manifest_sha256="source-hash",
            windows=[_window()],
        )
        path = args.output_dir / "seed_8_r0.meta.json"
        path.write_text(json.dumps(_meta(config)), encoding="utf-8")
        trace = {
            "world_seed": 8,
            "rollout_index": 0,
            "policy_seed": derive_policy_seed(8, 0),
            "decision_index": 0,
        }
        (args.output_dir / "seed_8_r0.jsonl").write_text(
            json.dumps(trace) + "\n",
            encoding="utf-8",
        )
        return args, config, path

    def test_exact_completed_meta_is_safe_to_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, config, path = self._setup(Path(tmp))
            validate_resume_meta(
                path,
                expected_eval_config=config,
                window=_window(),
                world_seed=8,
                rollout_index=0,
            )
            preflight_resume_outputs(
                args,
                windows=[_window()],
                expected_eval_config=config,
            )

    def test_refuses_stale_command_contract_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, config, path = self._setup(Path(tmp))
            value = json.loads(path.read_text(encoding="utf-8"))
            value["extra"]["local_task_eval_config"]["temperature"] = 0.2
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to resume incompatible"):
                preflight_resume_outputs(
                    args,
                    windows=[_window()],
                    expected_eval_config=config,
                )

            args.overwrite = True
            preflight_resume_outputs(
                args,
                windows=[_window()],
                expected_eval_config=config,
            )

    def test_refuses_tampered_generic_or_window_provenance(self):
        for path_to_change, changed in (
            (("model_id",), "other/model"),
            (("extra", "local_task", "window_id"), "other-window"),
            (("policy_seed",), 123),
        ):
            with self.subTest(field=path_to_change), tempfile.TemporaryDirectory() as tmp:
                _, config, path = self._setup(Path(tmp))
                value = json.loads(path.read_text(encoding="utf-8"))
                cursor = value
                for part in path_to_change[:-1]:
                    cursor = cursor[part]
                cursor[path_to_change[-1]] = changed
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "refusing to resume incompatible"):
                    validate_resume_meta(
                        path,
                        expected_eval_config=config,
                        window=_window(),
                        world_seed=8,
                        rollout_index=0,
                    )

    def test_refuses_unexpected_completed_episode_even_with_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, config, _ = self._setup(Path(tmp), overwrite=True)
            unexpected = args.output_dir / "seed_9_r0.meta.json"
            unexpected.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside the current command/cohort"):
                preflight_resume_outputs(
                    args,
                    windows=[_window()],
                    expected_eval_config=config,
                )

    def test_refuses_duplicate_manifest_output_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, config, _ = self._setup(Path(tmp), overwrite=True)
            duplicate = dict(_window())
            duplicate["window_id"] = "different-window"
            with self.assertRaisesRegex(ValueError, "multiple windows to output identity"):
                preflight_resume_outputs(
                    args,
                    windows=[_window(), duplicate],
                    expected_eval_config=config,
                )

    def test_repairs_streaming_meta_then_validates_before_resume(self):
        class FakeTask:
            task_id = "gremlin_nob"

            def completion_reason(self, terminal_state):
                _ = terminal_state
                return "task_complete"

            def metrics_from_episode(self, decisions, terminal_state, stopped_reason, window):
                _ = terminal_state, stopped_reason, window
                return {
                    "label": "won",
                    "reward": 0.5,
                    "hp_loss": 20,
                    "survived": True,
                    "n_decisions": len(decisions),
                    "n_turns": 1,
                    "action_counts": {"other": len(decisions)},
                }

        with tempfile.TemporaryDirectory() as tmp:
            args, config, path = self._setup(Path(tmp))
            value = json.loads(path.read_text(encoding="utf-8"))
            value["extra"].pop("local_task")
            path.write_text(json.dumps(value), encoding="utf-8")

            preflight_resume_outputs(
                args,
                windows=[_window()],
                expected_eval_config=config,
                task=FakeTask(),
            )
            repaired = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                repaired["extra"]["local_task"]["window_id"],
                "seed_8_r0_w0",
            )

    def test_postflight_requires_every_completed_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, config, path = self._setup(Path(tmp))
            path.unlink()
            with self.assertRaisesRegex(ValueError, "eval incomplete"):
                postflight_eval_outputs(
                    args,
                    windows=[_window()],
                    expected_eval_config=config,
                )


if __name__ == "__main__":
    unittest.main()
