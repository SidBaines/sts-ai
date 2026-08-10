from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.local_task_compare import build_report as _strict_build_report, parse_args
from sts_ai.local_tasks.runner import window_rollout_indices
from sts_ai.provenance import file_sha256
from sts_ai.seeding import derive_policy_seed


def _write_meta(
    root: Path,
    *,
    world_seed: int,
    rollout_index: int,
    window_id: str,
    reward: float,
    hp_loss: int,
    survived: bool,
    task_id: str = "gremlin_nob",
    source_hash: str = "source-hash",
    adapter_path: str | None = None,
    combat_observation: str = "legacy",
    temperature: float = 0.7,
    omit_metric: str | None = None,
    subdir: str | None = None,
) -> Path:
    metrics = {
        "reward": reward,
        "hp_loss": hp_loss,
        "survived": survived,
        "label": "won" if survived else "loss",
        "n_decisions": 10,
        "n_turns": 3,
        "action_counts": {"attack": 5, "block": 3, "skill": 1, "other": 1},
    }
    metrics.pop(omit_metric, None)
    local_task = {
        "task_id": task_id,
        "window_id": window_id,
        "source_stem": window_id.rsplit("_w", 1)[0],
        "split": "holdout",
        "label": "won" if survived else "loss",
        "reward": reward,
        "metrics": metrics,
    }
    if omit_metric == "reward":
        local_task.pop("reward")
    agent_config = {
        "model_id": "example/model",
        "framing": "neutral",
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 8192,
        "thinking": True,
        "reasoning_mode": "native",
        "max_retries": 1,
        "backend": "vllm",
        "preserve_special_tokens": True,
        "enable_prefix_caching": True,
        "adapter_path": adapter_path,
        "output_contract": "reasoning_action",
    }
    adapter_identity = (
        {"path": adapter_path, "identity_sha256": "c" * 64}
        if adapter_path is not None
        else None
    )
    meta = {
        "world_seed": world_seed,
        "rollout_index": rollout_index,
        "policy_seed": derive_policy_seed(world_seed, rollout_index),
        "agent": "vllm_json",
        "model_id": "example/model",
        "framing": "neutral",
        "temperature": temperature,
        "max_tokens": 8192,
        "thinking": True,
        "max_retries": 1,
        "battle_simulations": 50,
        "combat_control": "llm",
        "stopped_reason": "task_complete" if survived else "player_loss",
        "n_decisions": 10,
        "n_invalid": 0,
        "extra": {
            "local_task_eval": True,
            "task_id": task_id,
            "split": "holdout",
            "source_manifest_sha256": source_hash,
            "adapter_path": adapter_path,
            "adapter_provenance": adapter_identity,
            "combat_observation": combat_observation,
            "competence_interface_version": combat_observation,
            "output_contract": "reasoning_action",
            "orchestrator": "streaming",
            "concurrency": 12,
            "agent_config": agent_config,
            "interface_provenance": {
                "output_contract": "reasoning_action",
                "combat_observation": combat_observation,
                "competence_interface_version": combat_observation,
                "prompt_probe_sha256": "1" * 64,
                "chat_template_probe_hash": "2" * 64,
                "simulator_binary_sha256": "3" * 64,
                "python_serializer_sha256": "4" * 64,
                "glossary_sha256": "5" * 64,
                "prompting_sha256": "6" * 64,
                "simulator_patch_sha256": "7" * 64,
            },
            "local_task_eval_config": {
                "version": 1,
                "task_id": task_id,
                "split": "holdout",
                "source_manifest_sha256": source_hash,
                "model_id": "example/model",
                "framing": "neutral",
                "backend": "vllm",
                "adapter_path": adapter_path,
                "adapter_provenance": adapter_identity,
                "combat_observation": combat_observation,
                "max_decisions": 80,
                "max_tokens": 8192,
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 64,
                "max_retries": 1,
                "thinking": True,
                "output_contract": "reasoning_action",
                "preserve_special_tokens": "auto",
                "enable_prefix_caching": True,
                "concurrency": 12,
                "rollouts_per_window": 1,
                "battle_simulations": 50,
                "max_act": 3,
                "expected_window_count": 1,
                "expected_window_ids_sha256": "8" * 64,
                "expected_rollout_count": 1,
                "expected_rollout_identities_sha256": "9" * 64,
            },
            "local_task": local_task,
        },
    }
    directory = root / subdir if subdir else root
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"seed_{world_seed}_r{rollout_index}.meta.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    return path


def build_report(
    base: Path,
    trained: Path,
    *,
    metric: str,
    allowed_interventions=(),
    expected_windows: list[dict] | None = None,
    rollouts_per_window: int | None = None,
    trained_source_override: str | None = None,
):
    """Test helper which materializes the exact source manifest contract."""
    metas = []
    for root in (base, trained):
        for path in sorted(root.rglob("*.meta.json")):
            metas.append(json.loads(path.read_text(encoding="utf-8")))
    if rollouts_per_window is None:
        counts: dict[str, set[tuple[int, int]]] = {}
        for meta in metas:
            task = ((meta.get("extra") or {}).get("local_task") or {})
            window_id = task.get("window_id")
            if window_id:
                counts.setdefault(str(window_id), set()).add(
                    (int(meta["world_seed"]), int(meta["rollout_index"]))
                )
        rollouts_per_window = max((len(value) for value in counts.values()), default=1)
    if expected_windows is None:
        by_id: dict[str, dict] = {}
        for meta in metas:
            task = ((meta.get("extra") or {}).get("local_task") or {})
            window_id = task.get("window_id")
            if not window_id or window_id in by_id:
                continue
            by_id[str(window_id)] = {
                "window_id": str(window_id),
                "source_stem": task.get("source_stem"),
                "world_seed": int(meta["world_seed"]),
                "ordinal": int(meta["rollout_index"]) // rollouts_per_window,
                "split": task.get("split"),
            }
        expected_windows = list(by_id.values())
    task_id = "gremlin_nob"
    if metas:
        task_id = str(((metas[0].get("extra") or {}).get("local_task") or {}).get("task_id"))
    manifest_path = base.parent / "source.json"
    manifest_path.write_text(
        json.dumps({"task_id": task_id, "windows": expected_windows}),
        encoding="utf-8",
    )
    source_hash = file_sha256(manifest_path)
    expected_window_ids = sorted(str(window["window_id"]) for window in expected_windows)
    expected_identities = sorted(
        (int(window["world_seed"]), rollout_index)
        for window in expected_windows
        for rollout_index in window_rollout_indices(window, rollouts_per_window)
    )
    for root in (base, trained):
        for path in root.rglob("*.meta.json"):
            meta = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(meta.get("extra"), dict):
                meta["extra"]["source_manifest_sha256"] = source_hash
                eval_config = meta["extra"]["local_task_eval_config"]
                eval_config["source_manifest_sha256"] = source_hash
                eval_config["rollouts_per_window"] = rollouts_per_window
                eval_config["expected_window_count"] = len(expected_window_ids)
                eval_config["expected_window_ids_sha256"] = hashlib.sha256(
                    json.dumps(expected_window_ids, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                eval_config["expected_rollout_count"] = len(expected_identities)
                eval_config["expected_rollout_identities_sha256"] = hashlib.sha256(
                    json.dumps(expected_identities, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
            path.write_text(json.dumps(meta), encoding="utf-8")
    if trained_source_override is not None:
        for path in trained.rglob("*.meta.json"):
            meta = json.loads(path.read_text(encoding="utf-8"))
            meta["extra"]["source_manifest_sha256"] = trained_source_override
            meta["extra"]["local_task_eval_config"][
                "source_manifest_sha256"
            ] = trained_source_override
            path.write_text(json.dumps(meta), encoding="utf-8")
    return _strict_build_report(
        base,
        trained,
        metric=metric,
        allowed_interventions=allowed_interventions,
        manifest=manifest_path,
        rollouts_per_window=rollouts_per_window,
    )


class LocalTaskCompareTest(unittest.TestCase):
    def _dirs(self, root: Path) -> tuple[Path, Path]:
        base = root / "base"
        trained = root / "trained"
        base.mkdir()
        trained.mkdir()
        return base, trained

    def _matched_pair(self, base: Path, trained: Path, **kwargs) -> None:
        common = {
            "world_seed": 8,
            "rollout_index": 0,
            "window_id": "seed_8_r0_w0",
            "hp_loss": 40,
            "survived": False,
        }
        _write_meta(base, reward=0.0, **common, **kwargs)
        _write_meta(trained, reward=1.0, **common, **kwargs)

    def test_cli_accepts_only_explicit_intervention_allowlist(self):
        args = parse_args(
            [
                "--base",
                "base",
                "--trained",
                "trained",
                "--manifest",
                "source.json",
                "--rollouts-per-window",
                "8",
                "--allow-intervention",
                "adapter_path",
                "--allow-intervention",
                "combat_observation",
            ]
        )
        self.assertEqual(
            args.allow_intervention,
            ["adapter_path", "combat_observation"],
        )

    def test_multiple_rollouts_per_window_are_averaged(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, trained = self._dirs(Path(tmp))
            for rollout_index, base_reward in enumerate((0.0, 1.0)):
                common = dict(
                    world_seed=8,
                    rollout_index=rollout_index,
                    window_id="seed_8_r0_w0",
                    hp_loss=40,
                    survived=False,
                )
                _write_meta(base, reward=base_reward, **common)
                _write_meta(trained, reward=1.0, **common)

            report = build_report(base, trained, metric="reward")

            self.assertEqual(report["paired"]["n"], 1)
            self.assertAlmostEqual(report["paired"]["mean_delta"], 0.5)
            self.assertEqual(
                report["paired"]["rollouts_per_window"]["base"],
                {"min": 2, "max": 2},
            )
            self.assertEqual(report["validation"]["status"], "passed")
            self.assertTrue(report["validation"]["rollout_identities"]["exact_match"])

    def test_refuses_window_intersection_or_rollout_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, trained = self._dirs(Path(tmp))
            self._matched_pair(base, trained)
            _write_meta(
                base,
                world_seed=12,
                rollout_index=1,
                window_id="seed_12_r0_w0",
                reward=-1.0,
                hp_loss=80,
                survived=False,
            )
            with self.assertRaisesRegex(ValueError, "exact identical window sets"):
                build_report(base, trained, metric="reward")

    def test_refuses_identically_incomplete_manifest_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, trained = self._dirs(Path(tmp))
            self._matched_pair(base, trained)
            missing_window = {
                "window_id": "seed_12_r0_w0",
                "source_stem": "seed_12_r0",
                "world_seed": 12,
                "ordinal": 0,
                "split": "holdout",
            }
            with self.assertRaisesRegex(ValueError, "complete manifest window cohort"):
                build_report(
                    base,
                    trained,
                    metric="reward",
                    expected_windows=[
                        {
                            "window_id": "seed_8_r0_w0",
                            "source_stem": "seed_8_r0",
                            "world_seed": 8,
                            "ordinal": 0,
                            "split": "holdout",
                        },
                        missing_window,
                    ],
                    rollouts_per_window=1,
                )

    def test_refuses_different_rollout_count_in_same_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, trained = self._dirs(Path(tmp))
            self._matched_pair(base, trained)
            _write_meta(
                base,
                world_seed=8,
                rollout_index=1,
                window_id="seed_8_r0_w0",
                reward=0.0,
                hp_loss=40,
                survived=False,
            )
            with self.assertRaisesRegex(ValueError, "exact identical rollout identities"):
                build_report(base, trained, metric="reward")

    def test_hp_loss_metric_reads_nested_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, trained = self._dirs(Path(tmp))
            for rollout_index, base_hp, trained_hp in ((0, 40, 10), (1, 20, 10)):
                common = dict(
                    world_seed=4,
                    rollout_index=rollout_index,
                    window_id="seed_4_r0_w0",
                    reward=0.5,
                    survived=True,
                )
                _write_meta(base, hp_loss=base_hp, **common)
                _write_meta(trained, hp_loss=trained_hp, **common)
            report = build_report(base, trained, metric="hp_loss")
            self.assertAlmostEqual(report["paired"]["mean_delta"], 10.0 - 30.0)

    def test_refuses_empty_arm_and_missing_or_empty_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, trained = self._dirs(Path(tmp))
            with self.assertRaisesRegex(ValueError, "contains no rollout metadata"):
                build_report(base, trained, metric="reward")
            self._matched_pair(base, trained)
            with self.assertRaisesRegex(ValueError, "metric name must be non-empty"):
                build_report(base, trained, metric="")
            trained_meta = next(trained.glob("*.meta.json"))
            value = json.loads(trained_meta.read_text(encoding="utf-8"))
            value["extra"]["local_task"]["metrics"].pop("hp_loss")
            trained_meta.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required field 'hp_loss'"):
                build_report(base, trained, metric="hp_loss")

    def test_refuses_duplicate_rollout_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, trained = self._dirs(Path(tmp))
            self._matched_pair(base, trained)
            _write_meta(
                base,
                world_seed=8,
                rollout_index=0,
                window_id="seed_8_r0_w0",
                reward=0.0,
                hp_loss=40,
                survived=False,
                subdir="duplicate",
            )
            with self.assertRaisesRegex(ValueError, "duplicate rollout identity"):
                build_report(base, trained, metric="reward")

    def test_refuses_task_source_and_generation_mismatches(self):
        cases = (
            ("task cohort", {"task_id": "lagavulin"}, "task/source cohort"),
            ("generation", {"temperature": 0.2}, "generation config"),
        )
        for label, trained_kwargs, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                base, trained = self._dirs(Path(tmp))
                self._matched_pair(base, trained)
                path = next(trained.glob("*.meta.json"))
                value = json.loads(path.read_text(encoding="utf-8"))
                if "task_id" in trained_kwargs:
                    value["extra"]["task_id"] = trained_kwargs["task_id"]
                    value["extra"]["local_task"]["task_id"] = trained_kwargs["task_id"]
                if "temperature" in trained_kwargs:
                    value["temperature"] = trained_kwargs["temperature"]
                    value["extra"]["agent_config"]["temperature"] = trained_kwargs["temperature"]
                    value["extra"]["local_task_eval_config"][
                        "temperature"
                    ] = trained_kwargs["temperature"]
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    build_report(base, trained, metric="reward")

        with tempfile.TemporaryDirectory() as tmp:
            base, trained = self._dirs(Path(tmp))
            self._matched_pair(base, trained)
            with self.assertRaisesRegex(ValueError, "task/source cohort"):
                build_report(
                    base,
                    trained,
                    metric="reward",
                    trained_source_override="other",
                )

    def test_refuses_mixed_generation_config_within_one_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, trained = self._dirs(Path(tmp))
            for rollout_index in (0, 1):
                common = dict(
                    world_seed=8,
                    rollout_index=rollout_index,
                    window_id="seed_8_r0_w0",
                    reward=0.0,
                    hp_loss=40,
                    survived=False,
                )
                _write_meta(base, **common)
                _write_meta(trained, **common)
            path = trained / "seed_8_r1.meta.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["temperature"] = 0.2
            value["extra"]["agent_config"]["temperature"] = 0.2
            value["extra"]["local_task_eval_config"]["temperature"] = 0.2
            path.write_text(json.dumps(value), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "trained mixes generation configs"):
                build_report(base, trained, metric="reward")

    def test_refuses_malformed_reported_outcome_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, trained = self._dirs(Path(tmp))
            self._matched_pair(base, trained)
            path = next(trained.glob("*.meta.json"))
            value = json.loads(path.read_text(encoding="utf-8"))
            value["extra"]["local_task"]["metrics"]["survived"] = "false"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metrics.survived must be boolean"):
                build_report(base, trained, metric="reward")

    def test_refuses_interface_fingerprint_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, trained = self._dirs(Path(tmp))
            self._matched_pair(base, trained)
            path = next(trained.glob("*.meta.json"))
            value = json.loads(path.read_text(encoding="utf-8"))
            value["extra"]["interface_provenance"]["prompt_probe_sha256"] = "a" * 64
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "generation config"):
                build_report(base, trained, metric="reward")

    def test_adapter_and_observation_differences_require_explicit_allowlist(self):
        for intervention, trained_value in (
            ("adapter_path", "/tmp/adapter"),
            ("combat_observation", "combat_public_v1"),
        ):
            with self.subTest(intervention=intervention), tempfile.TemporaryDirectory() as tmp:
                base, trained = self._dirs(Path(tmp))
                self._matched_pair(base, trained)
                path = next(trained.glob("*.meta.json"))
                value = json.loads(path.read_text(encoding="utf-8"))
                value["extra"][intervention] = trained_value
                if intervention == "adapter_path":
                    value["extra"]["agent_config"][intervention] = trained_value
                    value["extra"]["adapter_provenance"] = {
                        "path": trained_value,
                        "identity_sha256": "d" * 64,
                    }
                    value["extra"]["local_task_eval_config"][intervention] = trained_value
                    value["extra"]["local_task_eval_config"][
                        "adapter_provenance"
                    ] = value["extra"]["adapter_provenance"]
                else:
                    value["extra"]["competence_interface_version"] = trained_value
                    value["extra"]["local_task_eval_config"][intervention] = trained_value
                    value["extra"]["interface_provenance"][intervention] = trained_value
                    value["extra"]["interface_provenance"][
                        "competence_interface_version"
                    ] = trained_value
                path.write_text(json.dumps(value), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, f"differs in {intervention}"):
                    build_report(base, trained, metric="reward")
                report = build_report(
                    base,
                    trained,
                    metric="reward",
                    allowed_interventions=[intervention],
                )
                observed = report["validation"]["observed_interventions"][intervention]
                self.assertTrue(observed["different"])
                self.assertTrue(observed["allowed"])

    def test_refuses_declared_but_absent_intervention(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, trained = self._dirs(Path(tmp))
            self._matched_pair(base, trained)
            with self.assertRaisesRegex(ValueError, "declared intervention adapter_path"):
                build_report(
                    base,
                    trained,
                    metric="reward",
                    allowed_interventions=["adapter_path"],
                )


class WindowRolloutIndicesTest(unittest.TestCase):
    def test_k1_matches_historical_ordinal_indexing(self):
        self.assertEqual(window_rollout_indices({"ordinal": 0}, 1), [0])
        self.assertEqual(window_rollout_indices({"ordinal": 2}, 1), [2])

    def test_k_gt_1_is_collision_free_across_ordinals(self):
        w0 = window_rollout_indices({"ordinal": 0}, 4)
        w1 = window_rollout_indices({"ordinal": 1}, 4)
        self.assertEqual(w0, [0, 1, 2, 3])
        self.assertEqual(w1, [4, 5, 6, 7])
        self.assertFalse(set(w0) & set(w1))

    def test_rejects_non_positive_k(self):
        with self.assertRaises(ValueError):
            window_rollout_indices({"ordinal": 0}, 0)


if __name__ == "__main__":
    unittest.main()
