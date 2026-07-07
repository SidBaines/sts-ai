"""Opt-in MLX policy-gradient LoRA smoke test.

Skipped by default because it needs Apple MLX, a Metal device, and a tiny model
named by STS_MLX_PG_SMOKE_MODEL.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from tests.support import requires_mlx


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


@requires_mlx
class MlxPGTrainSmokeTest(unittest.TestCase):
    def test_train_and_load_pg_adapter(self) -> None:
        base_model = os.environ.get("STS_MLX_PG_SMOKE_MODEL")
        if not base_model:
            self.skipTest("set STS_MLX_PG_SMOKE_MODEL to a tiny MLX model id")

        import mlx_lm
        from sts_ai.train import train_pg_mlx

        records = [
            {
                "prompt": "State: enemy has 6 HP. Legal actions: strike, defend.\nAction:",
                "completion": " strike",
                "advantage": 1.0,
            },
            {
                "prompt": "State: player has low block. Legal actions: strike, defend.\nAction:",
                "completion": " defend",
                "advantage": 0.25,
            },
            {
                "prompt": "State: enemy is attacking. Legal actions: strike, defend.\nAction:",
                "completion": " strike",
                "advantage": -0.5,
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "pg.jsonl"
            adapter_dir = root / "adapter"
            _write_jsonl(dataset_path, records)

            result = train_pg_mlx.train(
                dataset_path,
                base_model,
                adapter_dir,
                lora_r=4,
                grad_accum=2,
                max_seq_len=256,
                kl_beta=0.02,
                epochs=1,
            )

            self.assertEqual(result, adapter_dir)
            self.assertTrue((adapter_dir / "adapters.safetensors").exists())
            self.assertTrue((adapter_dir / "adapter_config.json").exists())
            self.assertTrue((adapter_dir / "trainer_log.json").exists())

            config = json.loads(
                (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config["fine_tune_type"], "lora")
            self.assertIn("num_layers", config)
            self.assertEqual(config["lora_parameters"]["rank"], 4)
            self.assertEqual(config["lora_parameters"]["scale"], 8.0)
            self.assertIn("dropout", config["lora_parameters"])

            history = json.loads(
                (adapter_dir / "trainer_log.json").read_text(encoding="utf-8")
            )
            self.assertIsInstance(history, list)
            self.assertTrue(history)
            self.assertGreaterEqual(len(history), 2)
            metric_keys = {
                "loss",
                "mean_kl",
                "mean_ratio",
                "clip_fraction",
                "mean_advantage",
                "step",
            }
            for entry in history:
                self.assertIsInstance(entry, dict)
                self.assertTrue(metric_keys.issubset(entry))

            mlx_lm.load(base_model, adapter_path=str(adapter_dir))


if __name__ == "__main__":
    unittest.main()
