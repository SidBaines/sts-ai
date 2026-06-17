"""Opt-in MLX LoRA smoke test.

Skipped by default because it needs the native simulator, Apple MLX, and a tiny
model named by STS_MLX_SMOKE_MODEL.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from sts_ai.prompting import NEUTRAL_FRAME
from sts_ai.schemas import AgentDecision, LegalAction
from tests.support import requires_mlx, requires_simulator


class ScriptedJsonAgent:
    name = "scripted-json"

    @property
    def config(self) -> dict[str, object]:
        return {
            "model_id": self.name,
            "framing": NEUTRAL_FRAME,
            "reasoning_mode": "none",
            "temperature": 0.0,
            "max_tokens": 0,
            "thinking": False,
            "max_retries": 0,
        }

    def reseed(self, policy_seed: int) -> None:
        return None

    def choose_action(
        self,
        state_text: str,
        legal_actions: list[LegalAction],
    ) -> AgentDecision:
        descriptions = [action.description for action in legal_actions]
        for index, description in enumerate(descriptions):
            if description.startswith("play ") or description.startswith("drink potion"):
                return self._decision(index)
        for index, description in enumerate(descriptions):
            if description == "end turn":
                return self._decision(index)
        return self._decision(0)

    def _decision(self, action_index: int) -> AgentDecision:
        response = json.dumps({"reasoning": "smoke", "action_index": action_index})
        return AgentDecision(
            action_index=action_index,
            raw_response=response,
            reasoning="smoke",
        )


def _write_dataset(path: Path, examples: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True) + "\n")


@requires_simulator
@requires_mlx
class MlxTrainSmokeTest(unittest.TestCase):
    def test_generate_build_train_and_load_adapter(self):
        model_id = os.environ.get("STS_MLX_SMOKE_MODEL")
        if not model_id:
            self.skipTest("set STS_MLX_SMOKE_MODEL to a tiny MLX model id")

        from mlx_lm import load
        from sts_ai.lightspeed import LightspeedHybridEnv
        from sts_ai.rollout import run_rollout
        from sts_ai.train import dataset_builder, train_mlx

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rollout_dir = root / "rollouts"
            rollout_dir.mkdir()
            agent = ScriptedJsonAgent()
            for seed in (200, 201):
                env = LightspeedHybridEnv(
                    world_seed=seed,
                    battle_simulations=50,
                    max_act=1,
                    combat_control="llm",
                )
                run_rollout(
                    env,
                    agent,
                    max_decisions=40,
                    output_path=rollout_dir / f"seed_{seed}_r0.jsonl",
                )

            _model, tokenizer = load(model_id)
            examples, manifest = dataset_builder.build_dataset(
                rollout_dir,
                framing=NEUTRAL_FRAME,
                tokenizer=tokenizer,
                tokenizer_id=model_id,
                min_positives=1,
                fallback_floor_quantile=0.0,
                require_no_thinking=True,
                require_framing_match=False,
            )
            self.assertTrue(examples)

            dataset_path = root / "sft.jsonl"
            manifest_path = root / "sft.manifest.json"
            _write_dataset(dataset_path, examples)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            adapter_dir = train_mlx.train(
                dataset_path,
                model_id,
                root / "adapter",
                iters=2,
                num_layers=2,
                data_dir=root / "mlx_data",
                manifest_path=manifest_path,
            )
            self.assertTrue(adapter_dir.exists())
            load(model_id, adapter_path=str(adapter_dir))


if __name__ == "__main__":
    unittest.main()
