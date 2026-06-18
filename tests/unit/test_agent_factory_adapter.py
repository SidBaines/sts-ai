import unittest
from unittest.mock import patch

from sts_ai.agent_factory import build_agent


class AgentFactoryAdapterTest(unittest.TestCase):
    def test_mlx_forwards_adapter_path(self):
        with patch("sts_ai.agent_factory.MlxQwenJsonAgent") as fake_agent:
            agent = build_agent("mlx", model="m", adapter_path="/tmp/ad")

        self.assertIs(agent, fake_agent.return_value)
        fake_agent.assert_called_once()
        self.assertEqual(fake_agent.call_args.kwargs["adapter_path"], "/tmp/ad")

    def test_vllm_forwards_adapter_path_and_rank(self):
        with patch("sts_ai.agent_factory.VllmJsonAgent") as fake_agent:
            agent = build_agent("vllm", model="m", adapter_path="/tmp/ad", max_lora_rank=32)

        self.assertIs(agent, fake_agent.return_value)
        fake_agent.assert_called_once()
        self.assertEqual(fake_agent.call_args.kwargs["adapter_path"], "/tmp/ad")
        self.assertEqual(fake_agent.call_args.kwargs["max_lora_rank"], 32)

    def test_vllm_forwards_preserve_special_tokens(self):
        with patch("sts_ai.agent_factory.VllmJsonAgent") as fake_agent:
            agent = build_agent("vllm", model="m", preserve_special_tokens=True)

        self.assertIs(agent, fake_agent.return_value)
        fake_agent.assert_called_once()
        self.assertTrue(fake_agent.call_args.kwargs["preserve_special_tokens"])

    def test_mlx_default_forwards_no_adapter(self):
        with patch("sts_ai.agent_factory.MlxQwenJsonAgent") as fake_agent:
            agent = build_agent("mlx", model="m")

        self.assertIs(agent, fake_agent.return_value)
        fake_agent.assert_called_once()
        self.assertIsNone(fake_agent.call_args.kwargs["adapter_path"])
        self.assertNotIn("preserve_special_tokens", fake_agent.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
