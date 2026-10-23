from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sts_env import cli
from sts_env.env import SlayTheSpireEnv
from tests.unit.test_environment_base import FakeBackend


def observation(step: int) -> dict:
    return {"text": f"state {step}", "state": {"step": step}, "phase": "combat",
            "legal_actions": [{"index": 0, "description": "end turn", "bits": 0}] if step == 0 else []}


class FakeEnv:
    def __init__(self, *, error: bool = False, mismatch: bool = False) -> None:
        self.error = error
        self.mismatch = mismatch
        self.closed = False

    def reset(self, *, seed: int) -> tuple[dict, dict]:
        return observation(0), {"world_seed": seed, "native_provenance": {"module_sha256": "fake"}}

    def step(self, action: int) -> tuple:
        if self.error:
            raise TimeoutError("native operation stalled")
        obs = observation(1)
        if self.mismatch:
            obs["text"] = "different"
        return obs, 1.0, True, False, {"outcome": "win"}

    def native_action(self) -> int:
        return 0

    def close(self) -> None:
        self.closed = True


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def run_episode(self, env: FakeEnv, path: Path) -> dict:
        with patch.object(cli, "make_env", return_value=env):
            return cli.run_episode(seed=3, policy_seed=4, agent_spec="first", agent_config={}, config={"mode": "full"}, output=path)

    def test_trace_is_incremental_replayable_and_never_overwritten(self) -> None:
        path = self.directory / "run.jsonl"
        env = FakeEnv()
        result = self.run_episode(env, path)
        self.assertTrue(env.closed)
        self.assertEqual(result["reward"], 1)
        self.assertEqual(result["status"], "terminated")
        records = cli.read_trace(path)
        self.assertEqual([r["type"] for r in records], ["header", "reset", "action", "step", "end"])
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_episode(FakeEnv(), path)
        self.assertEqual(before, path.read_bytes())
        with patch.object(cli, "make_env", return_value=FakeEnv()):
            self.assertEqual(cli.replay_trace(path)["verified_steps"], 1)
        with patch.object(cli, "make_env", return_value=FakeEnv(mismatch=True)):
            with self.assertRaisesRegex(ValueError, "step 0"):
                cli.replay_trace(path)

    def test_first_run_creates_nested_output_directories(self) -> None:
        path = self.directory / "data" / "nested" / "smoke.jsonl"
        self.assertFalse(path.parent.exists())
        self.assertEqual(self.run_episode(FakeEnv(), path)["status"], "terminated")
        self.assertTrue(path.exists())
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_episode(FakeEnv(), path)
        self.assertEqual(path.read_bytes(), before)

    def test_failure_retains_attempted_action_and_closes_environment(self) -> None:
        env = FakeEnv(error=True)
        path = self.directory / "failure.jsonl"
        result = self.run_episode(env, path)
        self.assertTrue(env.closed)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["kind"], "TimeoutError")
        summary = cli.inspect_trace(path)
        self.assertEqual(summary["attempted_actions"], 1)
        self.assertEqual(summary["completed_steps"], 0)
        with patch.object(cli, "make_env", return_value=FakeEnv()):
            replay = cli.replay_trace(path)
        self.assertEqual(replay["uncompleted_actions"], 1)

    def test_replay_rejects_conflicting_action_journal(self) -> None:
        path = self.directory / "tampered.jsonl"
        self.run_episode(FakeEnv(), path)
        records = cli.read_trace(path)
        records[2]["action"] = 99
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        with patch.object(cli, "make_env") as factory:
            with self.assertRaisesRegex(ValueError, "disagree at step 0"):
                cli.replay_trace(path)
            factory.assert_not_called()

    def test_replay_rejects_version_changes_and_post_end_records(self) -> None:
        path = self.directory / "version.jsonl"
        self.run_episode(FakeEnv(), path)
        records = cli.read_trace(path)
        records[0]["observation_version"] = "other"
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        with patch.object(cli, "make_env") as factory:
            with self.assertRaisesRegex(ValueError, "observation_version mismatch"):
                cli.replay_trace(path)
            factory.assert_not_called()
        path.write_text(path.read_text() + json.dumps(records[1]) + "\n")
        with self.assertRaisesRegex(ValueError, "after its end"):
            cli.read_trace(path)

    def test_forged_python_interface_digest_is_rejected_before_launching_native(self) -> None:
        path = self.directory / "forged-interface.jsonl"
        self.run_episode(FakeEnv(), path)
        records = cli.read_trace(path)
        records[0]["python_interface_sha256"] = "0" * 64
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        with patch.object(cli, "make_env") as factory:
            with self.assertRaisesRegex(ValueError, "python_interface_sha256 mismatch"):
                cli.replay_trace(path)
            factory.assert_not_called()

    def test_existing_evaluation_directory_is_preserved_before_agent_loading(self) -> None:
        output = self.directory / "existing"
        output.mkdir()
        sentinel = output / "precious.txt"
        sentinel.write_text("preserved")
        with patch.object(cli, "load_agent") as factory, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["evaluate", "--seeds", "3", "--output", str(output)]), 1)
            factory.assert_not_called()
        self.assertEqual(sentinel.read_text(), "preserved")

    def test_per_seed_output_failure_keeps_batch_accounting_and_continues(self) -> None:
        output = self.directory / "output-error"
        successful = {"status": "terminated", "reward": 1.0, "info": {"outcome": "win"}}
        with patch.object(cli, "run_episode", side_effect=[PermissionError("cannot write trace"), successful]), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["evaluate", "--seeds", "3,4", "--output", str(output)]), 1)
        report = json.loads((output / "summary.json").read_text())
        self.assertEqual(report["completed"], 2)
        self.assertEqual(report["errors"], 1)
        self.assertEqual(report["episodes"][0]["error"]["kind"], "PermissionError")

    def test_evaluation_continues_after_failed_seed_and_saves_all_outcomes(self) -> None:
        output = self.directory / "evaluate"
        with patch.object(cli, "make_env", side_effect=[FakeEnv(error=True), FakeEnv()]), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = cli.main(["evaluate", "--agent", "first", "--seeds", "3,4", "--output", str(output)])
        self.assertEqual(status, 1)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["completed"], 2)
        self.assertEqual(summary["mean_reward"], 0.5)
        self.assertTrue((output / "seed-3.jsonl").exists())
        self.assertTrue((output / "seed-4.jsonl").exists())

    def test_evaluation_counts_keyed_act3_win_as_victory(self) -> None:
        output = self.directory / "keyed-victory"
        with patch("sts_env.env._make_backend", return_value=FakeBackend(cutoff=True, act=3)):
            with patch.object(cli, "make_env", side_effect=lambda config: SlayTheSpireEnv(isolated=False, **config)):
                with contextlib.redirect_stdout(io.StringIO()):
                    status = cli.main(["evaluate", "--agent", "first", "--seeds", "3",
                                       "--max-act", "3", "--output", str(output)])
        self.assertEqual(status, 0)
        report = json.loads((output / "summary.json").read_text())
        self.assertEqual(report["mean_reward"], 1.0)
        self.assertEqual(report["statuses"]["terminated"], 1)
        self.assertEqual(report["statuses"]["truncated"], 0)
        self.assertEqual(report["outcomes"], {"GameOutcome.PLAYER_VICTORY": 1})
        episode = report["episodes"][0]
        self.assertEqual(episode["info"]["termination_reason"], "victory")
        records = cli.read_trace(output / "seed-3.jsonl")
        step = next(record for record in records if record["type"] == "step")
        self.assertEqual((step["reward"], step["terminated"], step["truncated"]),
                         (1.0, True, False))

    def test_evaluation_reuses_models_only_with_episode_reset_contract(self) -> None:
        class Policy:
            def __init__(self) -> None:
                self.seeds = []

            def reset(self, seed: int) -> None:
                self.seeds.append(seed)

            def choose_action(self, observation: dict) -> int:
                return 0

        policy = Policy()
        with patch.object(cli, "load_agent", return_value=policy) as factory, patch.object(cli, "make_env", side_effect=[FakeEnv(), FakeEnv()]), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["evaluate", "--seeds", "3,4", "--output", str(self.directory / "reuse")]), 0)
            self.assertEqual(factory.call_count, 1)
        self.assertEqual(policy.seeds, [3, 4])

        class StatefulNoReset:
            def __init__(self) -> None:
                self.called = False

            def choose_action(self, observation: dict) -> int:
                if self.called:
                    raise AssertionError("episode state leaked")
                self.called = True
                return 0

        with patch.object(cli, "load_agent", side_effect=lambda *args: StatefulNoReset()) as factory, patch.object(cli, "make_env", side_effect=[FakeEnv(), FakeEnv()]), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["evaluate", "--seeds", "3,4", "--output", str(self.directory / "fresh")]), 0)
            self.assertEqual(factory.call_count, 2)

    def test_inspect_accepts_partial_prefix_and_rejects_unknown_schema(self) -> None:
        path = self.directory / "partial.jsonl"
        self.run_episode(FakeEnv(), path)
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n")
        self.assertFalse(cli.inspect_trace(path)["complete"])
        path.write_text('{"type":"header","schema":"other"}\n')
        with self.assertRaisesRegex(ValueError, cli.TRACE_SCHEMA):
            cli.inspect_trace(path)

    def test_seed_parsing_and_cli_parameter_validation(self) -> None:
        self.assertEqual(cli.parse_seeds("3:6,10"), [3, 4, 5, 10])
        for value in ("3,3", "6:3", "", "1:3:4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cli.parse_seeds(value)
        for flag, value in (("--step-timeout", "nan"), ("--max-steps", "0"), ("--agent-config", "[]")):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.build_parser().parse_args(["run", flag, value])

    def test_play_reprompts_and_quit_is_explicit(self) -> None:
        with patch("builtins.input", side_effect=["bad", "99", "0"]), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.human_action(observation(0)), 0)
        with patch("builtins.input", return_value="q"), contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(cli.human_action(observation(0)))


if __name__ == "__main__":
    unittest.main()
