from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from sts_ai.state_sanity import SanityFinding, phantom_power_findings


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "audit_state_sanity.py"


def _state(
    *,
    enemy_suffix: str = (
        "intent GREMLIN_NOB_BELLOW (no damage; buffs itself with Enrage (2))"
    ),
    player_powers: str = "none",
    relics: str = "Burning Blood",
) -> str:
    return "\n".join(
        [
            "Battle turn 0",
            "Player HP: 70/80, block: 0, energy: 3/3",
            "Turn counters: cards played 0, attacks 0, skills 0, discarded 0",
            "Stance: NEUTRAL",
            "Player resources: draw per turn 5, orb slots 0",
            f"Player powers: {player_powers}",
            "Enemies:",
            f"  [0] GREMLIN_NOB HP 85/85, block 0, {enemy_suffix}",
            "Hand:",
            "  [0] Defend [Skill] (cost 1)",
            "Piles: draw 9, discard 0, exhaust 0",
            f"Relics: {relics}",
            "Potions (capacity 3): [0] empty, [1] empty, [2] empty",
        ]
    )


def _row(
    public_state_hash: str,
    *,
    window_id: str,
    world_seed: int,
    state_text: str,
) -> dict[str, object]:
    return {
        "state_text": state_text,
        "window_id": window_id,
        "public_state_hash": public_state_hash,
        "world_seed": world_seed,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _run_cli(
    labels: list[Path],
    out: Path,
    *,
    task_id: str = "gremlin_nob",
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(_SCRIPT)]
    for path in labels:
        command.extend(["--labels", str(path)])
    command.extend(["--task", task_id, "--out", str(out)])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_ROOT / "src")
    return subprocess.run(
        command,
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


class PhantomPowerFindingsTest(unittest.TestCase):
    def test_clean_gremlin_nob_state_has_no_findings(self) -> None:
        state_text = _state(
            enemy_suffix=(
                "intent GREMLIN_NOB_RUSH (deal 16), Strength -2, Enrage 2, "
                "Vulnerable 1, Weak 1"
            ),
            player_powers=(
                "Metallicize 3, Rupture 1, Vulnerable 2, Weak 1, Strength 4, "
                "Flame Barrier 4"
            ),
        )

        self.assertEqual(
            phantom_power_findings(state_text, task_id="gremlin_nob"),
            [],
        )

    def test_noncombat_state_text_fails_closed(self) -> None:
        for state_text in ("", "Map screen"):
            with self.subTest(state_text=state_text):
                with self.assertRaisesRegex(
                    ValueError,
                    "state_text_not_a_combat_state",
                ):
                    phantom_power_findings(
                        state_text,
                        task_id="gremlin_nob",
                    )

    def test_renamed_combat_headers_fail_closed(self) -> None:
        renamed_states = (
            _state().replace("Enemies:", "Opponents:"),
            _state().replace("Player powers:", "Player effects:"),
        )
        for state_text in renamed_states:
            with self.subTest(state_text=state_text):
                with self.assertRaisesRegex(
                    ValueError,
                    "state_text_not_a_combat_state",
                ):
                    phantom_power_findings(
                        state_text,
                        task_id="gremlin_nob",
                    )

    def test_impossible_enemy_powers_are_flagged_with_amounts(self) -> None:
        state_text = _state(
            enemy_suffix=(
                "intent GREMLIN_NOB_BELLOW, (no attack) Metallicize 4, Regen 3"
            )
        )

        self.assertEqual(
            phantom_power_findings(state_text, task_id="gremlin_nob"),
            [
                SanityFinding(
                    side="enemy",
                    power="Metallicize",
                    amount=4,
                    reason="enemy_power_impossible_for_encounter",
                ),
                SanityFinding(
                    side="enemy",
                    power="Regen",
                    amount=3,
                    reason="enemy_power_impossible_for_encounter",
                ),
            ],
        )

    def test_enemy_block_ends_at_inline_hand_header(self) -> None:
        state_text = _state().replace(
            "Hand:\n  [0] Defend [Skill] (cost 1)",
            "Hand: empty\nRecent intent history, Metallicize 4",
        )

        self.assertEqual(
            phantom_power_findings(state_text, task_id="gremlin_nob"),
            [],
        )

    def test_decoy_power_above_enemy_header_is_ignored(self) -> None:
        state_text = "Metallicize 4\n" + _state()

        self.assertEqual(
            phantom_power_findings(state_text, task_id="gremlin_nob"),
            [],
        )

    def test_player_thorns_requires_bronze_scales(self) -> None:
        without_source = _state(player_powers="Thorns 3")
        with_source = _state(
            player_powers="Thorns 3", relics="Burning Blood, Bronze Scales"
        )

        self.assertEqual(
            phantom_power_findings(without_source, task_id="gremlin_nob"),
            [
                SanityFinding(
                    side="player",
                    power="Thorns",
                    amount=3,
                    reason="player_thorns_without_source",
                )
            ],
        )
        self.assertEqual(
            phantom_power_findings(with_source, task_id="gremlin_nob"),
            [],
        )

    def test_relic_counters_do_not_hide_thorns_source(self) -> None:
        state_text = _state(
            player_powers="Thorns 3",
            relics="Burning Blood, Bronze Scales [counter 3, live]",
        )

        self.assertEqual(
            phantom_power_findings(state_text, task_id="gremlin_nob"),
            [],
        )

    def test_player_metallicize_is_not_flagged(self) -> None:
        self.assertEqual(
            phantom_power_findings(
                _state(player_powers="Metallicize 4"),
                task_id="gremlin_nob",
            ),
            [],
        )

    def test_player_buffer_is_flagged(self) -> None:
        self.assertEqual(
            phantom_power_findings(
                _state(player_powers="Buffer 1"),
                task_id="gremlin_nob",
            ),
            [
                SanityFinding(
                    side="player",
                    power="Buffer",
                    amount=1,
                    reason="player_buffer_without_source",
                )
            ],
        )

    def test_unknown_task_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            phantom_power_findings(_state(), task_id="lagavulin")


class AuditStateSanityCliTest(unittest.TestCase):
    def test_unknown_task_with_empty_input_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "empty.jsonl"
            labels.write_text("", encoding="utf-8")
            output = root / "quarantine.json"

            completed = _run_cli(
                [labels],
                output,
                task_id="lagavulin",
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("unsupported task_id", completed.stderr)
            self.assertFalse(output.exists())

    def test_end_to_end_output_is_deterministic_and_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels.jsonl"
            rows = [
                _row(
                    "hash-b",
                    window_id="window-b",
                    world_seed=2,
                    state_text=_state(player_powers="Buffer 2"),
                ),
                _row(
                    "hash-c",
                    window_id="window-c",
                    world_seed=3,
                    state_text=_state(player_powers="Buffer 3"),
                ),
                _row(
                    "hash-a",
                    window_id="window-a",
                    world_seed=1,
                    state_text=_state(
                        enemy_suffix=(
                            "intent GREMLIN_NOB_BELLOW, "
                            "(no attack) Metallicize 4, Regen 3"
                        )
                    ),
                ),
                _row(
                    "hash-clean",
                    window_id="window-d",
                    world_seed=4,
                    state_text=_state(),
                ),
            ]
            _write_jsonl(labels, rows)
            first = root / "first.json"
            second = root / "second.json"

            first_run = _run_cli([labels], first)
            second_run = _run_cli([labels], second)

            self.assertEqual(first_run.returncode, 0, first_run.stderr)
            self.assertEqual(second_run.returncode, 0, second_run.stderr)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            report = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(report["kind"], "state_sanity_quarantine")
            self.assertEqual(report["version"], 1)
            self.assertEqual(report["task_id"], "gremlin_nob")
            self.assertEqual(
                report["inputs"],
                [
                    {
                        "path": str(labels),
                        "sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
                        "n_rows": 4,
                    }
                ],
            )
            self.assertEqual(
                [
                    (item["window_id"], item["public_state_hash"])
                    for item in report["findings"]
                ],
                [
                    ("window-a", "hash-a"),
                    ("window-b", "hash-b"),
                    ("window-c", "hash-c"),
                ],
            )
            self.assertEqual(
                report["findings"][0],
                {
                    "public_state_hash": "hash-a",
                    "window_id": "window-a",
                    "world_seed": 1,
                    "findings": [
                        {
                            "side": "enemy",
                            "power": "Metallicize",
                            "amount": 4,
                            "reason": "enemy_power_impossible_for_encounter",
                        },
                        {
                            "side": "enemy",
                            "power": "Regen",
                            "amount": 3,
                            "reason": "enemy_power_impossible_for_encounter",
                        },
                    ],
                },
            )
            self.assertEqual(
                report["quarantined_windows"],
                ["window-a", "window-b", "window-c"],
            )
            self.assertEqual(
                report["quarantined_state_hashes"],
                ["hash-a", "hash-b", "hash-c"],
            )
            self.assertEqual(
                report["summary"],
                {
                    "n_rows": 4,
                    "n_states_flagged": 3,
                    "n_windows_flagged": 3,
                },
            )

    def test_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels.jsonl"
            _write_jsonl(
                labels,
                [
                    _row(
                        "hash-clean",
                        window_id="window-a",
                        world_seed=1,
                        state_text=_state(),
                    )
                ],
            )
            output = root / "quarantine.json"
            output.write_text("keep me", encoding="utf-8")

            completed = _run_cli([labels], output)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("output_must_be_fresh", completed.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me")

    def test_multiple_inputs_dedupe_by_public_state_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_labels = root / "first.jsonl"
            second_labels = root / "second.jsonl"
            duplicate = _row(
                "same-hash",
                window_id="same-window",
                world_seed=7,
                state_text=_state(player_powers="Buffer 1"),
            )
            _write_jsonl(first_labels, [duplicate])
            _write_jsonl(second_labels, [duplicate])
            output = root / "quarantine.json"

            completed = _run_cli([first_labels, second_labels], output)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(report["inputs"]), 2)
            self.assertEqual(len(report["findings"]), 1)
            self.assertEqual(report["summary"]["n_rows"], 2)
            self.assertEqual(report["summary"]["n_states_flagged"], 1)

    def test_duplicate_hash_with_different_findings_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_labels = root / "first.jsonl"
            second_labels = root / "second.jsonl"
            _write_jsonl(
                first_labels,
                [
                    _row(
                        "same-hash",
                        window_id="same-window",
                        world_seed=7,
                        state_text=_state(),
                    )
                ],
            )
            _write_jsonl(
                second_labels,
                [
                    _row(
                        "same-hash",
                        window_id="same-window",
                        world_seed=7,
                        state_text=_state(player_powers="Buffer 1"),
                    )
                ],
            )
            output = root / "quarantine.json"

            completed = _run_cli([first_labels, second_labels], output)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "conflicting_duplicate_public_state_hash",
                completed.stderr,
            )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
