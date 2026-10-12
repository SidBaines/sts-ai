from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.verify_nonllm_cli import verified_dev_seeds, verify_trace
from sts_env.cli import build_parser, observation_hash
from sts_policy.curriculum import world_split


class CliVerificationTests(unittest.TestCase):
    def trace(self, path, *, native_steps=0, error=None):
        seed = 10121
        checkpoint = Path('/tmp/frozen.pt')
        provenance = {'module_sha256': 'native'}
        info = {'native_steps': native_steps, 'native_provenance': provenance}
        ooc = {'phase': 'out_of_combat', 'text': 'map', 'legal_actions': [{'index': 2}]}
        combat = {'phase': 'combat', 'text': 'fight', 'legal_actions': [{'index': 4}]}
        done = {'phase': 'terminal', 'text': 'defeat', 'legal_actions': []}
        rows = [
            {'type': 'header', 'schema': 'sts_env_trace_v1', 'seed': seed,
             'agent': 'sts_policy.model_agent:make_agent',
             'agent_config': {'checkpoint': str(checkpoint), 'device': 'cpu', 'deterministic': True},
             'env_config': {'mode': 'full'}},
            {'type': 'reset', 'observation': ooc, 'info': info},
            {'type': 'action', 'step': 0, 'action': 2, 'observation_hash': observation_hash(ooc)},
            {'type': 'step', 'step': 0, 'action': 2, 'observation': combat, 'info': info},
            {'type': 'action', 'step': 1, 'action': 4, 'observation_hash': observation_hash(combat)},
            {'type': 'step', 'step': 1, 'action': 4, 'observation': done, 'info': info},
            {'type': 'end', 'steps': 2, 'status': 'terminated', 'error': error,
             'info': {**info, 'termination_reason': 'defeat'}, 'elapsed_seconds': 1.0},
        ]
        self.write_rows(path, rows)
        return rows, checkpoint, provenance

    def write_rows(self, path, rows):
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def test_both_heads_zero_native_steps_and_only_end_time_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            rows, checkpoint, provenance = self.trace(path)
            first = verify_trace(path, seed=10121, checkpoint=checkpoint, provenance=provenance)
            self.assertEqual(first['decision_phases'], {'combat': 1, 'out_of_combat': 1})
            rows[-1]['elapsed_seconds'] = 9.
            self.write_rows(path, rows)
            second = verify_trace(path, seed=10121, checkpoint=checkpoint, provenance=provenance)
            self.assertEqual(first['normalized_trace_sha256'], second['normalized_trace_sha256'])
            rows[3]['observation']['text'] = 'different combat'
            rows[4]['observation_hash'] = observation_hash(rows[3]['observation'])
            self.write_rows(path, rows)
            self.assertNotEqual(first['normalized_trace_sha256'],
                                verify_trace(path, seed=10121, checkpoint=checkpoint, provenance=provenance)['normalized_trace_sha256'])

    def test_actions_and_termination_remain_in_determinism_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            rows, checkpoint, provenance = self.trace(path)
            rows[1]['observation']['legal_actions'].append({'index': 5})
            rows[2]['observation_hash'] = observation_hash(rows[1]['observation'])
            self.write_rows(path, rows)
            original = verify_trace(path, seed=10121, checkpoint=checkpoint, provenance=provenance)
            rows[2]['action'] = rows[3]['action'] = 5
            self.write_rows(path, rows)
            changed_action = verify_trace(path, seed=10121, checkpoint=checkpoint, provenance=provenance)
            self.assertNotEqual(original['normalized_trace_sha256'], changed_action['normalized_trace_sha256'])
            rows[-1]['info']['termination_reason'] = 'victory'
            self.write_rows(path, rows)
            changed_termination = verify_trace(path, seed=10121, checkpoint=checkpoint, provenance=provenance)
            self.assertNotEqual(changed_action['normalized_trace_sha256'], changed_termination['normalized_trace_sha256'])

    def test_replay_argv_is_accepted_by_actual_public_cli_parser(self):
        args = build_parser().parse_args(['replay', '/tmp/verified.jsonl'])
        self.assertEqual(args.command, 'replay')
        self.assertEqual(args.path, Path('/tmp/verified.jsonl'))

    def test_automatic_native_decisions_are_rejected_even_after_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            rows, checkpoint, provenance = self.trace(path)
            rows[3]['info'] = {**rows[3]['info'], 'native_steps': 1}
            self.write_rows(path, rows)
            with self.assertRaisesRegex(ValueError, 'automatic native'):
                verify_trace(path, seed=10121, checkpoint=checkpoint, provenance=provenance)

    def test_illegal_public_action_and_native_errors_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            rows, checkpoint, provenance = self.trace(path)
            rows[2]['action'] = rows[3]['action'] = 99
            self.write_rows(path, rows)
            with self.assertRaisesRegex(ValueError, 'public legal'):
                verify_trace(path, seed=10121, checkpoint=checkpoint, provenance=provenance)
            self.trace(path, error={'kind': 'SimulatorError'})
            with self.assertRaisesRegex(ValueError, 'simulator error'):
                verify_trace(path, seed=10121, checkpoint=checkpoint, provenance=provenance)

    def test_missing_combat_phase_or_horizon_cutoff_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            rows, checkpoint, provenance = self.trace(path)
            rows[-1]['info']['termination_reason'] = 'max_steps'
            self.write_rows(path, rows)
            with self.assertRaisesRegex(ValueError, 'horizon'):
                verify_trace(path, seed=10121, checkpoint=checkpoint, provenance=provenance)
            rows[-1]['info']['termination_reason'] = 'defeat'
            rows[3]['observation']['phase'] = 'out_of_combat'
            rows[4]['observation_hash'] = observation_hash(rows[3]['observation'])
            self.write_rows(path, rows)
            with self.assertRaisesRegex(ValueError, 'both combat'):
                verify_trace(path, seed=10121, checkpoint=checkpoint, provenance=provenance)

    def test_success_cannot_hide_an_uncompleted_action(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            rows, checkpoint, provenance = self.trace(path)
            rows.pop(5)
            self.write_rows(path, rows)
            with self.assertRaisesRegex(ValueError, 'uncompleted policy action'):
                verify_trace(path, seed=10121, checkpoint=checkpoint, provenance=provenance)

    def test_only_known_dev_non_embargoed_worlds_are_allowed(self):
        evidence = {'config': {'split': 'dev', 'task_kind': 'run'},
                    'training_environment': {'native_module_sha256': 'native'},
                    'rounds': [{'episodes': [{'world_seed': 10121}, {'world_seed': 10011}]}]}
        snapshot = {'backend': {'module': {'sha256': 'native'}}, 'generated_definitions': {'worlds': []}}
        verified_dev_seeds([10121, 10011], evidence, snapshot)
        unknown_dev = next(seed for seed in range(1000, 1100) if world_split(seed) == 'dev')
        final_seed = next(seed for seed in range(1000, 1100) if world_split(seed) == 'final')
        for seed in (3, unknown_dev, final_seed):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, 'explicitly verified'):
                verified_dev_seeds([seed], evidence, snapshot)
        snapshot['generated_definitions']['worlds'] = [10121]
        with self.assertRaisesRegex(ValueError, 'non-embargoed'):
            verified_dev_seeds([10121], evidence, snapshot)
        snapshot['generated_definitions']['worlds'] = []
        evidence['training_environment']['native_module_sha256'] = 'old'
        with self.assertRaisesRegex(ValueError, 'different native'):
            verified_dev_seeds([10121], evidence, snapshot)
