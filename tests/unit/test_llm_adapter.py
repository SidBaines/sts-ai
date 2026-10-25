import hashlib
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from sts_llm.teacher_worker import checkpoint_versions, source_revision
from sts_llm.observations import PROMPT_VERSION, prompt
from sts_llm.generation import parse_action, K2_REASONING_CLOSERS
from sts_llm.teacher import NonLLMTeacher, teacher_reward, validate_teacher
from sts_llm.dataset import (SPLIT_SCHEME_VERSION, canonical_world_identity, prepare_rows, verified_split,
                             world_split)
from sts_llm.eval import LLMAgent, InvalidModelAction, evaluate_saved, main as eval_main

OBS = {'phase': 'combat', 'state': {'act': 2, 'observation_version': 'sts_env_public_v3'},
       'text': 'Player HP20/80. Enemy intends12. Strike: Deal6 damage. Defend: Gain5 Block.',
       'legal_actions': [{'index': 3, 'description': 'play Strike -> enemy'},
                         {'index': 9, 'description': 'end turn'}],
       'info': {'world_seed': 'SECRET-SEED'}, 'teacher': 'SECRET-TEACHER'}
ROW = {'id': 'test', 'phase': 'combat', 'legal_indices': [3, 9, 12],
       'teacher': {'legal_indices': [9, 3], 'probabilities': [.75, .25], 'excluded_indices': [12]}}


class ParserTests(unittest.TestCase):
    def test_all_effort_closers_and_actual_eos(self):
        for closer in K2_REASONING_CLOSERS:
            result = parse_action('consider {"action_index":999}' + closer +
                                  '{"action_index":9}<|ifm|im_end|><|endoftext|>', [3, 9])
            self.assertTrue(result.valid, result)
            self.assertEqual(result.action_index, 9)
            self.assertIn('999', result.reasoning)

    def test_unfinished_thought_cannot_become_an_action(self):
        result = parse_action('Maybe {"action_index":9}', [9])
        self.assertFalse(result.valid)
        self.assertEqual(result.error, 'missing_reasoning_boundary')

    def test_truncated_valid_json_is_still_failure(self):
        result = parse_action('</ifm|think>{"action_index":9}', [9], finish_reason='length')
        self.assertFalse(result.valid)
        self.assertTrue(result.truncated)

    def test_native_separate_final_channel(self):
        self.assertTrue(parse_action('{"action_index":9}', [9], reasoning_content='').valid)
        self.assertTrue(parse_action('{"action_index":9}', [9], final_channel=True).valid)

    def test_malformed_and_ambiguous_finals(self):
        values = ['{"action_index":true}', '{"action_index":9.0}',
                  '{"action_index":9,"action_index":3}', '{"action_index":9,"why":"x"}',
                  '{"action_index":9}{"action_index":3}', 'The answer is {"action_index":9}']
        for value in values:
            self.assertFalse(parse_action('</ifm|think>' + value, [3, 9]).valid)
        self.assertEqual(parse_action('</ifm|think>{"action_index":4}', [3, 9]).error, 'illegal_action_index')


class TeacherRewardTests(unittest.TestCase):
    def test_display_mapping_and_excluded_valid_action(self):
        self.assertEqual(teacher_reward(ROW, 3), .25)
        self.assertEqual(teacher_reward(ROW, 9), .75)
        self.assertEqual(teacher_reward(ROW, 12, invalid_reward=-1), 0)
        self.assertEqual(teacher_reward(ROW, None, invalid_reward=-1), -1)
        self.assertAlmostEqual(teacher_reward(ROW, 3, mode='log_probability'), math.log(.25))
        with self.assertRaises(ValueError):
            teacher_reward(ROW, None, mode='log_probability', invalid_reward=-1)

    def test_invalid_distribution_fails_instead_of_inventing_reward(self):
        for probs in ([.1, .2], [float('nan'), .2], [-.5, 1.5]):
            with self.assertRaises(ValueError):
                validate_teacher({**ROW, 'teacher': {**ROW['teacher'], 'probabilities': probs}})
        with self.assertRaises(ValueError):
            validate_teacher({**ROW, 'teacher': {**ROW['teacher'], 'excluded_indices': []}})


class RuntimeProvenanceTests(unittest.TestCase):
    def test_export_inside_another_checkout_retains_original_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'teacher_runtime_manifest.json').write_text(json.dumps({'source_revision': 'teacher-original'}))
            with patch('sts_llm.teacher_worker.subprocess.check_output', return_value='wrong-checkout') as git:
                self.assertEqual(source_revision(Path(tmp)), 'teacher-original')
                git.assert_not_called()

    def test_checkpoint_versions_come_from_metadata_with_recorded_fallback(self):
        recorded = checkpoint_versions({'format_version': 7, 'metadata': {'encoder_version': 'public_candidate_v9'}})
        self.assertEqual((recorded['encoder_version'], recorded['checkpoint_format_version']), ('public_candidate_v9', 7))
        self.assertEqual(set(recorded['version_source'].values()), {'checkpoint_metadata'})
        for payload in (None, {}, {'metadata': {}}):
            fallback = checkpoint_versions(payload)
            self.assertEqual((fallback['encoder_version'], fallback['checkpoint_format_version']),
                             ('public_candidate_v2', 1))
            self.assertEqual(set(fallback['version_source'].values()), {'fallback_constant'})

    def test_teacher_requests_are_bounded_chunks(self):
        teacher = NonLLMTeacher.__new__(NonLLMTeacher)
        teacher.batch_size = 2
        sizes = []
        def rpc(op, observations):
            sizes.append(len(observations))
            return [{'legal_indices': [3, 9], 'probabilities': [.5, .5], 'excluded_indices': []}
                    for _ in observations]
        teacher._rpc = rpc
        self.assertEqual(len(teacher.distributions([OBS] * 5)), 5)
        self.assertEqual(sizes, [2, 2, 1])
        teacher._rpc = lambda op, observations: []
        with self.assertRaisesRegex(Exception, 'incomplete batch'):
            teacher.distributions([OBS])


class PromptAndDatasetTests(unittest.TestCase):
    def test_prompt_whitelists_public_text_and_menu(self):
        rendered = json.dumps(prompt(OBS))
        self.assertNotIn('SECRET', rendered)
        self.assertNotIn('world_seed', rendered)
        self.assertIn('Strike: Deal6 damage', rendered)
        self.assertIn('[9] end turn', rendered)

    def test_same_world_held_out_as_unit(self):
        class Teacher:
            def distributions(self, observations):
                return [{'legal_indices': [3, 9], 'probabilities': [.25, .75], 'excluded_indices': []}
                        for _ in observations]
        rows = [{'id': 'a', 'observation': OBS, 'world_identity': 'same-world'},
                {'id': 'b', 'observation': {**OBS, 'text': OBS['text']+' Turn2'}, 'world_identity': 'same-world'}]
        prepared = prepare_rows(rows, Teacher())
        self.assertEqual(prepared[0]['split'], prepared[1]['split'])
        self.assertEqual(prepared[0]['split'], world_split('same-world'))
        self.assertNotIn('SECRET', json.dumps(prepared[0]['messages']))

    class UniformTeacher:
        def distributions(self, observations):
            return [{'legal_indices': [3, 9], 'probabilities': [.25, .75], 'excluded_indices': []}
                    for _ in observations]

    def test_inline_and_source_referenced_forms_of_one_world_share_a_split(self):
        # Find a seed whose raw decimal string and 'world:<seed>' hashed to different v1 buckets.
        def v1_split(text):
            bucket = int(hashlib.sha256(('sts-llm-world-split-v1:' + text).encode()).hexdigest()[:8], 16) % 10
            return 'test' if bucket == 0 else 'validation' if bucket == 1 else 'train'
        seed = next(n for n in range(1000) if v1_split(str(n)) != v1_split(f'world:{n}'))
        with tempfile.TemporaryDirectory() as tmp:
            blob = json.dumps({'world_seed': seed, 'labels': [{'public_hash': 'h', 'observation': OBS}]}).encode()
            source = Path(tmp) / f'{seed}.json'
            source.write_bytes(blob)
            reference = {'path': str(source), 'file_sha256': hashlib.sha256(blob).hexdigest(),
                         'label_ordinal': 0, 'public_hash': 'h'}
            rows = [{'id': 'int', 'observation': OBS, 'world_identity': seed},
                    {'id': 'text', 'observation': OBS, 'world_identity': str(seed)},
                    {'id': 'source', 'source_record_identity': reference}]
            prepared = prepare_rows(rows, self.UniformTeacher())
        self.assertEqual({row['world_identity'] for row in prepared}, {f'world:{seed}'})
        self.assertEqual({row['split'] for row in prepared}, {v1_split(f'world:{seed}')})
        self.assertEqual({row['split_scheme_version'] for row in prepared}, {SPLIT_SCHEME_VERSION})
        for row in prepared:
            self.assertEqual(verified_split(row), row['split'])

    def test_canonical_world_identity(self):
        for value in (12, '12', 'world:12', 'world:012'):
            self.assertEqual(canonical_world_identity(value), 'world:12')
        self.assertEqual(canonical_world_identity('same-world'), 'same-world')
        for bad in (True, None, 1.5, -1, ''):
            with self.assertRaises(ValueError):
                canonical_world_identity(bad)

    def test_verified_split_refuses_forged_or_unknown_scheme_rows(self):
        world = next(f'world:{n}' for n in range(1000) if world_split(f'world:{n}') == 'test')
        self.assertEqual(verified_split({'id': 'v1', 'world_identity': world, 'split': 'test'}), 'test')
        with self.assertRaisesRegex(ValueError, 'declares split'):
            verified_split({'id': 'forged', 'world_identity': world, 'split': 'train'})
        with self.assertRaisesRegex(ValueError, 'unknown split scheme'):
            verified_split({'id': 'future', 'world_identity': world, 'split': 'test',
                            'split_scheme_version': SPLIT_SCHEME_VERSION + 1})


class EvaluationTests(unittest.TestCase):
    def test_reused_agent_records_episodes_and_restarts_sampling_rng(self):
        class Client:
            def __init__(self):
                self.calls = []

            def generate(self, messages, **kwargs):
                self.calls.append((messages, kwargs['seed']))
                return SimpleNamespace(text='{"action_index":9}', reasoning_content='',
                                       finish_reason='stop', raw={}, elapsed_seconds=.1)

        client = Client()
        original_observation = json.dumps(OBS, sort_keys=True)
        expected_messages = prompt(OBS)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'responses.jsonl'
            agent = LLMAgent(client, response_log=str(path))
            self.assertEqual(path.read_text(), '')
            agent.reset(51)
            agent.choose_action(OBS)
            agent.choose_action(OBS)
            agent.reset(72)
            forced = {**OBS, 'legal_actions': [OBS['legal_actions'][0]]}
            agent.choose_action(forced)
            agent.choose_action(OBS)
            agent.reset(51)
            agent.choose_action(OBS)
            agent.close()
            agent.close()  # Repeated cleanup must be harmless.
            with self.assertRaises(RuntimeError):
                agent.reset(99)
            records = [json.loads(line) for line in path.read_text().splitlines()]

        resets = [r for r in records if r['event'] == 'reset']
        self.assertEqual([(r['episode_counter'], r['policy_seed']) for r in resets],
                         [(1, 51), (2, 72), (3, 51)])
        actions = [r for r in records if r['event'] != 'reset']
        self.assertEqual([(r['episode_counter'], r['decision']) for r in actions],
                         [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)])
        self.assertEqual(actions[2]['event'], 'forced_action')
        rng51, rng72 = random.Random(51), random.Random(72)
        first51 = rng51.randrange(2**31)
        self.assertEqual([seed for _, seed in client.calls],
                         [first51, rng51.randrange(2**31), rng72.randrange(2**31), first51])
        self.assertTrue(all(messages == expected_messages for messages, _ in client.calls))
        self.assertEqual(json.dumps(OBS, sort_keys=True), original_observation)
        self.assertTrue(all(r['schema'] == 'sts_llm_response_log_v1' for r in records))

    def test_invalid_action_is_model_failure_not_game_loss(self):
        class Client:
            def generate(self, messages, **kwargs):
                return SimpleNamespace(text='unfinished {"action_index":9}', reasoning_content=None,
                                       finish_reason='length', raw={}, elapsed_seconds=.1)
        agent = LLMAgent(Client())
        with self.assertRaises(InvalidModelAction):
            agent.choose_action(OBS)
        forced = {**OBS, 'legal_actions': [OBS['legal_actions'][0]]}
        self.assertEqual(agent.choose_action(forced), 3)

    def test_saved_metrics_keep_filtered_action_distinct(self):
        class Client:
            def generate(self, messages):
                return SimpleNamespace(text='{"action_index":12}', reasoning_content='',
                                       finish_reason='stop', raw={}, elapsed_seconds=.1)
        with tempfile.TemporaryDirectory() as tmp:
            summary = evaluate_saved([{**ROW, 'messages': prompt(OBS)}], Client(), output=str(Path(tmp)/'results.jsonl'))
        self.assertEqual(summary['legal_rate'], 1.)
        self.assertEqual(summary['mean_teacher_probability'], 0.)
        self.assertEqual(summary['teacher_filtered_action_rate'], 1.)



class SavedEvaluationCommandTests(unittest.TestCase):
    def run_eval(self, split_rows, *arguments):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / 'rows.jsonl'
            dataset.write_text(''.join(json.dumps(row) + '\n' for row in split_rows))
            argv = ['--dataset', str(dataset), '--base-url', 'http://127.0.0.1:1/v1', '--model', 'm',
                    '--output', str(Path(tmp) / 'out.jsonl'), *arguments]
            with patch('sts_llm.eval.OpenAIChatClient'), patch('sys.stdout'), \
                 patch('sts_llm.eval.evaluate_saved', return_value={}) as evaluate:
                eval_main(argv)
                summary = json.loads((Path(tmp) / 'out.jsonl.summary.json').read_text())
            return evaluate.call_args.args[0], summary

    @staticmethod
    def row(world_split_name, **extra):
        world = next(f'world:{n}' for n in range(1000) if world_split(f'world:{n}') == world_split_name)
        return {**ROW, 'id': world_split_name, 'split': world_split_name, 'world_identity': world,
                'messages': prompt(OBS), 'prompt_version': PROMPT_VERSION,
                'observation_version': 'sts_env_public_v3', **extra}

    def test_default_split_is_validation_and_test_needs_an_explicit_flag(self):
        rows = [self.row('train'), self.row('validation'), self.row('test')]
        selected, summary = self.run_eval(rows)
        self.assertEqual([row['id'] for row in selected], ['validation'])
        self.assertEqual((summary['split'], summary['split_scheme_version']), ('validation', SPLIT_SCHEME_VERSION))
        with self.assertRaises(SystemExit), patch('sys.stderr'):
            self.run_eval(rows, '--split', 'test')
        selected, _ = self.run_eval(rows, '--split', 'test', '--allow-test-split')
        self.assertEqual([row['id'] for row in selected], ['test'])

    def test_rows_with_forged_splits_are_refused(self):
        forged = self.row('test', split='validation')
        with self.assertRaisesRegex(ValueError, 'declares split'):
            self.run_eval([self.row('validation'), forged])


if __name__ == '__main__':
    unittest.main()
