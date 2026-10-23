import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from sts_llm.teacher_worker import source_revision
from sts_llm.observations import prompt
from sts_llm.generation import parse_action, K2_REASONING_CLOSERS
from sts_llm.teacher import teacher_reward, validate_teacher
from sts_llm.dataset import world_split, prepare_rows
from sts_llm.eval import LLMAgent, InvalidModelAction, evaluate_saved

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


if __name__ == '__main__':
    unittest.main()
