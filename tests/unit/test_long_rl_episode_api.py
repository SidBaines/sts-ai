from __future__ import annotations

import hashlib
import gzip
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.unit.test_long_rl_dashboard import dashboard


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def task_id(row): return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class EpisodeAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.study = Path(self.temp.name); self.campaign = self.study / 'campaign'
        for name in ('tasks', 'checkpoints', 'metrics'): (self.campaign / name).mkdir(parents=True, exist_ok=True)
        self.task_path = self.campaign / 'tasks/act2_continuations_00000002.jsonl'
        self.metas = []; raw_rows = []
        for seed, source, floor in ((11, 'native_teacher', 18), (12, 'learner', 17), (13, 'fresh_scratch', 0)):
            raw = {'seed': seed, 'kind': 'run', 'prefix': [], 'max_act': 2, 'horizon': 4096, 'split': 'train'}
            identity = task_id(raw)
            meta = {'task_id': identity, 'world_seed': seed, 'component': 'act2' if floor else 'scratch',
                    'start_source': source, 'start_act': 2 if floor else 1, 'start_floor': floor,
                    'goal_act': 2, 'horizon': 4096}
            self.metas.append(meta); raw_rows.append(dict(raw, long_metadata=meta))
        self.task_path.write_text(''.join(json.dumps(row) + '\n' for row in raw_rows))
        self.manifest_path = Path(str(self.task_path) + '.manifest.json')
        self.manifest_path.write_text(json.dumps({'round_index': 2, 'task_file_sha256': sha(self.task_path), 'tasks': self.metas}))
        a, b, c = [m['task_id'] for m in self.metas]
        self.metric_path = self.campaign / 'checkpoints/chunk.metrics.json'
        self.metric = {'config': {'tasks': str(self.task_path), 'reward_mode': 'progress'}, 'rounds': [{
            'task_ids': [a, b, c], 'unstarted_tasks': 1, 'episodes': [
                {'task_id': a, 'outcome': 'defeat', 'terminated': True, 'floor': 22, 'reward': -.8, 'hp_cost': -12, 'steps': 9}],
            'faults': [{'task_id': b, 'error': 'SimulatorError'}]}]}
        self.metric_path.write_text(json.dumps(self.metric))
        self.job = {'kind': 'train', 'status': 'complete', 'started': 1, 'finished': 2, 'session_kind': 'main',
                    'metrics': str(self.metric_path), 'command': ['train', '--tasks', str(self.task_path)],
                    'signature': {'condition': {'digest': 'condition', 'inputs': {'--tasks': {'path': str(self.task_path), 'sha256': sha(self.task_path)}}}},
                    'optimizer_goal': {'target': {'stage': 'act2_continuations'}}, 'artifacts': {str(self.metric_path): sha(self.metric_path)}}
        self.journal = {'status': 'running', 'jobs': [self.job]}
        self.write_journal(); self.store = dashboard.TelemetryStore(self.campaign)

    def write_journal(self):
        for name, value in (('journal.json', self.journal), ('progress.json', {}), ('config.json', {})):
            (self.campaign / name).write_text(json.dumps(value))

    def test_join_preserves_negative_hp_cost_and_synthetic_requests(self):
        response = self.store.episodes({})
        self.assertEqual(response['total'], 3)
        rows = response['episodes']; normal, fault, missing = rows
        self.assertEqual(normal['round'], 2); self.assertEqual(normal['start_floor'], 18)
        self.assertEqual(normal['hp_cost'], -12); self.assertEqual(normal['floor_progress'], 4)
        self.assertIsNone(normal.get('end_hp')); self.assertFalse(normal['coverage_valid'])
        self.assertFalse(normal['numeric_eligible'])  # Incomplete collector cohort is explicit.
        self.assertTrue(fault['fault']); self.assertIsNone(fault['reward']); self.assertIsNone(fault['floor'])
        self.assertEqual(missing['outcome'], 'unstarted'); self.assertTrue(missing['requested_occurrence'])
        self.assertIsNone(missing['hp_cost']); self.assertEqual(response['coverage']['metadata_joined_rows'], 3)
        self.assertEqual(response['availability']['scoped_requested_rows'], 3)
        self.assertIn('end_hp', response['availability']['unavailable'])

    def test_filters_floor_horizon_sources_and_paginated_scope(self):
        first = self.store.episodes({'limit': ['1']})
        self.assertEqual(first['total'], 3); self.assertTrue(first['has_more']); self.assertEqual(first['next_offset'], 1)
        second = self.store.episodes({'limit': ['1'], 'offset': ['1']})
        self.assertNotEqual(first['episodes'][0]['task_id'], second['episodes'][0]['task_id'])
        filtered = self.store.episodes({'source': ['native_teacher'], 'start_floor_min': ['18'], 'start_floor_max': ['18'], 'horizon': ['4096']})
        self.assertEqual(filtered['total'], 1)
        self.assertEqual(set(filtered['availability']['filter_options']['start_source']), {'native_teacher', 'learner', 'fresh_scratch'})
        self.assertEqual(self.store.episodes({'start_floor_min': ['19']})['total'], 0)
        for query in ({'limit': ['20001']}, {'max_jobs': ['1001']}, {'offset': ['-1']}, {'kind': ['final']},
                      {'path': ['/secret']}, {'round_min': ['3'], 'round_max': ['2']},
                      {'start_floor_min': ['20'], 'start_floor_max': ['18']}):
            with self.assertRaises(ValueError): self.store.episodes(query)

    def test_metadata_cache_invalidates_on_manifest_change_without_metric_change(self):
        rows = self.store.episodes({})['episodes']; self.assertTrue(rows[0]['metadata_joined'])
        changed = json.loads(self.manifest_path.read_text()); changed['tasks'][0]['start_floor'] = 999
        self.manifest_path.write_text(json.dumps(changed))
        result = self.store.episodes({})
        self.assertEqual(result['coverage']['unjoined_rows'], 3)
        self.assertIsNone(result['episodes'][0]['start_floor'])
        self.assertEqual(result['coverage']['missing_task_metadata'], 1)
        self.assertTrue(result['errors'])

    def test_manifest_symlink_is_rejected_before_outside_read(self):
        outside = self.study / 'private.json'; outside.write_text(self.manifest_path.read_text())
        self.manifest_path.unlink(); self.manifest_path.symlink_to(outside)
        original = Path.read_text
        def checked(path, *args, **kwargs):
            if path == self.manifest_path or path == outside: raise AssertionError('Escaped manifest must not be read')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', checked):
            response = self.store.episodes({})
        self.assertEqual(response['coverage']['unjoined_rows'], 3)

    def test_duplicate_and_unexpected_observations_cannot_fake_denominator(self):
        a, b, c = [m['task_id'] for m in self.metas]
        metrics = {'rounds': [{'task_ids': [a, b], 'unstarted_tasks': 0, 'episodes': [
            {'task_id': a, 'outcome': 'act_clear', 'terminated': True, 'reward': 1},
            {'task_id': a, 'outcome': 'act_clear', 'terminated': True, 'reward': 1},
            {'task_id': c, 'outcome': 'act_clear', 'terminated': True, 'reward': 1}], 'faults': []}]}
        rows = dashboard.compact_episodes(metrics, self.job, 0, {m['task_id']: m for m in self.metas})
        self.assertEqual(sum(row['requested_occurrence'] for row in rows), 2)
        self.assertEqual(sum(not row['requested_occurrence'] for row in rows), 2)
        self.assertTrue(all(not row['numeric_eligible'] for row in rows))
        self.assertEqual(rows[-1]['outcome'], 'missing'); self.assertFalse(rows[-1]['unstarted'])

    def make_full3(self):
        task_path = self.study / 'dev_act3.jsonl'
        raw = [{'seed': 200000 + i, 'prefix': [], 'split': 'dev', 'kind': 'run', 'max_act': 3, 'horizon': 4096} for i in range(48)]
        ids = [task_id(row) for row in raw]; task_path.write_text(''.join(json.dumps(r) + '\n' for r in raw))
        cp = 'a' * 64; module = 'b' * 64
        condition = {'digest': 'full3', 'inputs': {'--tasks': {'path': str(task_path), 'sha256': sha(task_path)},
                                               '--checkpoint': {'path': '/snapshot.pt', 'sha256': cp}},
                     'arguments': ['evaluate', '--tasks', str(task_path)]}
        metric = {'config': {'checkpoint': '/snapshot.pt', 'tasks': str(task_path), 'seed': 777, 'max_act': 3,
                            'horizon': 4096, 'split': 'dev', 'simulations': 200},
                  'training_environment': {'native_module_sha256': module},
                  'rounds': [{'task_ids': ids, 'unstarted_tasks': 0, 'episodes': [
                      {'task_id': identity, 'outcome': 'defeat', 'terminated': True, 'floor': 20, 'hp_cost': 80} for identity in ids], 'faults': []}]}
        directory = self.study / 'full_run_eval'; (directory / 'jobs').mkdir(parents=True)
        metric_path = directory / 'jobs' / (('c' * 64) + '_a0.metrics.json'); metric_path.write_text(json.dumps(metric))
        record = {'roles': ['selected_best'], 'checkpoint_sha256': cp, 'task_ids': ids, 'captured_global_round': 7,
                  'metrics_path': str(metric_path), 'metrics_sha256': sha(metric_path), 'condition_signature': condition,
                  'complete': True, 'captured_at': 10, 'evaluated_at': 20}
        baseline_path = self.study / 'anchor_dev_act3.metrics.json'; baseline_path.write_text(json.dumps(metric))
        baseline = dict(record, roles=['initial_anchor'], metrics_path=str(baseline_path), metrics_sha256=sha(baseline_path), captured_at=0)
        observer = {'monitoring_only': True, 'version': 1, 'panel': 'dev_act3', 'histories': [record], 'baseline': baseline}
        self.journal.update(identity={'initial_checkpoint_sha256': cp, 'backend': {'provenance': {'module_sha256': module}}},
                            goal_leaders={'dev_act3': {'quality': {'cohort': ids}}})
        self.write_journal(); telemetry_path = directory / 'telemetry.json'; telemetry_path.write_text(json.dumps(observer))
        self.store = dashboard.TelemetryStore(self.campaign)
        return observer, telemetry_path, metric_path

    def test_full3_baseline_fixed_path_provenance_and_role_filter(self):
        observer, path, metrics = self.make_full3()
        response = self.store.episodes({'kind': ['full3']})
        self.assertEqual(response['total'], 96); self.assertEqual(response['coverage']['metadata_joined_rows'], 96)
        baseline = self.store.episodes({'kind': ['full3'], 'role': ['initial_anchor']})
        self.assertEqual(baseline['total'], 48); self.assertTrue(all(row['round'] == 0 for row in baseline['episodes']))
        outside = self.study / 'other.metrics.json'; outside.write_text(metrics.read_text())
        observer['baseline']['metrics_path'] = str(outside); path.write_text(json.dumps(observer))
        self.store.last_refresh = -float('inf')
        self.assertEqual(self.store.episodes({'kind': ['full3'], 'role': ['initial_anchor']})['total'], 0)

    def test_full3_foreign_cohort_checkpoint_or_producer_rejected(self):
        observer, path, metrics = self.make_full3()
        observer['histories'][0]['checkpoint_sha256'] = 'f' * 64
        observer.pop('baseline'); path.write_text(json.dumps(observer))
        response = self.store.episodes({'kind': ['full3']})
        self.assertEqual(response['total'], 0); self.assertEqual(response['coverage']['skipped_invalid_panels'], 1)
        observer['histories'][0]['checkpoint_sha256'] = 'a' * 64
        observer['histories'][0]['task_ids'][0] = 'foreign'; path.write_text(json.dumps(observer))
        self.store.last_refresh = -float('inf')
        self.assertEqual(self.store.episodes({'kind': ['full3']})['total'], 0)
        observer['monitoring_only'] = False; path.write_text(json.dumps(observer)); self.store.last_refresh = -float('inf')
        self.assertEqual(self.store.episodes({'kind': ['full3']})['total'], 0)

    def test_warm_query_reuses_compact_rows_and_bounded_job_window(self):
        first = self.store.episodes({})
        with patch.object(dashboard, 'content_sha', side_effect=AssertionError('No reread unchanged bytes')):
            again = self.store.episodes({})
        self.assertEqual(first['episodes'], again['episodes'])
        self.journal['jobs'].append(dict(self.job, started=3, finished=4))
        self.write_journal(); self.store.last_refresh = -float('inf')
        bounded = self.store.episodes({'max_jobs': ['1']})
        self.assertEqual(bounded['coverage']['omitted_jobs'], 1); self.assertEqual(bounded['total'], 3)

    def test_http_gzip_roundtrip_and_qzero_respected(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), dashboard.handler_for(self.store))
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        url = f'http://127.0.0.1:{server.server_port}/api/episodes?limit=1'
        with urlopen(Request(url, headers={'Accept-Encoding': 'br, gzip'})) as response:
            self.assertEqual(response.headers['Content-Encoding'], 'gzip')
            self.assertEqual(response.headers['Vary'], 'Accept-Encoding')
            data = json.loads(gzip.decompress(response.read()))
            self.assertEqual(data['total'], 3); self.assertEqual(len(data['episodes']), 1)
        with urlopen(Request(url, headers={'Accept-Encoding': 'gzip;q=0'})) as response:
            self.assertIsNone(response.headers.get('Content-Encoding'))
            self.assertEqual(json.loads(response.read())['total'], 3)
        with self.assertRaises(HTTPError) as error: urlopen(url + '&limit=20001')
        self.assertEqual(error.exception.code, 400)

    def test_gzip_negotiation_does_not_accept_refused_or_invalid_encodings(self):
        self.assertTrue(dashboard.accepts_gzip('GZip; q=0.5'))
        for header in ('br', 'gzip;q=0', 'gzip;q=invalid', 'gzip;q=nan', 'xgzip', None):
            self.assertFalse(dashboard.accepts_gzip(header))

    def test_dataset_identity_ignores_page_but_detects_mutable_inputs(self):
        first = self.store.episodes({'limit': ['1']})
        second = self.store.episodes({'limit': ['2'], 'offset': ['1']})
        self.assertEqual(first['dataset_id'], second['dataset_id'])
        self.metric['rounds'][0]['episodes'][0]['floor'] = 23
        self.metric_path.write_text(json.dumps(self.metric))
        third = self.store.episodes({'limit': ['1']})
        self.assertNotEqual(first['dataset_id'], third['dataset_id'])
        self.assertNotEqual(third['dataset_id'], self.store.episodes({'goal_act': ['2']})['dataset_id'])


if __name__ == '__main__': unittest.main()
