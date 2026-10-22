"""Disk inventory and cleanup operate exclusively on disposable temporary trees."""
from contextlib import redirect_stdout
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from sts_rl import disk_report as dr


class DiskReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / 'data' / 'rl_runs'
        self.root.mkdir(parents=True)
        self.out = self.base / 'reports'
        self.out.mkdir()
        self.now = time.time()
        self.opened = patch.object(dr, 'open_paths', return_value=(set(), None)).start()
        self.addCleanup(patch.stopall)
        # Keep compression tests small; the production threshold gets its own check.
        patch.object(dr, 'MANIFEST_MIN_BYTES', 128).start()

    def write(self, path, content):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
        return path

    def fit(self, name, status='completed', best=False):
        self.write(f'{name}/.fresh-training-claim', '1')
        self.write(f'{name}/manifest.json', json.dumps({'rows': ['row'] * 300}, indent=2))
        if status:
            self.write(f'{name}/status.json', json.dumps({'status': status}))
        if best:
            self.write(f'{name}/best.pt', b'best')
        return self.root / name

    def old(self, seconds=3 * dr.DAY):
        for path in sorted(self.root.rglob('*'), reverse=True):
            if not path.is_symlink():
                os.utime(path, (self.now-seconds, self.now-seconds))
        os.utime(self.root, (self.now-seconds, self.now-seconds))

    def report(self):
        return dr.build_report([self.root], now=self.now, out=self.out)

    def candidates(self, report, rule):
        return [c for c in report['candidates'] if c['rule'] == rule]

    def tree(self):
        return {str(p.relative_to(self.root)): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in self.root.rglob('*') if p.is_file() and not p.is_symlink()}

    def test_categories_and_birth_time_growth_are_disjoint_sums(self):
        paths = ['manifest.json.gz', 'manifest.json', 'collector/worlds/1.json.gz', 'summary.json',
                 'eval_dev.json', 'iter_0002.pt', 'latest.pt', 'best.pt',
                 'records/launch-a/source.zip', 'record_objects/one', 'stdout.log', 'log.jsonl', 'misc.bin']
        born = {}
        expected_growth = {str(d): 0 for d in (1, 3, 7)}
        for index, relative in enumerate(paths, 1):
            path = self.write('joint/loop/fit_c00_seed1/' + relative, b'x' * index)
            age_days = [0.5, 2, 5, 8][index % 4]
            born[path.stat().st_ino] = self.now - age_days * dr.DAY
            for d in (1, 3, 7):
                if age_days < d:
                    expected_growth[str(d)] += index
        with patch.object(dr, 'birth_time', side_effect=lambda s: born[s.st_ino]):
            report = self.report()
        self.assertEqual(report['total_bytes'], sum(range(1, len(paths)+1)))
        self.assertEqual(sum(report['categories'].values()), report['total_bytes'])
        self.assertTrue(all(report['categories'][c] > 0 for c in dr.CATEGORIES))
        self.assertEqual(report['categories']['launch_records'], 19)
        self.assertEqual(report['growth_bytes'], expected_growth)
        self.assertEqual(len(report['runs']), 1)
        self.assertEqual(report['runs'][0]['run'], 'joint/loop/fit_c00_seed1')
        self.assertEqual(report['runs'][0]['growth_bytes'], expected_growth)
        with patch.object(dr, 'birth_time', return_value=None):
            unknown = self.report()
        self.assertEqual(unknown['growth_bytes'], {'1': 0, '3': 0, '7': 0})
        self.assertEqual(unknown['birth_time_unavailable_bytes'], unknown['total_bytes'])

    def test_compression_selection_size_status_idle_and_ambiguous_forms(self):
        for name, status in [('done', 'finished'), ('stop', 'interrupted'), ('idle', None),
                             ('failed_recently', 'failed'), ('ambiguous', 'completed')]:
            self.fit(name, status)
        self.write('ambiguous/manifest.json.gz', b'existing')
        small = self.fit('small')
        (small/'manifest.json').write_bytes(b'{}')
        self.old()
        os.utime(self.root/'failed_recently/manifest.json', (self.now-3*3600, self.now-3*3600))
        report = self.report()
        selected = {Path(c['directory']).name: c for c in self.candidates(report, dr.RULES[0])}
        self.assertEqual(set(selected), {'done', 'stop', 'idle', 'ambiguous'})
        self.assertIn('both', selected['ambiguous']['blocked_reason'])
        self.assertTrue(all(selected[n]['blocked_reason'] is None for n in ('done', 'stop', 'idle')))
        self.assertTrue(selected['done']['savings_estimated'])
        self.assertGreater(selected['done']['bytes_would_free'], 0)
        with patch.object(dr, 'MANIFEST_MIN_BYTES', 50_000_000):
            self.assertEqual(self.candidates(self.report(), dr.RULES[0]), [])

    def test_interrupted_selection_preserves_best_and_nonfit_outputs(self):
        for name, status, best in [('a', 'interrupted', False), ('b', 'failed', False),
                                  ('keep', 'interrupted', True), ('done', 'completed', False)]:
            self.fit(name, status, best)
        self.write('collector/status.json', '{"status":"failed"}')
        self.write('fit_failed_before_manifest/status.json', '{"status":"failed"}')
        self.write('fit_parent/status.json', '{"status":"failed"}')
        self.write('fit_parent/child/best.pt', b'preserve completed outputs')
        self.old()
        selected = self.candidates(self.report(), dr.RULES[1])
        self.assertEqual({Path(c['path']).name for c in selected}, {'a', 'b', 'fit_failed_before_manifest'})
        self.assertTrue(all(c['bytes_would_free'] == c['bytes'] for c in selected))

    def test_duplicate_selection_requires_finished_and_identical_final_iteration(self):
        for name in ('same', 'different', 'notlast', 'running', 'sizediff', 'logdone'):
            self.write(f'{name}/latest.pt', b'saved')
            self.write(f'{name}/iter_0002.pt', b'saved')
            self.write(f'{name}/iter_0010.pt', b'other' if name == 'notlast' else b'saved')
            if name == 'logdone':
                self.write(f'{name}/log.jsonl', '{"event":"done"}\n')
            else:
                self.write(f'{name}/status.json', json.dumps({'status': 'running' if name == 'running' else 'completed'}))
        self.write('different/latest.pt', b'wrong')
        self.write('sizediff/latest.pt', b'shorter file')
        self.old()
        with patch.object(dr, '_sha', wraps=dr._sha) as sha:
            selected = self.candidates(self.report(), dr.RULES[2])
        self.assertEqual({Path(c['directory']).name for c in selected}, {'same', 'logdone'})
        self.assertTrue(all(Path(c['duplicate_of']).name == 'iter_0010.pt' for c in selected))
        self.assertFalse(any('sizediff' in str(call) for call in sha.call_args_list))

    def test_cli_dry_run_writes_timestamped_reports_and_changes_no_data(self):
        self.fit('fit')
        self.old()
        before = self.tree()
        script = Path(__file__).resolve().parents[2] / 'scripts/rl/disk_report.py'
        spec = importlib.util.spec_from_file_location('disk_report_cli', script)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with redirect_stdout(io.StringIO()):
            report = cli.main([str(self.root), '--out', str(self.out)])
        self.assertEqual(self.tree(), before)
        self.assertEqual(report['applied_rules'], [])
        self.assertEqual(len(list(self.out.glob('disk_report_*.json'))), 1)
        self.assertEqual(len(list(self.out.glob('disk_report_*.txt'))), 1)
        self.assertFalse((self.out/'cleanup_log.tsv').exists())

    def test_apply_only_selected_rule_lossless_compression_and_append_log(self):
        compressed = self.fit('compress')
        interrupted = self.fit('interrupted', 'interrupted')
        latest = self.write('trainer/latest.pt', b'weights')
        iteration = self.write('trainer/iter_0009.pt', b'weights')
        self.write('trainer/status.json', '{"status":"completed"}')
        self.old()
        original = (compressed/'manifest.json').read_bytes()
        actions = dr.apply_rules(self.report(), [dr.RULES[0]], self.out)
        self.assertEqual(len(actions), 2)
        self.assertTrue(interrupted.exists())
        self.assertTrue(latest.exists())
        self.assertTrue(iteration.exists())
        self.assertFalse((compressed/'manifest.json').exists())
        with gzip.open(compressed/'manifest.json.gz', 'rb') as stream:
            self.assertEqual(stream.read(), original)
        first_log = (self.out/'cleanup_log.tsv').read_text()
        self.assertEqual(len(first_log.splitlines()), 3)
        self.assertTrue(all(line.split('\t')[1] == dr.RULES[0] for line in first_log.splitlines()[1:]))
        self.old()
        dr.apply_rules(self.report(), [dr.RULES[2]], self.out)
        self.assertFalse(latest.exists())
        self.assertTrue(iteration.exists())
        self.assertTrue((self.out/'cleanup_log.tsv').read_text().startswith(first_log))
        self.old()
        dr.apply_rules(self.report(), [dr.RULES[1]], self.out)
        self.assertFalse(interrupted.exists())
        self.assertTrue(compressed.exists())
        lines = (self.out/'cleanup_log.tsv').read_text().splitlines()
        self.assertEqual(len(lines), 5)
        self.assertEqual(lines[-1].split('\t')[1], dr.RULES[1])

    def test_two_hour_window_checks_files_and_parent_directories_and_rechecks_apply(self):
        fit = self.fit('group/fit', 'failed')
        self.old()
        report = self.report()
        self.assertIsNone(self.candidates(report, dr.RULES[1])[0]['blocked_reason'])
        # Editing a file does not change its parent directory's mtime.
        os.utime(fit/'status.json', (self.now-3600, self.now-3600))
        blocked = self.candidates(self.report(), dr.RULES[1])[0]
        self.assertIn('2 hours', blocked['blocked_reason'])
        dr.apply_rules(report, [dr.RULES[1]], self.out)
        self.assertTrue(fit.exists())
        self.old()
        os.utime(fit.parent, (self.now, self.now))
        self.assertIn('2 hours', self.candidates(self.report(), dr.RULES[1])[0]['blocked_reason'])

    def test_queue_open_files_and_unavailable_inspection_block_cleanup(self):
        fit = self.fit('group/fit', 'failed')
        self.write('queue.json', json.dumps({'runs': [dict(name='different', status='pending',
            cmd=['python', 'job.py', '--out', str(fit.parent)])]}))
        self.old()
        self.assertIn('pending', self.candidates(self.report(), dr.RULES[1])[0]['blocked_reason'])
        (self.root/'queue.json').unlink()
        self.opened.return_value = ({fit/'manifest.json'}, None)
        self.assertIn('open', self.candidates(self.report(), dr.RULES[1])[0]['blocked_reason'])
        self.opened.return_value = (set(), 'inspection unavailable')
        self.assertIn('unavailable', self.candidates(self.report(), dr.RULES[1])[0]['blocked_reason'])

    def test_pending_input_and_activity_during_compression_are_protected(self):
        fit = self.fit('fit', 'failed')
        self.write('queue.json', json.dumps({'runs': [dict(name='future', status='pending',
            cmd=['python', 'job.py', '--manifest', str(fit/'manifest.json')])]}))
        self.old()
        self.assertIn('pending', self.candidates(self.report(), dr.RULES[1])[0]['blocked_reason'])
        (self.root/'queue.json').unlink()
        self.old()
        report = self.report()
        original_copy = dr.shutil.copyfileobj
        def start_reader(source, target, **kwargs):
            original_copy(source, target, **kwargs)
            self.opened.return_value = ({fit/'manifest.json'}, None)
        with patch.object(dr.shutil, 'copyfileobj', side_effect=start_reader):
            actions = dr.apply_rules(report, [dr.RULES[0]], self.out)
        self.assertIn('skipped', actions[0]['status'])
        self.assertTrue((fit/'manifest.json').exists())
        self.assertFalse((fit/'manifest.json.gz').exists())
        self.assertFalse(any(fit.glob('.manifest-gzip-*')))

    def test_symlink_escape_and_hardlinks_are_never_changed(self):
        outside = self.base/'outside'
        outside.mkdir()
        sentinel = outside/'keep'
        sentinel.write_text('keep')
        fit = self.fit('fit', 'failed')
        (fit/'escape').symlink_to(outside, target_is_directory=True)
        (self.root/'linked-run').symlink_to(outside, target_is_directory=True)
        self.old()
        report = self.report()
        self.assertEqual(len(report['skipped_symlinks']), 2)
        self.assertIn('symlink', self.candidates(report, dr.RULES[1])[0]['blocked_reason'])
        dr.apply_rules(report, dr.RULES, self.out)
        self.assertTrue(fit.exists())
        self.assertEqual(sentinel.read_text(), 'keep')
        (fit/'escape').unlink()
        os.link(fit/'manifest.json', self.base/'preserved-hardlink')
        self.old()
        self.assertIn('hard-linked', self.candidates(self.report(), dr.RULES[1])[0]['blocked_reason'])

    def test_multiple_roots_and_low_free_space(self):
        self.write('a/file', b'123')
        second = self.base/'second'
        second.mkdir()
        (second/'file').write_bytes(b'45')
        with patch.object(dr.shutil, 'disk_usage', return_value=type('Usage', (), {'free': 10 * dr.GIB})()):
            report = dr.build_report([self.root, second], min_free_gib=30, now=self.now)
        self.assertEqual(report['total_bytes'], 5)
        self.assertTrue(all(v['below_minimum'] for v in report['volumes']))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            dr.build_report([self.root, self.root/'a'])


if __name__ == '__main__':
    unittest.main()
