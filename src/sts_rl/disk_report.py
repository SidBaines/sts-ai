"""Bounded disk inventory and explicit, conservative cleanup of local RL outputs."""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import time

GIB = 1024**3
DAY = 86400
SAFETY_SECONDS = 2 * 3600
MANIFEST_MIN_BYTES = 50_000_000
RULES = ('compress-fit-manifests', 'interrupted-fit-outputs', 'trainer-latest-duplicates')
CATEGORIES = ('fit_manifests_compressed', 'fit_manifests_uncompressed', 'collector_world_shards',
              'eval_summaries', 'trainer_checkpoints', 'other_pt', 'launch_records', 'logs', 'other')
ARTIFACT_DIRS = {'worlds', 'quarantine', 'records', 'record_objects', 'logs'}
FINISHED = {'finished', 'completed', 'done'}


def category(path):
    if path.name == 'manifest.json.gz':
        return 'fit_manifests_compressed'
    if path.name == 'manifest.json':
        return 'fit_manifests_uncompressed'
    if path.parent.name == 'worlds' and path.name.endswith('.json.gz'):
        return 'collector_world_shards'
    if 'record_objects' in path.parts or any(
            a == 'records' and b.startswith('launch-') for a, b in zip(path.parts, path.parts[1:])):
        return 'launch_records'
    if path.name == 'summary.json' or (path.name.startswith('eval_') and path.suffix == '.json'):
        return 'eval_summaries'
    if path.name == 'latest.pt' or re.fullmatch(r'iter_\d+\.pt', path.name):
        return 'trainer_checkpoints'
    if path.suffix == '.pt':
        return 'other_pt'
    if path.suffix in {'.log', '.jsonl'} or 'logs' in path.parts:
        return 'logs'
    return 'other'


def birth_time(info):
    # ctime is metadata-change time on Unix, not creation time.
    return getattr(info, 'st_birthtime', None)


def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _within(path, parent):
    return path == parent or parent in path.parents


def _plain(path):
    """No component may be a symlink, including a parent swapped since inventory."""
    return not any(p.is_symlink() for p in (path, *path.parents))


def _small_json(path):
    if not _plain(path) or not path.is_file() or path.stat().st_size > 16 * 1024**2:
        return {}
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _status(directory):
    for name in ('status.json', 'summary.json'):
        value = _small_json(directory / name)
        if isinstance(value.get('status'), str):
            return value['status'].lower()
    # PPO trainers finish with an event in log.jsonl, rather than a status file.
    path = directory / 'log.jsonl'
    if _plain(path) and path.is_file():
        try:
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 65536))
                lines = stream.read().splitlines()
            last = json.loads(lines[-1]) if lines else {}
            if last.get('event') == 'done':
                return 'completed'
        except (OSError, ValueError, AttributeError):
            pass
    return None


def _is_fit(directory, files):
    names = {p.name for p in files if p.parent == directory}
    if ('.fresh-training-claim' in names or 'best.pt' in names
            or directory.name.startswith('fit')):
        return True
    if not names.intersection({'manifest.json', 'manifest.json.gz'}):
        return False
    config = _small_json(directory / 'config.json')
    receipt = _small_json(directory / 'receipt.json')
    config = receipt.get('config', config)
    return (directory.name.startswith('fit') or receipt.get('stage') == 'supervised_teacher_warmup'
            or config.get('architecture') in {'old', 'new', 'controlled'})


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _compression_saving(path, size):
    """Estimate from <=3 MiB at the start, middle and end; never parse huge JSON."""
    samples = []
    with path.open('rb') as stream:
        for offset in sorted({0, max(0, size // 2 - 524288), max(0, size - 1048576)}):
            stream.seek(offset)
            samples.append(stream.read(1048576))
    ratio = sum(len(gzip.compress(x, compresslevel=1, mtime=0)) for x in samples) / sum(map(len, samples))
    return max(0, size - round(size * ratio))


def _queue_runs(roots):
    active = []
    for root in roots:
        # Accept a data/rl_runs root or a single run under that standard layout.
        queue_root = next((p for p in (root, *root.parents) if p.name == 'rl_runs'), root)
        queue = queue_root / 'queue.json'
        if queue.exists() and not _small_json(queue):
            # A malformed/unreadable queue must not silently permit cleanup.
            active.append((root, 'queue cannot be read'))
        for job in _small_json(queue).get('runs', []):
            if job.get('status') not in {'pending', 'running', 'paused'}:
                continue
            targets = [queue_root / job.get('name', '')]
            cmd = job.get('cmd', [])
            for i, arg in enumerate(cmd):
                value = cmd[i+1] if arg == '--out' and i+1 < len(cmd) else (
                    arg.split('=', 1)[1] if arg.startswith('--out=') else None)
                if value:
                    target = Path(value)
                    targets.append(target if target.is_absolute() else queue_root.parent.parent / target)
                # Pending jobs can reference an old fit as input before opening it.
                if isinstance(arg, str) and not arg.startswith('-'):
                    target = Path(arg)
                    target = target if target.is_absolute() else queue_root.parent.parent / target
                    if target.exists():
                        targets.append(target)
            active.extend((p.resolve(), 'queue job ' + str(job.get('name')) + ' is ' + job['status'])
                          for p in targets)
    return active


def open_paths():
    """One read-only process snapshot; uncertainty fails closed for cleanup."""
    executable = shutil.which('lsof')
    if executable:
        try:
            result = subprocess.run([executable, '-nP', '-Fn'], capture_output=True, text=True, timeout=20)
            if result.returncode not in (0, 1) or not result.stdout:
                return set(), 'open-file inspection failed'
            paths = {Path(line[1:].removesuffix(' (deleted)')).resolve()
                     for line in result.stdout.splitlines() if line.startswith('n/')}
            return paths, None
        except (OSError, subprocess.TimeoutExpired):
            return set(), 'open-file inspection failed'
    if Path('/proc/self/fd').is_dir():
        paths = set()
        try:
            for process in Path('/proc').iterdir():
                if not process.name.isdigit():
                    continue
                try:
                    links = [process / 'cwd', *(process / 'fd').iterdir()]
                    for link in links:
                        try:
                            target = os.readlink(link)
                            if target.startswith('/'):
                                paths.add(Path(target.removesuffix(' (deleted)')).resolve())
                        except FileNotFoundError:
                            pass  # A descriptor or process closed during the snapshot.
                except FileNotFoundError:
                    pass
            return paths, None
        except OSError:
            return set(), 'open-file inspection incomplete'
    return set(), 'open-file inspection unavailable'


def _walk(root, excluded=()):
    files, directories, links, errors = {}, {}, [], []
    def error(exc):
        errors.append(str(exc))
    for current, children, names in os.walk(root, followlinks=False, onerror=error):
        directory = Path(current)
        if not _plain(directory):
            children[:] = []
            links.append(directory)
            continue
        directories[directory] = directory.stat()
        for name in children[:]:
            path = directory / name
            if path.is_symlink() or any(_within(path, p) for p in excluded):
                children.remove(name)
                if path.is_symlink():
                    links.append(path)
        for name in names:
            path = directory / name
            if any(_within(path, p) for p in excluded):
                continue
            try:
                info = path.lstat()
                if stat.S_ISREG(info.st_mode):
                    files[path] = info
                else:
                    links.append(path)
            except OSError as exc:
                error(exc)
    return files, directories, links, errors


def _safety(candidate, roots, files, directories, links, active, opened, open_error, now):
    path, directory = Path(candidate['path']), Path(candidate['directory'])
    root = next((r for r in roots if _within(path, r)), None)
    if root is None or not _plain(path):
        return 'outside roots or symlink'
    if path == root and candidate['rule'] == 'interrupted-fit-outputs':
        return 'will not delete a supplied root'
    for target, reason in active:
        if _within(directory, target) or _within(target, directory):
            return reason
    parents = [p for p in (directory, *directory.parents) if _within(p, root) and p != root]
    activity = [info.st_mtime for p, info in {**directories, **files}.items() if _within(p, directory)]
    activity.extend(directories[p].st_mtime for p in parents if p in directories)
    if activity and max(activity) > now - SAFETY_SECONDS:
        return 'run directory or contents modified in the last 2 hours'
    if any(_within(p, directory) for p in links):
        return 'run contains symlinks or special files'
    if open_error:
        return open_error
    if any(_within(p, directory) for p in opened):
        return 'running process has a path open in this run'
    touched = [p for p in files if _within(p, path)]
    if any(files[p].st_nlink > 1 for p in touched):
        return 'hard-linked output; reclaimable bytes are uncertain'
    return None


def build_report(roots, *, min_free_gib=30, now=None, out=None):
    now = time.time() if now is None else now
    roots = list(dict.fromkeys(Path(p).expanduser().resolve() for p in roots))
    if not roots or any(not p.is_dir() for p in roots):
        raise ValueError('Every data root must be an existing directory')
    if any(a != b and _within(a, b) for a in roots for b in roots):
        raise ValueError('Data roots must not overlap')
    excluded = [Path(out).resolve()] if out else []
    opened, open_error = open_paths()
    active = _queue_runs(roots)
    report = dict(time=datetime.fromtimestamp(now, timezone.utc).isoformat(), roots=list(map(str, roots)),
                  min_free_gib=min_free_gib, volumes=[], runs=[], candidates=[], skipped_symlinks=[], warnings=[],
                  total_bytes=0, categories={k: 0 for k in CATEGORIES}, growth_bytes={str(d): 0 for d in (1, 3, 7)},
                  birth_time_unavailable_bytes=0, search_details_share=None,
                  search_details_note='Skipped: search details are embedded in compressed world JSON; no cheap byte attribution.',
                  size_note='Logical file bytes; hard links counted per path. Compression savings are sampled estimates.')
    if open_error:
        report['warnings'].append(open_error + '; cleanup is disabled')
    for root in roots:
        files, directories, links, errors = _walk(root, excluded)
        report['warnings'].extend(errors)
        report['skipped_symlinks'].extend(map(str, links))
        usage = shutil.disk_usage(root)
        report['volumes'].append(dict(root=str(root), device=root.stat().st_dev, free_bytes=usage.free,
                                      below_minimum=usage.free < min_free_gib * GIB))
        groups = {}
        for path, info in files.items():
            parts = path.relative_to(root).parts[:-1]
            parts = parts[:next((i for i, p in enumerate(parts) if p in ARTIFACT_DIRS), len(parts))]
            name = str(Path(*parts[:3])) if parts else '.'
            row = groups.setdefault(name, dict(root=str(root), run=name, total_bytes=0,
                categories={k: 0 for k in CATEGORIES}, growth_bytes={str(d): 0 for d in (1, 3, 7)},
                birth_time_unavailable_bytes=0))
            kind = category(path.relative_to(root))
            row['total_bytes'] += info.st_size
            row['categories'][kind] += info.st_size
            born = birth_time(info)
            if born is None:
                row['birth_time_unavailable_bytes'] += info.st_size
            else:
                for days in (1, 3, 7):
                    if now - days * DAY <= born <= now:
                        row['growth_bytes'][str(days)] += info.st_size
        report['runs'].extend(groups.values())
        # No giant manifest is parsed, and hashing happens only for equal-sized final checkpoints.
        by_directory = {}
        for path in files:
            by_directory.setdefault(path.parent, []).append(path)
        for directory, children in by_directory.items():
            fit = _is_fit(directory, children)
            if not fit and directory / 'latest.pt' not in files:
                continue
            status = _status(directory)
            candidates = []
            manifest = directory / 'manifest.json'
            activity = max(info.st_mtime for p, info in {**files, **directories}.items() if _within(p, directory))
            if (fit and manifest in files and files[manifest].st_size > MANIFEST_MIN_BYTES
                    and (status in FINISHED | {'interrupted'} or activity <= now - DAY)):
                candidates.append(dict(rule=RULES[0], path=str(manifest), directory=str(directory),
                    bytes=files[manifest].st_size, bytes_would_free=None, savings_estimated=True,
                    reason='Large plain fit manifest; fit is finished/interrupted or inactive for 24 hours.'))
            has_best = any(p.name == 'best.pt' and _within(p, directory) for p in files)
            if fit and status in {'interrupted', 'failed'} and not has_best:
                size = sum(s.st_size for p, s in files.items() if _within(p, directory))
                candidates.append(dict(rule=RULES[1], path=str(directory), directory=str(directory), bytes=size,
                    bytes_would_free=size, savings_estimated=False, reason=f'Fit status is {status}; no best.pt.'))
            latest = directory / 'latest.pt'
            iterations = [p for p in children if re.fullmatch(r'iter_\d+\.pt', p.name)]
            if latest in files and status in FINISHED and iterations:
                last = max(iterations, key=lambda p: int(p.stem.split('_')[1]))
                if files[latest].st_size == files[last].st_size:
                    candidates.append(dict(rule=RULES[2], path=str(latest), directory=str(directory),
                        duplicate_of=str(last), bytes=files[latest].st_size,
                        bytes_would_free=files[latest].st_size, savings_estimated=False,
                        reason=f'Finished trainer; latest.pt is byte-identical to {last.name}.'))
            for candidate in candidates:
                reason = _safety(candidate, roots, files, directories, links, active, opened, open_error, now)
                if errors:
                    reason = 'inventory incomplete'
                if candidate['rule'] == RULES[0]:
                    if (directory / 'manifest.json.gz').exists():
                        reason = 'both manifest forms exist'
                    candidate['bytes_would_free'] = _compression_saving(manifest, candidate['bytes'])
                if candidate['rule'] == RULES[2]:
                    # Do not read active checkpoint files merely to make a dry-run candidate.
                    if reason:
                        continue
                    if _sha(latest) != _sha(Path(candidate['duplicate_of'])):
                        continue
                candidate['blocked_reason'] = reason
                report['candidates'].append(candidate)
        for row in groups.values():
            report['total_bytes'] += row['total_bytes']
            report['birth_time_unavailable_bytes'] += row['birth_time_unavailable_bytes']
            for key, value in row['categories'].items():
                report['categories'][key] += value
            for key, value in row['growth_bytes'].items():
                report['growth_bytes'][key] += value
    report['runs'].sort(key=lambda r: (r['root'], r['run']))
    return report


def apply_rules(report, rules, out):
    """Re-inventory and recheck immediately before each selected action; append receipts."""
    if not set(rules) <= set(RULES):
        raise ValueError('Unknown cleanup rule')
    log = Path(out).resolve() / 'cleanup_log.tsv'
    if not _plain(log):
        raise ValueError('Cleanup log must not be a symlink')
    # Check that an action can be recorded before changing any run data.
    with log.open('a', encoding='utf-8'):
        pass
    roots, results = [Path(p) for p in report['roots']], []
    for candidate in sorted(report['candidates'], key=lambda c: c['rule'] != RULES[1]):
        if candidate['rule'] not in rules or candidate['blocked_reason']:
            continue
        path, directory = Path(candidate['path']), Path(candidate['directory'])
        # Never rely on a previous report to authorize deletion, or follow replaced parents.
        fresh = build_report(roots, min_free_gib=report['min_free_gib'], out=out)
        current = next((c for c in fresh['candidates'] if c['path'] == str(path)
                        and c['rule'] == candidate['rule'] and not c['blocked_reason']), None)
        if current is None or _within(Path(out).resolve(), directory):
            results.append(dict(path=str(path), rule=candidate['rule'], status='skipped after safety recheck'))
            continue
        root = next(r for r in roots if _within(path, r))
        files, directories, links, errors = _walk(root, [Path(out).resolve()])
        opened, open_error = open_paths()
        if errors or _safety(current, roots, files, directories, links, _queue_runs(roots),
                             opened, open_error, time.time()):
            results.append(dict(path=str(path), rule=candidate['rule'], status='skipped after safety recheck'))
            continue
        saved = current['bytes']
        if candidate['rule'] == RULES[0]:
            destination = path.with_name('manifest.json.gz')
            before = _stamp(path.stat())
            fd, temporary = tempfile.mkstemp(prefix='.manifest-gzip-', dir=directory)
            try:
                with os.fdopen(fd, 'wb') as binary, path.open('rb') as source:
                    with gzip.GzipFile(filename='', fileobj=binary, mode='wb', compresslevel=1, mtime=0) as target:
                        shutil.copyfileobj(source, target, length=1024**2)
                    binary.flush()
                    os.fsync(binary.fileno())
                if not _plain(path) or before != _stamp(path.stat()):
                    raise RuntimeError('Manifest changed during compression')
                # Compression can take seconds: check new activity/open files again.
                updated_files, updated_dirs, updated_links, errors = _walk(root, [Path(out).resolve()])
                updated_files.pop(Path(temporary), None)
                updated_dirs[directory] = directories[directory]  # Our temp file changed this directory's mtime.
                opened, open_error = open_paths()
                if errors or _safety(current, roots, updated_files, updated_dirs, updated_links,
                                     _queue_runs(roots), opened, open_error, time.time()):
                    results.append(dict(path=str(path), rule=candidate['rule'],
                                        status='skipped after compression safety recheck'))
                    continue
                saved -= Path(temporary).stat().st_size
                shutil.copystat(path, temporary)
                os.link(temporary, destination)  # Atomic publication, never overwrite another manifest.
                path.unlink()
            finally:
                Path(temporary).unlink(missing_ok=True)
        elif candidate['rule'] == RULES[1]:
            shutil.rmtree(path)
        else:
            path.unlink()
        timestamp = datetime.now(timezone.utc).isoformat()
        with log.open('a', encoding='utf-8') as stream:
            if stream.tell() == 0:
                stream.write('time\trule\tbytes\tpath\n')
            # JSON escaping keeps unusual file names on a single TSV line.
            stream.write(f'{timestamp}\t{candidate["rule"]}\t{saved}\t{json.dumps(str(path))}\n')
            stream.flush()
            os.fsync(stream.fileno())
        results.append(dict(path=str(path), rule=candidate['rule'], bytes=saved, time=timestamp, status='applied'))
    return results


def report_text(report):
    def size(value):
        return f'{value / GIB:.3f} GiB'
    lines = [f'Disk report {report["time"]}', report['size_note']]
    for volume in report['volumes']:
        state = 'BELOW MINIMUM' if volume['below_minimum'] else 'OK'
        lines.append(f'Volume {volume["root"]}: {size(volume["free_bytes"])} free '
                     f'(minimum {report["min_free_gib"]:g} GiB; {state})')
    lines.append(f'Total: {size(report["total_bytes"])} ({report["total_bytes"]:,} bytes)')
    lines.append('Categories: ' + ', '.join(f'{key}={size(value)}' for key, value in report['categories'].items()))
    lines.append('Created by birth time: ' + ', '.join(
        f'{days}d={size(value)}' for days, value in report['growth_bytes'].items()))
    if report['birth_time_unavailable_bytes']:
        lines.append(f'Birth time unavailable for {size(report["birth_time_unavailable_bytes"])}; excluded from growth.')
    lines.append(report['search_details_note'])
    lines.append('Runs (up to two nested folders beneath each top-level run; disjoint totals):')
    for row in report['runs']:
        nonzero = ', '.join(f'{key}={size(value)}' for key, value in row['categories'].items() if value)
        growth = '/'.join(f'{row["growth_bytes"][str(d)] / GIB:.3f}' for d in (1, 3, 7))
        prefix = row['root'] + '/' if len(report['roots']) > 1 else ''
        lines.append(f'  {prefix}{row["run"]}: {size(row["total_bytes"])}; {nonzero}; created 1/3/7d={growth} GiB')
    lines.append('Cleanup candidates (savings by rule can overlap):')
    for rule in RULES:
        candidates = [c for c in report['candidates'] if c['rule'] == rule]
        eligible = [c for c in candidates if not c['blocked_reason']]
        lines.append(f'  {rule}: {len(eligible)} eligible, {len(candidates)-len(eligible)} blocked; '
                     f'{size(sum(c["bytes_would_free"] for c in eligible))} potential savings'
                     + (' (estimated)' if rule == RULES[0] else ''))
        for c in candidates:
            status = 'BLOCKED: ' + c['blocked_reason'] if c['blocked_reason'] else 'eligible'
            lines.append(f'    {c["path"]}: {size(c["bytes_would_free"])}; {status}; {c["reason"]}')
    lines.append(f'Skipped symlinks/special files: {len(report["skipped_symlinks"])}')
    lines.extend('WARNING: ' + warning for warning in report['warnings'])
    for action in report.get('actions', []):
        lines.append(f'Action: {action["status"]}: {action["rule"]} {action["path"]}')
    lines.append('Inventory is from before cleanup.' if report.get('applied_rules') else 'Dry run: no run data changed.')
    return '\n'.join(lines) + '\n'
