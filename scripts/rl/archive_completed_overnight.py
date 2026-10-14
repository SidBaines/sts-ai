#!/usr/bin/env python3
"""Content-addressed archive of explicitly completed overnight artifact directories."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    from sts_rl.recording import begin_run
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--directories', nargs='+', required=True)
    p.add_argument('--max-gib', type=float, default=2.)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        p.error('Fresh archive required')
    directories = [Path(name).resolve() for name in a.directories]
    files = sorted({path.resolve() for directory in directories for path in directory.rglob('*') if path.is_file()})
    total = sum(path.stat().st_size for path in files)
    if total > a.max_gib * 2**30 or shutil.disk_usage(out.parent).free - total < 6 * 2**30:
        raise RuntimeError('Archive exceeds declared size/disk reserve')
    record = begin_run(str(out) + '.record', vars(a), a.checkpoint)
    identities = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        stat = path.stat()
        identities.append({'path': str(path), 'sha256': digest, 'bytes': stat.st_size,
                           'mtime_ns': stat.st_mtime_ns})
    temporary = Path(str(out) + '.tmp')
    manifest = {'schema': 'sts_completed_overnight_archive_v1', 'created_unix': time.time(),
                'run_record': record, 'directories': [str(d) for d in directories], 'files': identities,
                'scope': 'Explicitly completed artifact directories only; local preservation, not an off-machine backup.'}
    seen = set()
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for identity in identities:
            path = Path(identity['path'])
            stat = path.stat()
            if stat.st_size != identity['bytes'] or stat.st_mtime_ns != identity['mtime_ns']:
                raise RuntimeError('Input changed after archive snapshot; do not preserve a live output')
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != identity['sha256']:
                raise RuntimeError('Input changed during archive snapshot')
            if identity['sha256'] not in seen:
                z.writestr('objects/' + identity['sha256'], content)
                seen.add(identity['sha256'])
        z.writestr('manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(temporary) as z:
        for digest in seen:
            if hashlib.sha256(z.read('objects/' + digest)).hexdigest() != digest:
                raise RuntimeError('Decompressed archive verification failed')
    temporary.replace(out)
    receipt = {'archive': str(out.resolve()), 'bytes': out.stat().st_size,
               'sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
               'files': len(files), 'unique_objects': len(seen), 'all_object_hashes_verified': True,
               'run_record': record}
    Path(str(out) + '.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
