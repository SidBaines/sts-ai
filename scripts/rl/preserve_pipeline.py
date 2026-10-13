#!/usr/bin/env python3
"""Make a bounded, portable, checksum-verified archive of recorded training inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


def preserve(audit_path, destination, *, max_gib=6.0, reserve_gib=6.0):
    audit = json.loads(Path(audit_path).read_text())
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Archive destination already exists')
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    temporary = destination.with_suffix('.partial')
    if temporary.exists():
        raise ValueError('Partial archive exists; preserve it and choose a new destination')
    files = {Path(audit_path).resolve()}
    for row in audit['original_ancestry']:
        path = Path(row['path']); files.add(path)
        files.update(p for p in [path.with_suffix('.metrics.json'),path.with_suffix('.log')] if p.exists())
        for value in row.get('data_inputs', {}).values():
            data = Path(value['path']); files.add(data)
            files.update(data.parent.glob('*.json'))
    live = Path(audit['locations']['rl_debug_artifacts'])
    files.update(p for p in live.rglob('*') if p.is_file() and p.suffix in {'.pt','.json','.jsonl','.log'}
                 and p.name!='runner_status.json')
    for record in audit['current_native_builds']:
        files.add(Path(record['binary']['path'])); files.add(Path(record['manifest_file']['path']))
    original_dir = Path(audit['locations']['original_checkpoint']).parents[3]/'external/sts_lightspeed_v011/build'
    if original_dir.exists():
        files.update(original_dir.glob('*.so')); files.update(original_dir.glob('build_manifest.json'))
    files.update(p for section in ['src','scripts','tests','docs','patches'] for p in (root/section).rglob('*')
                 if p.is_file() and p.suffix in {'.py','.json','.md','.sh','.patch'} and '__pycache__' not in p.parts)
    files.update(root/p for p in ['README.md','pyproject.toml','CLAUDE.md'] if (root/p).exists())
    git_bundle = destination.with_suffix('.git.bundle')
    subprocess.run(['git','bundle','create',str(git_bundle),'--all'], cwd=root, check=True, capture_output=True)
    files.add(git_bundle)
    manifest = {'schema':'sts_training_preservation_v1','files':[],
                'scope':'Original selected-model ancestry, its recorded data inputs/metrics/logs, all RL-debug runs, source/docs and Git history.',
                'limitations':'Local archive, not an off-machine backup. Historical missing source/RNG/fault evidence cannot be reconstructed by archiving.'}
    known = {}
    try:
        with zipfile.ZipFile(temporary,'x',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
            for path in sorted(files):
                if shutil.disk_usage(destination.parent).free < reserve_gib * 2**30:
                    raise RuntimeError('Disk reserve reached; partial archive preserved')
                digest = hashlib.sha256()
                with path.open('rb') as source:
                    for chunk in iter(lambda:source.read(1<<20),b''): digest.update(chunk)
                identity = digest.hexdigest()
                name = 'objects/'+identity
                if identity not in known:
                    archive.write(path,name)
                    known[identity]=name
                manifest['files'].append({'original_path':str(path.resolve()), 'member':name,
                                          'sha256':identity,'bytes':path.stat().st_size})
                if temporary.stat().st_size > max_gib * 2**30:
                    raise RuntimeError('Archive size budget reached; partial archive preserved')
            archive.writestr('manifest.json',json.dumps(manifest,indent=2)+'\n')
        # Read every archived object and check content against its identity.
        with zipfile.ZipFile(temporary) as archive:
            for identity,name in known.items():
                digest=hashlib.sha256()
                with archive.open(name) as source:
                    for chunk in iter(lambda:source.read(1<<20),b''): digest.update(chunk)
                if digest.hexdigest()!=identity: raise RuntimeError(f'Archive content mismatch: {name}')
        temporary.rename(destination)
        digest=hashlib.sha256()
        with destination.open('rb') as source:
            for chunk in iter(lambda:source.read(1<<20),b''): digest.update(chunk)
        receipt={'archive':str(destination),'sha256':digest.hexdigest(),'bytes':destination.stat().st_size,
                 'source_files':len(manifest['files']),'unique_objects':len(known),'all_object_hashes_verified':True,
                 'free_bytes_after':shutil.disk_usage(destination.parent).free}
        destination.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return receipt
    finally:
        # The Git bundle's bytes are inside the verified archive on success.
        if destination.exists(): git_bundle.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',required=True)
    parser.add_argument('--destination',required=True)
    parser.add_argument('--max-gib',type=float,default=6)
    parser.add_argument('--reserve-gib',type=float,default=6)
    args=parser.parse_args()
    print(json.dumps(preserve(args.audit,args.destination,max_gib=args.max_gib,reserve_gib=args.reserve_gib)))


if __name__=='__main__': main()
