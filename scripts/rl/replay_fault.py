#!/usr/bin/env python3
"""Bounded exact replay, or an explicitly labelled clean-backend candidate test."""
import argparse
import json
from pathlib import Path
import sys
import tempfile



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record')
    parser.add_argument('--output', required=True)
    parser.add_argument('--build-dir')
    parser.add_argument('--allow-native-change', action='store_true')
    parser.add_argument('--environment-source', help='Candidate checkout src directory; requires --allow-native-change')
    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument('--seconds', type=float, default=120.0)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Replay output already exists')
    source = Path(__file__).resolve().parents[2]/'src'
    with tempfile.TemporaryDirectory(prefix='sts-fault-replay-') as temporary:
        if args.environment_source:
            if not args.allow_native_change:
                parser.error('--environment-source requires --allow-native-change')
            candidate = Path(args.environment_source).resolve()
            if not (candidate/'sts_env').is_dir(): parser.error('Candidate source must contain sts_env')
            overlay = Path(temporary)
            for name, package in [('sts_env',candidate/'sts_env'), ('sts_policy',source/'sts_policy'), ('sts_rl',source/'sts_rl')]:
                (overlay/name).symlink_to(package,target_is_directory=True)
            sys.path.insert(0,str(overlay))
        else:
            sys.path.insert(0,str(source))
        from sts_policy.replay_fault import replay_fault
        result = replay_fault(json.loads(Path(args.record).read_text()), build_dir=args.build_dir,
                              allow_native_change=args.allow_native_change, timeout=args.timeout, seconds=args.seconds,
                              fault_dir=str(output.with_suffix('.faults')),allow_interface_change=bool(args.environment_source))
        from sts_env._backend.lightspeed_import import native_metadata
        from sts_env.provenance import interface_identity
        result['actual_native_provenance'] = native_metadata(args.build_dir)
        result['actual_interface_identity'] = interface_identity()
    result['environment_source'] = args.environment_source
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'input':str(Path(args.record).resolve()), **result}, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
