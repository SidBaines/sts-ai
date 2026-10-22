#!/usr/bin/env python3
"""Report RL disk use; clean up only with explicit --apply RULE options."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from sts_rl.disk_report import RULES, apply_rules, build_report, report_text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='*', type=Path, default=[ROOT / 'data/rl_runs'])
    parser.add_argument('--min-free-gib', type=float, default=30)
    parser.add_argument('--out', type=Path, help='Report directory (default: a new temporary directory)')
    parser.add_argument('--apply', action='append', choices=RULES, default=[], metavar='RULE')
    args = parser.parse_args(argv)
    if not 0 <= args.min_free_gib < float('inf'):
        parser.error('--min-free-gib must be finite and nonnegative')
    out = args.out or Path(tempfile.mkdtemp(prefix='sts-disk-report-'))
    out.mkdir(parents=True, exist_ok=True)
    try:
        report = build_report(args.roots, min_free_gib=args.min_free_gib, out=out)
        report['applied_rules'] = args.apply
        if args.apply:
            report['actions'] = apply_rules(report, args.apply, out)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    text = report_text(report)
    (out / f'disk_report_{timestamp}.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / f'disk_report_{timestamp}.txt').write_text(text)
    print(text, end='')
    print(f'Reports: {out}')
    return report


if __name__ == '__main__':
    main()
