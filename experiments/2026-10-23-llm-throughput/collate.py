#!/usr/bin/env python3
"""Collate the 23-Oct K2 throughput study without modifying its run artifacts.

One timed cohort does not supply a timing confidence interval. All ratios here
are descriptive. Rerun after synchronizing cache cases to refresh pending rows.
"""
from __future__ import annotations
import argparse
import collections
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics

DEFAULT_SOURCE = 'data/llm_runs/k2_throughput_20261023/remote'
PROVENANCE = {}


def identity(path):
    path = Path(path)
    blob = path.read_bytes()
    result = {'path': str(path.resolve()), 'bytes': len(blob),
              'sha256': hashlib.sha256(blob).hexdigest(), 'mtime_ns': path.stat().st_mtime_ns}
    PROVENANCE[str(path.resolve())] = result
    return result


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    identity(path)
    return json.loads(path.read_text())


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    identity(path)
    result = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except ValueError:
            # A trailing partial write is not silently presented as a completed row.
            result.append({'_partial_or_invalid_json_line': True})
    return result


def ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator not in (None, 0) else None


def stat(values, operation):
    return operation(values) if values else None


def public_config(config):
    return {key: config.get(key) for key in ('model', 'revision', 'effort', 'dtype', 'attention',
        'gradient_checkpointing', 'microbatch', 'gradient_accumulation', 'lora_rank', 'lora_alpha',
        'lora_targets', 'temperature', 'sampling_top_p', 'max_new_tokens', 'max_sequence_length',
        'generation_microbatch', 'group_size', 'states_per_iteration', 'loss_normalization',
        'loss_token_budget', 'warmup_steps', 'benchmark_steps', 'device_name', 'trainable_parameters',
        'total_parameters', 'torch_version', 'cuda_version', 'reuse_prompt_cache',
        'static_cache_compile', 'static_cache_size', 'check_prompt_cache') if key in config}


def phase(name):
    return 'warmup' if name.startswith('warmup_') else 'calibration' if name.startswith('calibration_') else 'measurement'


def base_record(source, name, summary, config, failure, stop_request, directory):
    if summary is not None:
        status = 'completed'
        if summary.get('end_reason', 'completed') != 'completed':
            status = 'partial'
    elif failure is not None:
        status = 'oom' if 'OutOfMemory' in str(failure.get('type', failure.get('error_kind', ''))) or 'out of memory' in str(failure).lower() else 'error'
    elif directory.exists():
        status = 'partial_or_started'
    else:
        status = 'pending' if name.startswith('prefix_') else 'not_launched'
    record = {'source_dir': str(source.resolve()), 'case': name, 'designation': phase(name),
              'status': status, 'summary_path': str(directory / 'summary.json') if summary is not None else None,
              'failure_type': failure.get('type', failure.get('error_kind', failure.get('kind'))) if failure else None,
              'error': failure.get('error', failure.get('message')) if failure else None,
              'operator_stop_request_record_present': stop_request is not None,
              'operator_stop_requested': True if stop_request is not None else None,
              'cancellation_effective': False if stop_request and summary is not None and status == 'completed' else None}
    if summary is not None and summary.get('status') not in (None, 'complete', 'completed', 'success', 'passed'):
        record['status'] = summary['status']
    return record


def family_of(name, summary, config):
    if any(token in name for token in ('cache_check', 'teacher_rl_mechanics', 'resume', 'adapter', 'smoke')) or config.get('check_prompt_cache'):
        return 'mechanics'
    if name.startswith('hf_generation_') or config.get('benchmark_generation') or summary.get('purpose') == 'hf_training_generation_throughput_only':
        return 'hf_generation'
    if name.startswith('update_') or config.get('benchmark_updates') or summary.get('purpose', '').startswith('update_throughput_only'):
        return 'updates'
    if name.startswith(('fa_', 'fi_', 'warmup_', 'calibration_', 'prefix_')) or isinstance(summary.get('metrics'), dict):
        return 'serving'
    return 'mechanics'


def metrics_snapshots(source, case):
    directory = source / 'cache_metrics' / case
    before, after = directory / 'before.txt', directory / 'after.txt'
    result = {'cache_metrics_status': 'complete' if before.exists() and after.exists() else
              'before_only' if before.exists() else 'missing', 'cache_counter_deltas': {}}
    def parse(path):
        identity(path)
        totals = collections.defaultdict(float)
        for line in path.read_text().splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            match = re.match(r'^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+)', line)
            if match and 'prefix_cache' in match[1] and not match[1].endswith('_created'):
                totals[match[1]] += float(match[2])
        return totals
    b = parse(before) if before.exists() else None
    a = parse(after) if after.exists() else None
    if b is not None and a is not None:
        result['cache_counter_deltas'] = {name: a.get(name, 0.) - b.get(name, 0.) for name in sorted(set(a) | set(b))}
        result['cache_counter_reset_detected'] = any(v < 0 for v in result['cache_counter_deltas'].values())
        def pick(stem):
            for name in ('vllm:' + stem + '_total', 'vllm:' + stem):
                if name in result['cache_counter_deltas']:
                    return result['cache_counter_deltas'][name]
            return None
        result['prefix_query_delta'] = pick('prefix_cache_queries')
        result['prefix_hit_delta'] = pick('prefix_cache_hits')
        result['prefix_hit_fraction'] = ratio(result['prefix_hit_delta'], result['prefix_query_delta'])
    return result


def output_identity(source, first_case, second_case):
    a = read_jsonl(source / 'results' / first_case / 'requests.jsonl')
    b = read_jsonl(source / 'results' / second_case / 'requests.jsonl')
    if not a or not b:
        return {'identity_status': 'pending'}
    def key(row):
        return row.get('prompt_id'), row.get('repeat', 0)
    a, b = {key(r): r for r in a}, {key(r): r for r in b}
    common = set(a) & set(b)
    have_ids = [k for k in common if isinstance(a[k].get('output_token_ids'), list) and isinstance(b[k].get('output_token_ids'), list)]
    return {'identity_status': 'complete', 'first_requests': len(a), 'second_requests': len(b),
            'matched_requests': len(common), 'same_request_key_set': set(a) == set(b),
            'identical_prompt_hashes': sum(a[k].get('prompt_sha256') == b[k].get('prompt_sha256') for k in common),
            'identical_request_seeds': sum(a[k].get('request_seed') == b[k].get('request_seed') for k in common),
            'pairs_with_output_token_ids': len(have_ids),
            'identical_output_token_sequences': sum(a[k]['output_token_ids'] == b[k]['output_token_ids'] for k in have_ids),
            'identical_action_indices': sum(a[k].get('action_index') == b[k].get('action_index') for k in common)}


def gpu_samples(source):
    path = source / 'logs' / 'gpu.csv'
    if not path.exists():
        return []
    identity(path)
    samples = []
    with path.open() as stream:
        for row in csv.DictReader(stream, skipinitialspace=True):
            try:
                stamp = datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
                number = lambda key: float(re.search(r'[-+0-9.]+', row[key]).group())
                samples.append((stamp, number('memory.used [MiB]'), number('utilization.gpu [%]'), number('power.draw [W]')))
            except (ValueError, KeyError, AttributeError):
                continue
    return samples


def collate(source):
    gpu = gpu_samples(source)
    results = source / 'results'
    discovered = {p.name for p in results.iterdir() if p.is_dir()} if results.exists() else set()
    expected_hf = {f'hf_generation_{effort}_mb{batch}' for effort in ('low', 'medium', 'high') for batch in (1, 4, 8)}
    expected_updates = {'update_sdpa_gc_mb1', 'update_sdpa_gc_mb2', 'update_sdpa_gc_mb4',
                        'update_sdpa_no_gc_mb1', 'update_sdpa_no_gc_mb2', 'update_eager_gc_mb1'}
    expected_cache = {f'prefix_{prefix}_r{repeats}_medium_c32' for prefix in ('off', 'on') for repeats in (1, 4)}
    serving, hf, updates, hardware, pairs, mechanics, raw = [], [], [], [], [], [], []
    for name in sorted(discovered | expected_hf | expected_updates | expected_cache):
        directory = results / name
        summary = read_json(directory / 'summary.json')
        config = read_json(directory / 'config.json') or {}
        failure = read_json(directory / 'failure.json')
        stop_request = read_json(directory / 'operator_stop.json')
        rec = base_record(source, name, summary, config, failure, stop_request, directory)
        s = summary or {}
        raw.append({'case': name, 'source_dir': str(source.resolve()), 'summary': summary,
                    'config': config, 'failure': failure, 'operator_stop': stop_request})
        family = family_of(name, s, config)
        rec['family'] = family
        if family == 'hf_generation':
            rec.update(public_config(config))
            rec.update({key: s.get(key, config.get(key)) for key in ('purpose', 'effort', 'generation_microbatch',
                'group_size', 'timed_states', 'timed_responses', 'warmup_groups_excluded',
                'warmup_responses_per_group', 'timed_seconds', 'generated_tokens',
                'generated_tokens_per_second', 'decisions_per_second', 'valid_decisions_per_second',
                'valid_fraction', 'truncation_fraction', 'mean_reward', 'mean_group_reward_variance',
                'zero_variance_group_fraction', 'optimizer_steps', 'adapter_saved')})
            match = re.match(r'hf_generation_(low|medium|high)_mb(\d+)', name)
            if match:
                rec['effort'] = rec.get('effort') or match[1]
                rec['generation_microbatch'] = rec.get('generation_microbatch') or int(match[2])
            for flag in ('reuse_prompt_cache', 'static_cache_compile'):
                rec[flag] = s.get(flag, config.get(flag, False))
            rec['cache_flags_inferred_legacy_defaults'] = any(flag not in s and flag not in config for flag in ('reuse_prompt_cache', 'static_cache_compile'))
            rec['static_cache_size'] = s.get('static_cache_size', config.get('static_cache_size'))
            cold = s.get('cold_compile') or read_json(directory / 'cold_compile.json')
            cold_started = read_json(directory / 'cold_compile_started.json')
            rec['cold_compile'] = cold
            rec['cold_compile_status'] = 'completed' if cold else 'started_without_completion' if cold_started else 'not_requested' if not rec['static_cache_compile'] else 'pending'
            rec['cold_compile_seconds'] = cold.get('seconds') if cold else None
            lengths = s.get('response_token_lengths', [])
            rec.update(response_tokens_min=stat(lengths, min), response_tokens_mean=stat(lengths, statistics.mean),
                response_tokens_median=stat(lengths, statistics.median), response_tokens_max=stat(lengths, max),
                prompt_tokens_min=stat(s.get('prompt_token_lengths', []), min),
                prompt_tokens_max=stat(s.get('prompt_token_lengths', []), max),
                peak_allocated_GiB=ratio(s.get('peak_allocated_bytes'), 1024**3),
                peak_reserved_GiB=ratio(s.get('peak_reserved_bytes'), 1024**3),
                partial_generation_rows=sum(len(read_jsonl(p)) for p in directory.glob('generation_*.partial.jsonl')),
                completed_generation_rows=sum(len(read_jsonl(p)) for p in directory.glob('generation_*.jsonl') if '.partial.' not in p.name))
            hf.append(rec)
        elif family == 'updates':
            rec.update(public_config(config))
            rec.update({key: s.get(key, config.get(key)) for key in ('purpose', 'timed_optimizer_steps', 'warmup_steps_excluded',
                'timed_seconds', 'optimizer_steps_per_second', 'responses_per_second',
                'generated_training_tokens_per_second', 'frozen_logprob_seconds', 'loss_normalization',
                'loss_token_budget', 'adapter_saved', 'original_checkpoint_modified')})
            rec.update(peak_allocated_GiB=ratio(s.get('peak_allocated_bytes'), 1024**3),
                peak_reserved_GiB=ratio(s.get('peak_reserved_bytes'), 1024**3),
                response_token_lengths=s.get('response_token_lengths'),
                timed_generated_tokens=sum(row.get('generated_tokens', 0) for row in s.get('steps', []) if not row.get('warmup')),
                timed_responses=sum(row.get('responses', 0) for row in s.get('steps', []) if not row.get('warmup')))
            rec['timed_optimizer_steps'] = next((s[key] for key in ('timed_optimizer_steps', 'timed_steps') if key in s),
                len([row for row in s.get('steps', []) if not row.get('warmup')]) if s else None)
            rec['timed_steps_field_in_source'] = next((key for key in ('timed_optimizer_steps', 'timed_steps') if key in s), 'derived_from_steps' if s.get('steps') else None)
            rec['trainable_weights_changed'] = s.get('trainable_weights_changed')
            rec['timed_trainable_weights_changed'] = s.get('timed_trainable_weights_changed')
            updates.append(rec)
        elif family == 'mechanics':
            rec.update(public_config(config))
            rec['mechanics_kind'] = 'prompt_cache_equivalence' if 'cache_check' in name or config.get('check_prompt_cache') else 'resume' if 'resume' in name else 'adapter_reload' if 'adapter' in name else 'teacher_rl_smoke' if 'teacher_rl' in name else 'other'
            rec.update({key: value for key, value in s.items() if key not in rec})
            mechanics.append(rec)
        else:
            cfg = read_json(directory / 'server_config.json') or {}
            if isinstance(config.get('server_config'), str):
                cfg = read_json(source / config['server_config']) or cfg
            # Initial sweep omitted server config from summaries; its names identify
            # the setup and retained server logs, rather than inventing a new arm.
            if name.startswith(('fa_', 'fi_', 'warmup_fa_', 'warmup_fi_')):
                stem = name.removeprefix('warmup_')
                cfg = {**({'backend': 'FLASHINFER' if stem.startswith('fi_') else 'FLASH_ATTN',
                    'execution': 'no_graphs' if 'no_graphs' in stem else 'graphs',
                    'prefix': 'off', 'max_num_batched_tokens': 2048 if 'b2048' in stem else 8192}), **cfg}
            rec.update({key: s.get(key, cfg.get(key)) for key in ('backend', 'execution', 'prefix',
                'max_num_batched_tokens', 'max_num_seqs', 'gpu_memory_utilization', 'dtype', 'kv_cache_dtype')})
            rec.update({key: s.get(key, config.get(key)) for key in ('effort', 'concurrency', 'max_tokens',
                'repeats', 'seed', 'temperature', 'top_p', 'tokenizer_revision', 'end_reason', 'request_wall_seconds')})
            metrics = s.get('metrics', {})
            rec.update(metrics)
            if summary and (metrics.get('attempted') != metrics.get('scheduled') or metrics.get('http_ok') != metrics.get('scheduled')):
                rec['status'] = 'partial_or_completed_with_errors'
            rec.update(metrics_snapshots(source, name))
            if s.get('started_at') and s.get('ended_at'):
                start = datetime.fromisoformat(s['started_at']).timestamp()
                end = datetime.fromisoformat(s['ended_at']).timestamp()
                samples = [row for row in gpu if start <= row[0] <= end]
                rec.update(gpu_monitor_samples=len(samples),
                    gpu_peak_used_MiB=stat([r[1] for r in samples], max),
                    gpu_median_used_MiB=stat([r[1] for r in samples], statistics.median),
                    gpu_mean_utilization_percent=stat([r[2] for r in samples], statistics.mean),
                    gpu_mean_power_W=stat([r[3] for r in samples], statistics.mean))
            rec['partial_request_rows'] = len(read_jsonl(directory / 'requests.partial.jsonl'))
            rec['completed_request_rows'] = len(read_jsonl(directory / 'requests.jsonl'))
            serving.append(rec)
    for directory in sorted((source / 'hardware').glob('*')):
        path = directory / 'metrics.json'
        s = read_json(path)
        if s is None:
            continue
        hardware.append({'source_dir': str(source.resolve()), 'case': directory.name, 'status': 'completed' if not s.get('failed') else 'completed_with_errors',
            **{key: s.get(key) for key in ('num_prompts', 'max_concurrency', 'duration', 'completed', 'failed',
                'total_input_tokens', 'total_output_tokens', 'request_throughput', 'output_throughput',
                'total_token_throughput', 'mean_ttft_ms', 'median_ttft_ms', 'mean_tpot_ms')},
            'input_tokens_per_request': ratio(s.get('total_input_tokens'), s.get('completed')),
            'output_tokens_per_request': ratio(s.get('total_output_tokens'), s.get('completed')),
            'semantic_game_decisions': False})
    index = {row['case']: row for row in serving}
    for repeats in (1, 4):
        off_name, on_name = (f'prefix_{p}_r{repeats}_medium_c32' for p in ('off', 'on'))
        off, on = index.get(off_name, {}), index.get(on_name, {})
        record = {'source_dir': str(source.resolve()), 'repeats': repeats,
            'off_case': off_name, 'on_case': on_name,
            'status': 'completed' if off.get('status') == on.get('status') == 'completed' else 'pending',
            'off_status': off.get('status'), 'on_status': on.get('status'),
            'valid_decisions_per_second_on_over_off': ratio(on.get('valid_decisions_per_second'), off.get('valid_decisions_per_second')),
            'generated_tokens_per_second_on_over_off': ratio(on.get('generated_tokens_per_second'), off.get('generated_tokens_per_second')),
            'wall_seconds_off_over_on': ratio(off.get('request_wall_seconds'), on.get('request_wall_seconds')),
            'generated_token_count_on_over_off': ratio(on.get('generated_tokens'), off.get('generated_tokens')),
            'off_prefix_query_delta': off.get('prefix_query_delta'), 'on_prefix_query_delta': on.get('prefix_query_delta'),
            'off_prefix_hit_delta': off.get('prefix_hit_delta'), 'on_prefix_hit_delta': on.get('prefix_hit_delta'),
            'off_prefix_hit_fraction': off.get('prefix_hit_fraction'), 'on_prefix_hit_fraction': on.get('prefix_hit_fraction'),
            'interpretation': 'Descriptive paired natural-output throughput; no timing CI. Compare output identity and token counts before attributing differences to caching.'}
        record.update(output_identity(source, off_name, on_name))
        pairs.append(record)
    comparisons = []
    for first, second in (('fa_graphs_b8192_medium_c32', 'fa_graphs_b8192_repeat_medium_c32'),
                          ('fa_graphs_b8192_medium_c32', 'fi_graphs_b8192_medium_c32'),
                          ('fa_graphs_b8192_high_c32', 'fi_graphs_b8192_high_c32')):
        a, b = index.get(first, {}), index.get(second, {})
        comparisons.append({'source_dir': str(source.resolve()), 'first': first, 'second': second,
            'valid_decisions_per_second_second_over_first': ratio(b.get('valid_decisions_per_second'), a.get('valid_decisions_per_second')),
            'tokens_per_second_second_over_first': ratio(b.get('generated_tokens_per_second'), a.get('generated_tokens_per_second')),
            'generated_token_count_second_over_first': ratio(b.get('generated_tokens'), a.get('generated_tokens')),
            'wall_seconds_second_over_first': ratio(b.get('request_wall_seconds'), a.get('request_wall_seconds')),
            **output_identity(source, first, second)})
    h = {r['case']: r for r in hardware}
    reference = h.get('fa_graphs_b8192', {})
    for row in hardware:
        row['output_throughput_over_fa_graphs_b8192'] = ratio(row.get('output_throughput'), reference.get('output_throughput'))
    return {'serving': serving, 'hf_generation': hf, 'updates': updates, 'hardware': hardware,
            'cache_pairs': pairs, 'mechanics': mechanics, 'natural_comparisons': comparisons, 'raw_records': raw}


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(',', ':')) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', help='Canonical remote snapshot directory; repeatable')
    parser.add_argument('--output', default='data/llm_runs/k2_throughput_20261023/analysis_complete')
    args = parser.parse_args()
    sources = [Path(p).resolve() for p in args.source or [DEFAULT_SOURCE]]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data = collections.defaultdict(list)
    for source in sources:
        for key, records in collate(source).items():
            data[key].extend(records)
    for name in ('serving', 'hf_generation', 'updates', 'hardware', 'cache_pairs', 'mechanics', 'natural_comparisons'):
        write_csv(output / (name + '.csv'), data[name])
    document = {'schema': 'sts_k2_throughput_collation_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source_dirs': [str(s) for s in sources],
        'timing_inference': 'Descriptive measurements only; no confidence intervals from a single timed cohort.',
        'status_note': 'Expected but absent HF cells are not_launched; absent cache cases remain pending. Stop requests do not override completed summaries.',
        'records': data, 'source_file_identities': list(PROVENANCE.values()), 'collator': identity(Path(__file__))}
    (output / 'collated.json').write_text(json.dumps(document, indent=2) + '\n')
    (output / 'README.md').write_text('''# K2 throughput collation\n\nGenerated by collate.py from the canonical source directories recorded in collated.json. Rerun after synchronizing new cache cases. Source artifacts are read only.\n\n- serving.csv includes every serving measurement, calibration and warmup, plus pending cache arms.\n- hf_generation.csv includes completed and unlaunched HF cells and retained partial artifact counts.\n- updates.csv includes every optimizer cell, including OOM.\n- hardware.csv holds fixed equal-token probes: these are not game decisions.\n- mechanics.csv isolates cache-equivalence, teacher-RL and adapter/resume smoke checks from serving measurements.\n- cache_pairs.csv pairs on/off for one and four repetitions, including before/after prefix-query/hit counter deltas and exact prompt/seed/output identity checks when both arms exist.\n- natural_comparisons.csv describes backend and repeat differences; differences in generated work are explicit.\n- collated.json preserves original summaries/configuration/failures and source file identities.\n\nRatios are descriptive. Single timed cohorts do not justify timing confidence intervals; natural-output comparisons also vary in completion length. Partial files left by completed cells are retained artifacts, not additional completed measurements. No absent cache result is presented as evidence.\n''')
    print(json.dumps({'output': str(output.resolve()), 'counts': {key: len(value) for key, value in data.items() if key != 'raw_records'}}))


if __name__ == '__main__':
    main()
