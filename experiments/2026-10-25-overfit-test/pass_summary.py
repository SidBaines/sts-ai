"""Per-pass averages (8 iterations = one pass over the 32 states) from a run's iterations.jsonl.
Usage: python3 pass_summary.py RUN_DIR/iterations.jsonl
"""
import json, sys
rs = [json.loads(l) for l in open(sys.argv[1])]
for k in range(0, len(rs), 8):
    ch = rs[k:k + 8]
    if len(ch) < 8: break
    def w(key, ph='all'):
        xs = [r[ph] for r in ch if r[ph].get('decisions') and key in r[ph]]
        return sum(x[key] * x['decisions'] for x in xs) / sum(x['decisions'] for x in xs)
    print('pass', k // 8 + 1, 'p=%.3f top1=%.3f A=%.2f valid=%.3f med=%.0f s/it=%.0f | combat p=%.3f ooc p=%.3f' % (
        w('mean_teacher_probability'), w('teacher_top_agreement'), w('first_displayed_option_rate'), w('valid_fraction'),
        sum(r['all']['median_reasoning_tokens'] for r in ch) / 8, sum(r['iteration_seconds'] for r in ch) / 8,
        w('mean_teacher_probability', 'combat'), w('mean_teacher_probability', 'out_of_combat')))
