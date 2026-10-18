"""Paired complete-model pp512 timings and bit-exact fixed-token logits.

Run from the project root under scripts/run_board.py. Both executables must
use the same C7 weights, quantized tuning cache and ordinary TF32 GDN.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline', type=Path, required=True)
parser.add_argument('--candidate', type=Path, default=Path('target/float-autotune/release/open_rknpu2_lm_infer'))
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--cache', type=Path, required=True)
parser.add_argument('--reference-root', type=Path, required=True)
parser.add_argument('--output', type=Path, default=Path('logs/gdn-boundary-dataflow/paired'))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
common = ['--model', str(args.model), '--ctx-size', '640', '--ubatch-size', '128',
          '--tune-cache', str(args.cache)]
bins = {'before': str(args.baseline), 'after': str(args.candidate)}
lock_fd = int(os.environ['LLM_INFER_LOCK_FD'])

def invoke(kind, name, command):
    report = args.output / (name + '.json')
    with report.with_suffix('.log').open('w') as log:
        subprocess.run([bins[kind], *common, *command, '--report', str(report)],
                       stdout=log, stderr=subprocess.STDOUT, check=True, pass_fds=(lock_fd,))
    return json.loads(report.read_text())

def check_cache(stats):
    assert stats['tune_cache_hits'] > 0, stats
    assert stats['tune_cache_misses'] == 0 and stats['tune_calls'] == 0, stats
    assert stats['strategy_fallback_calls'] == 0, stats

records = {kind: [] for kind in bins}
for index, kind in enumerate(['before', 'after', 'after', 'before']):
    data = invoke(kind, f'bench-{index}-{kind}', ['bench', '-p', '512', '-n', '0', '-r', '3'])
    check_cache(data['cases'][0]['stats'])
    records[kind].append(data)
    print('bench complete', index, kind, statistics.median(data['prefill']['samples_ms']), flush=True)

summary = {}
for kind, runs in records.items():
    samples = [x for run in runs for x in run['prefill']['samples_ms']]
    stages = [x['prefill'] for run in runs for x in run['cases'][0]['stage_samples']]
    summary[kind] = dict(samples_ms=samples, median_ms=statistics.median(samples),
        mean_ms=statistics.mean(samples), tps=512000 / statistics.median(samples),
        stage_median_ms={key: statistics.median(stage[key] for stage in stages) for key in stages[0]},
        stage_samples=stages)
summary['latency_change_ratio'] = summary['after']['median_ms'] / summary['before']['median_ms'] - 1
(args.output / 'performance.json').write_text(json.dumps(summary, indent=2) + '\n')

logits = {}
for kind in bins:
    output = args.output / (kind + '.logits.f32')
    data = invoke(kind, 'eval-' + kind, ['eval', '--tokens-file', str(args.reference_root / 'logs/gdn-prefill-tuned/prompt.json'),
        '--continuation-file', str(args.reference_root / 'logs/gdn-prefill-tuned/continuation.json'),
        '--dump-logits', str(output)])
    check_cache(data['stats'])
    logits[kind] = np.fromfile(output, dtype='<u4')
    print('fixed-token eval complete', kind, int(logits[kind].size), flush=True)
assert logits['before'].size == logits['after'].size == 17 * 248320, 'logit shape changed'
assert np.array_equal(logits['before'], logits['after']), 'logits changed'
assert np.isfinite(logits['after'].view('<f4')).all(), 'nonfinite logits'
accuracy = dict(bitwise_equal=True, all_finite=True, logit_elements=int(logits['before'].size), rows=17, vocab=248320)
(args.output / 'accuracy.json').write_text(json.dumps(accuracy, indent=2) + '\n')
print(json.dumps(dict(latency_change_ratio=summary['latency_change_ratio'], accuracy=accuracy)), flush=True)
