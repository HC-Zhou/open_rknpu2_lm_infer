"""C7 pp512 CPU-processing profile and exact-logit comparison.

Run on the board from the repository root under scripts/run_board.py. Requires
the pre-scan-removal executable at logs/gdn-cpu-breakdown/before, the current
target/float-autotune release executable, and the tuned cache/reference logits
from the preceding GDN experiment. Uses ubatch128 and before/after/after/before.
"""

import json
import os
from pathlib import Path
import statistics
import subprocess
import numpy as np

out = Path('logs/gdn-cpu-breakdown')
common = ['--model', '/home/firefly/mnt/3090x4/llm/w4a8_quant/transfer/qwen35/models/C7',
          '--ctx-size', '640', '--ubatch-size', '128', '--tune-cache', '.cache/qwen35-gdn-prefill']
bins = {'before': str(out / 'before'), 'after': 'target/float-autotune/release/open_rknpu2_lm_infer'}
runs = {k: [] for k in bins}
for i, kind in enumerate(['before', 'after', 'after', 'before']):
    report = out / f'bench-{i}-{kind}.json'
    with report.with_suffix('.log').open('w') as f:
        subprocess.run([bins[kind], *common, 'bench', '-p', '512', '-n', '0', '-r', '3',
                        '--report', str(report)], stdout=f, stderr=subprocess.STDOUT,
                       check=True, pass_fds=(int(os.environ['LLM_INFER_LOCK_FD']),))
    data = json.loads(report.read_text())
    stats = data['cases'][0]['stats']
    assert stats['tune_cache_hits'] > 0 and stats['tune_cache_misses'] == 0, stats
    assert stats['tune_calls'] == 0 and stats['strategy_fallback_calls'] == 0, stats
    runs[kind].append(data)
    print('profile bench complete', i, kind, statistics.median(data['prefill']['samples_ms']), flush=True)
summary = {}
for kind, records in runs.items():
    samples = [x for r in records for x in r['prefill']['samples_ms']]
    stages = [x['prefill'] for r in records for x in r['cases'][0]['stage_samples']]
    summary[kind] = dict(prefill_median_ms=statistics.median(samples), samples_ms=samples,
                         tps=512000/statistics.median(samples),
                         stage_median_ms={k: statistics.median(s[k] for s in stages) for k in stages[0]},
                         stage_samples=stages)
    if kind == 'after':
        fields = ['gdn_split_ms', 'gdn_layer_prepare_ms', 'gdn_buffers_ms', 'gdn_gates_ms',
                  'gdn_output_norm_ms', 'gdn_output_copy_ms', 'gdn_npu_ms']
        summary[kind]['gdn_remainder_ms'] = [s['gdn_ms']-sum(s[k] for k in fields) for s in stages]
(out / 'performance.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: {x:v for x,v in r.items() if x != 'stage_samples'} for k,r in summary.items()}, indent=2), flush=True)
report = out / 'after-eval.json'
with report.with_suffix('.log').open('w') as f:
    subprocess.run([bins['after'], *common, 'eval', '--tokens-file', 'logs/gdn-prefill-tuned/prompt.json',
                    '--continuation-file', 'logs/gdn-prefill-tuned/continuation.json',
                    '--dump-logits', str(out / 'after.logits.f32'), '--report', str(report)],
                   stdout=f, stderr=subprocess.STDOUT, check=True,
                   pass_fds=(int(os.environ['LLM_INFER_LOCK_FD']),))
a = np.fromfile('logs/gdn-prefill-tuned/npu.logits.f32', dtype='<u4')
b = np.fromfile(out / 'after.logits.f32', dtype='<u4')
assert a.size == b.size and np.array_equal(a, b), 'logits changed after removing scans'
s = json.loads(report.read_text())['stats']
assert s['tune_cache_misses'] == 0 and s['strategy_fallback_calls'] == 0, s
accuracy = dict(logit_elements=int(a.size), bitwise_equal=True, rows=17, vocab=248320)
(out / 'accuracy.json').write_text(json.dumps(accuracy, indent=2) + '\n')
print('accuracy', json.dumps(accuracy), flush=True)
