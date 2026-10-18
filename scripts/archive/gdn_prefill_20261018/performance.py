import json
import os
from pathlib import Path
import statistics
import subprocess

out = Path('logs/gdn-prefill-tuned')
model = '/home/firefly/mnt/3090x4/llm/w4a8_quant/transfer/qwen35/models/C7'
binaries = {'cpu': 'logs/gdn-prefill-cpu-baseline/open_rknpu2_lm_infer',
            'npu': 'target/float-autotune/release/open_rknpu2_lm_infer'}
order = [('cpu', 128), ('npu', 128), ('npu', 256),
         ('npu', 256), ('npu', 128), ('cpu', 128)]
runs = {}
for i, (kind, ubatch) in enumerate(order):
    report = out / f'bench-{i}-{kind}-{ubatch}.json'
    with report.with_suffix('.log').open('w') as f:
        subprocess.run([binaries[kind], '--model', model, '--ctx-size', '640',
                        '--ubatch-size', str(ubatch), '--tune-cache', '.cache/qwen35-gdn-prefill',
                        'bench', '-p', '12,13,128,512', '-n', '0', '-pg', '128,16',
                        '-r', '3', '--report', str(report)], stdout=f, stderr=subprocess.STDOUT,
                       check=True, pass_fds=(int(os.environ['LLM_INFER_LOCK_FD']),))
    data = json.loads(report.read_text())
    for case in data['cases']:
        stats = case['stats']
        assert stats['tune_cache_hits'] > 0, (i, kind, ubatch, stats)
        assert stats['tune_cache_misses'] == 0, (i, kind, ubatch, stats)
        assert stats['tune_calls'] == 0 and stats['strategy_fallback_calls'] == 0, (i, kind, ubatch, stats)
    runs.setdefault(f'{kind}{ubatch}', []).append(data)
    pp = next(c for c in data['cases'] if c['prompt'] == 512 and c['generate'] == 0)
    print('tuned bench complete', i, kind, ubatch, 'pp512_ms', statistics.median(pp['prefill']['samples_ms']),
          'cache_hits', pp['stats']['tune_cache_hits'], flush=True)
summary = []
for index, case in enumerate(runs['cpu128'][0]['cases']):
    row = {k: case[k] for k in ('prompt', 'generate', 'depth')}
    for key, data in runs.items():
        samples = [x for run in data for x in run['cases'][index]['prefill']['samples_ms']]
        row[key] = {'prefill_median_ms': statistics.median(samples), 'samples_ms': samples,
                    'stage_samples': [x for run in data for x in run['cases'][index]['stage_samples']],
                    'stats': [run['cases'][index]['stats'] for run in data]}
        if case['generate']:
            samples = [x for run in data for x in run['cases'][index]['decode']['samples_ms']]
            row[key]['decode_median_ms'] = statistics.median(samples)
            row[key]['decode_samples_ms'] = samples
    summary.append(row)
pp = next(r for r in summary if r['prompt'] == 512 and r['generate'] == 0)
selected = min([128, 256], key=lambda m: pp[f'npu{m}']['prefill_median_ms'])
(out / 'performance.json').write_text(json.dumps({'fastest_median_ubatch': selected, 'default_ubatch': 128, 'cases': summary}, indent=2) + '\n')
print('selected ubatch', selected, flush=True)
for key in pp:
    if key.startswith(('cpu', 'npu')):
        print(key, 'pp512_ms', pp[key]['prefill_median_ms'],
              'tps', 512000 / pp[key]['prefill_median_ms'], flush=True)
