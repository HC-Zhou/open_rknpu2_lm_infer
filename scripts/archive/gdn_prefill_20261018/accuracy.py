import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np

out = Path('logs/gdn-prefill-tuned')
ubatch = 128
for name in ['tokens.json', 'prompt.json', 'continuation.json']:
    shutil.copyfile(Path('logs/gdn-prefill') / name, out / name)
model = '/home/firefly/mnt/3090x4/llm/w4a8_quant/transfer/qwen35/models/C7'
bins = {'cpu': 'logs/gdn-prefill-cpu-baseline/open_rknpu2_lm_infer',
        'npu': 'target/float-autotune/release/open_rknpu2_lm_infer'}
common = ['--model', model, '--ctx-size', '640', '--ubatch-size', str(ubatch),
          '--tune-cache', '.cache/qwen35-gdn-prefill']
for kind, binary in bins.items():
    for command, args in [
        ('eval', ['--tokens-file', str(out / 'prompt.json'), '--continuation-file', str(out / 'continuation.json'),
                  '--dump-logits', str(out / (kind + '.logits.f32'))]),
        ('ppl', ['--tokens-file', str(out / 'tokens.json'), '--score-start', '257'])]:
        with (out / (kind + '-' + command + '.log')).open('w') as f:
            subprocess.run([binary, *common, command, *args, '--report', str(out / (kind + '-' + command + '.json'))],
                           stdout=f, stderr=subprocess.STDOUT, check=True,
                           pass_fds=(int(os.environ['LLM_INFER_LOCK_FD']),))
        report = json.loads((out / (kind + '-' + command + '.json')).read_text())
        stats = report['stats']
        assert stats['tune_cache_hits'] > 0 and stats['tune_cache_misses'] == 0, stats
        assert stats['tune_calls'] == 0 and stats['strategy_fallback_calls'] == 0, stats
        print('tuned accuracy complete', kind, command, flush=True)
a = np.fromfile(out / 'cpu.logits.f32', dtype='<f4').reshape(-1, 248320).astype('float64')
b = np.fromfile(out / 'npu.logits.f32', dtype='<f4').reshape(a.shape).astype('float64')
assert np.isfinite(a).all() and np.isfinite(b).all()
cpu = json.loads((out / 'cpu-ppl.json').read_text())
npu = json.loads((out / 'npu-ppl.json').read_text())
assert cpu['scored_tokens'] == npu['scored_tokens'] == 510
changes = []
for i in np.flatnonzero(a.argmax(1) != b.argmax(1)):
    ac, bc = int(a[i].argmax()), int(b[i].argmax())
    changes.append(dict(row=int(i), cpu_token=ac, npu_token=bc,
                        cpu_margin=float(a[i, ac] - a[i, bc]), npu_margin=float(b[i, bc] - b[i, ac])))
ac = a - a.mean(1, keepdims=True)
bc = b - b.mean(1, keepdims=True)
result = dict(ubatch=ubatch, logits_relative_l2=float(np.linalg.norm(a-b) / np.linalg.norm(a)),
              logits_centered_relative_l2=float(np.linalg.norm(ac-bc) / np.linalg.norm(ac)),
              logits_max_abs=float(np.abs(a-b).max()), top1_matches=int(np.count_nonzero(a.argmax(1) == b.argmax(1))),
              logit_rows=a.shape[0], changed_top1=changes, cpu_ppl=cpu['ppl'], npu_ppl=npu['ppl'],
              ppl_relative_change=npu['ppl']/cpu['ppl']-1, scored_tokens=npu['scored_tokens'])
(out / 'accuracy.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
